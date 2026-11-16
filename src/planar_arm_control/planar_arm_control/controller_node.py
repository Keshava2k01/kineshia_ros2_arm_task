#!/usr/bin/env python3
"""
controller_node.py  —  owns the arm state, plans and executes trajectories.

Interfaces
    Publishes  /joint_states    sensor_msgs/JointState   at publish_rate_hz
    Publishes  /controller_status planar_arm_interfaces/msg/ControllerStatus @ 10 Hz
    Publishes  /tracking_error  planar_arm_interfaces/msg/TrackingError at publish_rate_hz
    Service    /move_to_target  planar_arm_interfaces/srv/MoveToTarget
    Service    /stop            std_srvs/Trigger  (smooth deceleration to rest)

Parameters
    publish_rate_hz          control/publish loop rate            [Hz]   50.0
    max_joint_velocity       per-joint velocity limit             [rad/s] 1.0
    max_joint_acceleration   per-joint acceleration limit         [rad/s^2] 2.0
    min_duration             shortest allowed move                [s]    0.5
    initial_joint_positions  start pose                           [rad]  [pi/2, 0, 0]
    joint_names              names in JointState                  [joint1..3]
    backend                  hardware backend ("sim")
    control_mode             "position" (ideal servo) | "pid" (torque PID on dynamics)
    pid_bandwidth            PID closed-loop natural frequency    [rad/s] 10.0
    pid_damping              PID damping ratio                            1.0
    pid_integral_ratio       Ki = Kp * bandwidth * ratio                  0.3
    settle_tolerance         max |error| to declare a move done   [rad]  0.001
    settle_timeout           give up settling after               [s]    2.0

Data flow
    service ──► MotionPlanner ──► active Trajectory
    timer   ──► sample(t) ──► backend.write_command ──► backend.advance(dt)
                          ──► backend.read_state ──► /joint_states, /tracking_error

A move is complete when its trajectory has ended AND the measured joints are
within settle_tolerance of the goal (immediately in position mode; after the
PID settles in pid mode) — like a servo's "in-position" flag.

Threading
    The timer and the service run in separate mutually-exclusive callback
    groups on a MultiThreadedExecutor. IK + trajectory validation typically
    takes ~5-10 ms, but the provided IK's Jacobian fallback can run for up to
    1000 iterations; on a single-threaded executor that would stall the
    control loop and cause a gap in /joint_states. Shared state (the
    active trajectory and its start time) is guarded by a lock, and the lock
    is NOT held while planning.
"""

import threading

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from planar_arm_interfaces.msg import ControllerStatus, TrackingError
from planar_arm_interfaces.srv import MoveToTarget

# The provided kinematics library — do not modify it.
from planar_arm_control.planar_arm import PlanarArm
from planar_arm_control.backends import make_backend
from planar_arm_control.planner import MotionPlanner

LINK_LENGTHS = [3.0, 2.0, 1.5]


class ControllerNode(Node):

    def __init__(self):
        super().__init__("controller_node")
        self.arm = PlanarArm(LINK_LENGTHS)

        # ---------------- parameters ----------------
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("max_joint_velocity", 1.0)
        self.declare_parameter("max_joint_acceleration", 2.0)
        self.declare_parameter("min_duration", 0.5)
        self.declare_parameter("initial_joint_positions", [np.pi / 2, 0.0, 0.0])
        self.declare_parameter("joint_names", ["joint1", "joint2", "joint3"])
        self.declare_parameter("backend", "sim")
        self.declare_parameter("control_mode", "position")
        self.declare_parameter("pid_bandwidth", 10.0)
        self.declare_parameter("pid_damping", 1.0)
        self.declare_parameter("pid_integral_ratio", 0.3)
        self.declare_parameter("settle_tolerance", 0.001)
        self.declare_parameter("settle_timeout", 2.0)

        rate = self._param("publish_rate_hz")
        if rate <= 0.0:
            raise ValueError("publish_rate_hz must be > 0")
        self.joint_names = list(self._param("joint_names"))
        q_init = np.asarray(self._param("initial_joint_positions"), dtype=float)
        if len(q_init) != self.arm.n_joints or len(self.joint_names) != self.arm.n_joints:
            raise ValueError("initial_joint_positions/joint_names must have 3 entries")

        self.planner = MotionPlanner(
            self.arm,
            v_max=self._param("max_joint_velocity"),
            a_max=self._param("max_joint_acceleration"),
            t_min=self._param("min_duration"),
            ready_pose=(np.pi / 2, 0.0, 0.0),
        )
        if not self.planner.pose_valid(q_init):
            raise ValueError(f"initial_joint_positions {q_init.tolist()} violate "
                             "joint limits or the ground constraint")

        self.control_mode = self._param("control_mode")
        pid_kwargs = {}
        if self.control_mode == "pid":
            pid_kwargs = dict(bandwidth=self._param("pid_bandwidth"),
                              damping=self._param("pid_damping"),
                              integral_ratio=self._param("pid_integral_ratio"))
        self.backend = make_backend(self._param("backend"), self.control_mode,
                                    q_init, arm=self.arm, **pid_kwargs)
        self.settle_tolerance = self._param("settle_tolerance")
        self.settle_timeout = self._param("settle_timeout")

        # Trajectory timing runs on a STEADY clock: if the wall clock is
        # stepped (NTP, WSL resync after a pause) a ROS-time trajectory would
        # jump along its path. Message stamps stay on ROS time. Under
        # use_sim_time (e.g. Gazebo) we must follow the simulation clock.
        if self.get_parameter("use_sim_time").value:
            self._timing_clock = self.get_clock()
        else:
            self._timing_clock = Clock(clock_type=ClockType.STEADY_TIME)

        # ---------------- shared state (guarded by _lock) ----------------
        self._lock = threading.Lock()
        self._traj = None          # active Trajectory or None when holding
        self._t_start = 0.0        # node-clock time the trajectory started [s]
        self._q_cmd = q_init.copy()
        self._qd_cmd = np.zeros_like(q_init)
        ee = self.arm.end_effector(q_init)
        self._goal_xy = (float(ee[0]), float(ee[1]))
        self._projected = False
        self._last_message = "Ready."
        # Incremented for every accepted command (move or stop). Clients wait
        # for status.command_id == their id && IDLE, which avoids mistaking a
        # stale IDLE (published just before their command) for completion.
        self._command_id = 0
        self._settle_start = None   # time the active trajectory ended [s]
        self._last_tick = None      # for the plant's dt (steady clock)

        # ---------------- ROS interfaces ----------------
        self._control_group = MutuallyExclusiveCallbackGroup()
        self._service_group = MutuallyExclusiveCallbackGroup()

        self.joint_pub = self.create_publisher(JointState, "joint_states", 10)
        self.status_pub = self.create_publisher(
            ControllerStatus, "controller_status", 10)
        self.tracking_pub = self.create_publisher(TrackingError, "tracking_error", 10)
        self.move_srv = self.create_service(
            MoveToTarget, "move_to_target", self._on_move_to_target,
            callback_group=self._service_group)
        self.stop_srv = self.create_service(
            Trigger, "stop", self._on_stop, callback_group=self._service_group)
        # The timers stay on the node clock: measured on Humble, a timer on
        # the STEADY clock fired ~1.4 ms late every period (46.6 Hz instead
        # of 50), while the ROS-time timer holds 50.0 Hz. Only trajectory
        # sampling needs to be immune to clock steps.
        self.timer = self.create_timer(
            1.0 / rate, self._on_control_tick, callback_group=self._control_group)
        # Low-rate status for GUIs/sequencers (multi-rate timers, same group).
        self.status_timer = self.create_timer(
            0.1, self._on_status_tick, callback_group=self._control_group)

        self.get_logger().info(
            f"controller_node ready: backend={self.backend.name} "
            f"(control_mode={self.control_mode}), "
            f"rate={rate:.0f} Hz, q0={np.degrees(q_init).round(1).tolist()} deg")

    # ------------------------------------------------------------------
    def _param(self, name):
        return self.get_parameter(name).value

    def _now(self):
        """Trajectory time [s] (steady clock unless use_sim_time)."""
        return self._timing_clock.now().nanoseconds * 1e-9

    def _desired_state(self, t):
        """Reference (q, qd, qdd) at time t. Caller must hold _lock."""
        if self._traj is None:
            zero = np.zeros_like(self._q_cmd)
            return self._q_cmd.copy(), zero, zero.copy()
        return self._traj.sample(t - self._t_start)

    # ------------------------------------------------------------------
    # Control loop (fixed rate)
    # ------------------------------------------------------------------
    def _on_control_tick(self):
        now = self.get_clock().now()        # for the message stamps
        t = self._now()                     # for trajectory sampling / dt
        with self._lock:
            traj = self._traj
            q, qd, qdd = self._desired_state(t)
            ended = traj is not None and t - self._t_start >= traj.duration
            self._q_cmd, self._qd_cmd = q, qd

        dt = 0.0 if self._last_tick is None else min(max(t - self._last_tick, 0.0), 0.1)
        self._last_tick = t
        self.backend.write_command(q, qd, qdd)
        self.backend.advance(dt)
        q_meas, qd_meas, effort = self.backend.read_state()
        error = q - q_meas

        if ended:
            self._check_settled(traj, t, error, q_meas)

        msg = JointState()
        msg.header.stamp = now.to_msg()
        msg.name = self.joint_names
        msg.position = q_meas.tolist()
        msg.velocity = qd_meas.tolist()
        msg.effort = effort.tolist()
        self.joint_pub.publish(msg)

        track = TrackingError()
        track.header.stamp = msg.header.stamp
        track.reference = q.tolist()
        track.measured = q_meas.tolist()
        track.error = error.tolist()
        track.effort = effort.tolist()
        self.tracking_pub.publish(track)

    def _check_settled(self, traj, t, error, q_meas):
        """Retire a finished trajectory once the joints are in position."""
        with self._lock:
            if self._traj is not traj:          # replaced meanwhile
                return
            if self._settle_start is None:
                self._settle_start = t
            err = float(np.max(np.abs(error)))
            settled = err < self.settle_tolerance
            timed_out = t - self._settle_start > self.settle_timeout
            if not (settled or timed_out):
                return
            self._traj = None
            self._settle_start = None
        ee = self.arm.end_effector(q_meas)
        text = (f"q = {np.degrees(q_meas).round(2).tolist()} deg, "
                f"ee = ({ee[0]:.3f}, {ee[1]:.3f}), |e|max = {np.degrees(err):.4f} deg")
        if settled:
            self.get_logger().info(f"Goal reached: {text}")
        else:
            self.get_logger().warn(f"Settle timeout ({self.settle_timeout:.1f}s): {text}")

    def _on_status_tick(self):
        now = self.get_clock().now()
        t = self._now()
        msg = ControllerStatus()
        msg.header.stamp = now.to_msg()
        msg.control_mode = self.control_mode
        with self._lock:
            moving = self._traj is not None
            settling = moving and t - self._t_start >= self._traj.duration
            msg.state = "SETTLING" if settling else ("MOVING" if moving else "IDLE")
            msg.progress = (min(1.0, (t - self._t_start) / self._traj.duration)
                            if moving else 1.0)
            msg.goal_x, msg.goal_y = self._goal_xy
            msg.projected = self._projected
            msg.message = self._last_message
            msg.command_id = self._command_id
        self.status_pub.publish(msg)

    # ------------------------------------------------------------------
    # Service: move to a Cartesian target
    # ------------------------------------------------------------------
    def _on_move_to_target(self, request, response):
        # Snapshot the commanded state at t_snap. The new trajectory starts at
        # t_snap from exactly that position/velocity, so when it replaces the
        # old one a few ms later the hand-over is continuous to first order.
        with self._lock:
            t_snap = self._now()
            q0, qd0, _ = self._desired_state(t_snap)

        result = self.planner.plan(q0, qd0, (request.x, request.y),
                                   duration=request.duration)

        with self._lock:
            self._last_message = result.message
            if result.success:
                self._command_id += 1
                response.command_id = self._command_id
                self._traj = result.trajectory
                self._t_start = t_snap
                self._settle_start = None
                self._goal_xy = result.goal_xy
                self._projected = result.projected

        response.success = result.success
        response.projected = result.projected
        response.goal_x, response.goal_y = result.goal_xy
        response.joint_goal = ([] if result.q_goal is None
                               else [float(v) for v in result.q_goal])
        response.duration = float(result.duration)
        response.message = result.message

        # Note: rclpy forbids changing severity at the same call site, so the
        # two log levels need separate calls.
        prefix = f"move_to_target({request.x:.3f}, {request.y:.3f}) -> "
        if result.success:
            self.get_logger().info(
                f"{prefix}OK: {result.message} [{result.duration:.2f}s]")
        else:
            self.get_logger().warn(f"{prefix}REJECTED: {result.message}")
        return response

    # ------------------------------------------------------------------
    # Service: smooth stop (used by cancel)
    # ------------------------------------------------------------------
    def _on_stop(self, request, response):
        with self._lock:
            t_snap = self._now()
            q0, qd0, _ = self._desired_state(t_snap)
        result = self.planner.plan_stop(q0, qd0)
        with self._lock:
            self._command_id += 1
            self._traj = result.trajectory
            self._t_start = t_snap
            self._settle_start = None
            ee = self.arm.end_effector(result.q_goal)
            self._goal_xy = (float(ee[0]), float(ee[1]))
            self._projected = False
            self._last_message = result.message
        response.success = True
        response.message = result.message
        self.get_logger().info(f"stop -> {result.message} [{result.duration:.2f}s]")
        return response

    def destroy_node(self):
        self.backend.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
