#!/usr/bin/env python3
"""
controller_node.py  —  owns the arm state, plans and executes trajectories.

Interfaces
    Publishes  /joint_states    sensor_msgs/JointState   at publish_rate_hz
    Service    /move_to_target  planar_arm_interfaces/srv/MoveToTarget

Parameters
    publish_rate_hz          control/publish loop rate            [Hz]   50.0
    max_joint_velocity       per-joint velocity limit             [rad/s] 1.0
    max_joint_acceleration   per-joint acceleration limit         [rad/s^2] 2.0
    min_duration             shortest allowed move                [s]    0.5
    initial_joint_positions  start pose                           [rad]  [pi/2, 0, 0]
    joint_names              names in JointState                  [joint1..3]
    backend                  hardware backend ("sim")

Data flow
    service ──► MotionPlanner ──► active Trajectory
    timer   ──► sample(t) ──► backend.write_command ──► backend.read_state
                                                    ──► /joint_states

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
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

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

        self.backend = make_backend(self._param("backend"), q_init)

        # ---------------- shared state (guarded by _lock) ----------------
        self._lock = threading.Lock()
        self._traj = None          # active Trajectory or None when holding
        self._t_start = 0.0        # node-clock time the trajectory started [s]
        self._q_cmd = q_init.copy()
        self._qd_cmd = np.zeros_like(q_init)

        # ---------------- ROS interfaces ----------------
        self._control_group = MutuallyExclusiveCallbackGroup()
        self._service_group = MutuallyExclusiveCallbackGroup()

        self.joint_pub = self.create_publisher(JointState, "joint_states", 10)
        self.move_srv = self.create_service(
            MoveToTarget, "move_to_target", self._on_move_to_target,
            callback_group=self._service_group)
        self.timer = self.create_timer(
            1.0 / rate, self._on_control_tick, callback_group=self._control_group)

        self.get_logger().info(
            f"controller_node ready: backend={self.backend.name}, "
            f"rate={rate:.0f} Hz, q0={np.degrees(q_init).round(1).tolist()} deg")

    # ------------------------------------------------------------------
    def _param(self, name):
        return self.get_parameter(name).value

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _desired_state(self, t):
        """Commanded (q, qd) at time t. Caller must hold _lock."""
        if self._traj is None:
            return self._q_cmd.copy(), np.zeros_like(self._q_cmd)
        q, qd, _ = self._traj.sample(t - self._t_start)
        return q, qd

    # ------------------------------------------------------------------
    # Control loop (fixed rate)
    # ------------------------------------------------------------------
    def _on_control_tick(self):
        now = self.get_clock().now()
        t = now.nanoseconds * 1e-9
        with self._lock:
            q, qd = self._desired_state(t)
            if self._traj is not None and t - self._t_start >= self._traj.duration:
                self._traj = None
                qd = np.zeros_like(q)
                self.get_logger().info(
                    "Goal reached: q = "
                    f"{np.degrees(q).round(2).tolist()} deg, ee = "
                    f"({self.arm.end_effector(q)[0]:.3f}, "
                    f"{self.arm.end_effector(q)[1]:.3f})")
            self._q_cmd, self._qd_cmd = q, qd

        self.backend.write_command(q, qd)
        q_meas, qd_meas, effort = self.backend.read_state()

        msg = JointState()
        msg.header.stamp = now.to_msg()
        msg.name = self.joint_names
        msg.position = q_meas.tolist()
        msg.velocity = qd_meas.tolist()
        msg.effort = effort.tolist()
        self.joint_pub.publish(msg)

    # ------------------------------------------------------------------
    # Service: move to a Cartesian target
    # ------------------------------------------------------------------
    def _on_move_to_target(self, request, response):
        # Snapshot the commanded state at t_snap. The new trajectory starts at
        # t_snap from exactly that position/velocity, so when it replaces the
        # old one a few ms later the hand-over is continuous to first order.
        with self._lock:
            t_snap = self._now()
            q0, qd0 = self._desired_state(t_snap)

        result = self.planner.plan(q0, qd0, (request.x, request.y),
                                   duration=request.duration)

        if result.success:
            with self._lock:
                self._traj = result.trajectory
                self._t_start = t_snap

        response.success = result.success
        response.projected = result.projected
        response.goal_x, response.goal_y = result.goal_xy
        response.joint_goal = ([] if result.q_goal is None
                               else [float(v) for v in result.q_goal])
        response.duration = float(result.duration)
        response.message = result.message

        log = self.get_logger().info if result.success else self.get_logger().warn
        log(f"move_to_target({request.x:.3f}, {request.y:.3f}) -> "
            f"{'OK' if result.success else 'REJECTED'}: {result.message}"
            + (f" [{result.duration:.2f}s]" if result.success else ""))
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
