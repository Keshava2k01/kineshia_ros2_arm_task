#!/usr/bin/env python3
"""
pick_place_node.py  —  pick-and-place sequencer exposed as a ROS 2 action.

Interfaces
    Action     /pick_place        planar_arm_interfaces/action/PickPlace
    Client     /move_to_target    planar_arm_interfaces/srv/MoveToTarget
    Client     /stop              std_srvs/Trigger
    Subscribes /controller_status planar_arm_interfaces/msg/ControllerStatus
    Subscribes /joint_states      sensor_msgs/JointState
    Publishes  /gripper           std_msgs/Bool  (transient-local: late joiners
                                                  get the current state)

Parameters
    approach_offset      height of approach/retreat points above targets [m] 0.5
    grip_time            dwell for grip / release (simulated gripper)    [s] 0.5
    move_timeout_margin  extra time allowed on top of the planned duration [s] 5.0
    arrival_tolerance    max end-effector error accepted after a move     [m] 0.01

How a move step completes
    The controller returns a command_id with each accepted move and reports
    it in /controller_status. A step is done when status.command_id equals
    ours AND state == IDLE; a stale IDLE published just before our command
    has an older id, so it cannot be mistaken for completion. A NEWER id
    means someone else commanded the arm (GUI/CLI) -> the sequence aborts.
    Finally the end-effector position from /joint_states is checked against
    the goal: in simulation this always passes, on hardware it is the check
    that catches a stalled or blocked joint.

Threading
    The execute callback blocks while it waits on moves, so the node runs on
    a MultiThreadedExecutor with everything in one ReentrantCallbackGroup:
    status / joint-state callbacks and service responses keep being
    processed on other threads while execute() waits.
"""

import threading
import time

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from planar_arm_interfaces.action import PickPlace
from planar_arm_interfaces.msg import ControllerStatus
from planar_arm_interfaces.srv import MoveToTarget

from planar_arm_control.planar_arm import PlanarArm
from planar_arm_control.sequence import (
    GRIP, MOVE, RELEASE, build_sequence, validate_targets,
)

LINK_LENGTHS = [3.0, 2.0, 1.5]
POLL_S = 0.02

OK, FAILED, CANCELED = "ok", "failed", "canceled"


class PickPlaceNode(Node):

    def __init__(self):
        super().__init__("pick_place_node")
        self.arm = PlanarArm(LINK_LENGTHS)
        self.reach = float(sum(LINK_LENGTHS))

        self.declare_parameter("approach_offset", 0.5)
        self.declare_parameter("grip_time", 0.5)
        self.declare_parameter("move_timeout_margin", 5.0)
        self.declare_parameter("arrival_tolerance", 0.01)

        self._lock = threading.Lock()
        self._busy = False
        self._status = None
        self._q = None
        self._holding = False
        self._last_cid = 0      # newest controller command id we know is ours

        cb = ReentrantCallbackGroup()
        self.move_client = self.create_client(MoveToTarget, "move_to_target",
                                              callback_group=cb)
        self.stop_client = self.create_client(Trigger, "stop", callback_group=cb)
        self.create_subscription(ControllerStatus, "controller_status",
                                 self._on_status, 10, callback_group=cb)
        self.create_subscription(JointState, "joint_states",
                                 self._on_joint_state, 10, callback_group=cb)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.gripper_pub = self.create_publisher(Bool, "gripper", latched)
        self._set_gripper(False)

        self.action_server = ActionServer(
            self, PickPlace, "pick_place",
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=cb)
        self.get_logger().info("pick_place_node ready (action: /pick_place).")

    # ------------------------------------------------------------------
    def _param(self, name):
        return self.get_parameter(name).value

    def _on_status(self, msg):
        with self._lock:
            self._status = msg

    def _on_joint_state(self, msg):
        if len(msg.position) == self.arm.n_joints:
            with self._lock:
                self._q = np.asarray(msg.position, dtype=float)

    def _set_gripper(self, closed):
        self._holding = bool(closed)
        self.gripper_pub.publish(Bool(data=self._holding))

    def _on_goal(self, goal):
        with self._lock:
            if self._busy:
                self.get_logger().warn("Rejecting goal: a pick-and-place is already running.")
                return GoalResponse.REJECT
            self._busy = True
        return GoalResponse.ACCEPT

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------
    def _execute(self, goal_handle):
        t0 = time.monotonic()
        g = goal_handle.request
        pick, place = (g.pick_x, g.pick_y), (g.place_x, g.place_y)
        self.get_logger().info(
            f"Pick-and-place: pick ({pick[0]:.2f}, {pick[1]:.2f}) -> "
            f"place ({place[0]:.2f}, {place[1]:.2f})")
        try:
            # Baseline for pre-emption detection: any controller command id
            # newer than the last one we issued was sent by someone else.
            with self._lock:
                self._last_cid = self._status.command_id if self._status else 0
            ok, msg = validate_targets(pick, place, self.reach, self.arm.min_y)
            if not ok:
                goal_handle.abort()
                return self._result(False, msg, t0)

            steps = build_sequence(pick, place, self._param("approach_offset"),
                                   self.reach)
            for i, step in enumerate(steps, start=1):
                self._feedback(goal_handle, step.phase, i, len(steps),
                               f"{step.phase.replace('_', ' ').lower()}"
                               + (f" -> ({step.xy[0]:.2f}, {step.xy[1]:.2f})"
                                  if step.xy else ""))
                if step.kind == MOVE:
                    outcome, msg = self._move(goal_handle, step.xy)
                else:
                    self._set_gripper(step.kind == GRIP)
                    outcome, msg = self._wait(goal_handle, self._param("grip_time"))

                if outcome == CANCELED:
                    self._stop_arm()
                    goal_handle.canceled()
                    return self._result(False, f"Cancelled during {step.phase}.", t0)
                if outcome == FAILED:
                    self._stop_arm()
                    goal_handle.abort()
                    return self._result(False, f"{step.phase} failed: {msg}", t0)

            goal_handle.succeed()
            return self._result(
                True, f"Object placed at ({place[0]:.2f}, {place[1]:.2f}).", t0)
        finally:
            with self._lock:
                self._busy = False

    def _result(self, success, message, t0):
        res = PickPlace.Result()
        res.success = success
        res.message = message
        res.elapsed = time.monotonic() - t0
        res.holding_object = self._holding
        if success:
            self.get_logger().info(f"Pick-and-place done in {res.elapsed:.2f}s: {message}")
        else:
            self.get_logger().warn(f"Pick-and-place ended: {message}")
        return res

    def _feedback(self, goal_handle, phase, step, total, message):
        fb = PickPlace.Feedback()
        fb.phase = phase
        fb.step = step
        fb.total_steps = total
        fb.holding_object = self._holding
        fb.message = message
        goal_handle.publish_feedback(fb)
        self.get_logger().info(f"[{step}/{total}] {phase}: {message}")

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------
    def _preempted(self):
        """True if the controller received a command we did not send."""
        with self._lock:
            st = self._status
        return st is not None and st.command_id > self._last_cid

    def _wait(self, goal_handle, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if goal_handle.is_cancel_requested:
                return CANCELED, ""
            if self._preempted():
                return FAILED, "pre-empted by another command to the controller."
            time.sleep(POLL_S)
        return OK, ""

    def _wait_future(self, future, timeout):
        end = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() > end:
                return False
            time.sleep(POLL_S)
        return True

    def _move(self, goal_handle, xy):
        if not self.move_client.wait_for_service(timeout_sec=2.0):
            return FAILED, "controller not available (/move_to_target)."
        if self._preempted():
            return FAILED, "pre-empted by another command to the controller."
        req = MoveToTarget.Request()
        req.x, req.y, req.duration = float(xy[0]), float(xy[1]), 0.0
        future = self.move_client.call_async(req)
        if not self._wait_future(future, 5.0):
            return FAILED, "no response from /move_to_target."
        res = future.result()
        if not res.success:
            return FAILED, res.message
        if res.projected:
            # Can only happen for an approach point; never for pick/place.
            return FAILED, f"target ({xy[0]:.2f}, {xy[1]:.2f}) was projected."

        cid = res.command_id
        self._last_cid = cid
        deadline = time.monotonic() + res.duration + self._param("move_timeout_margin")
        while True:
            if goal_handle.is_cancel_requested:
                return CANCELED, ""
            if self._preempted():
                return FAILED, "pre-empted by another command to the controller."
            with self._lock:
                st = self._status
            if st is not None and st.command_id == cid and st.state == "IDLE":
                break
            if time.monotonic() > deadline:
                return FAILED, f"timed out after {res.duration:.1f}s + margin."
            time.sleep(POLL_S)

        with self._lock:
            q = self._q
        if q is None:
            return FAILED, "no /joint_states received."
        ee = np.asarray(self.arm.end_effector(q))
        err = float(np.linalg.norm(ee - np.array([res.goal_x, res.goal_y])))
        if err > self._param("arrival_tolerance"):
            return FAILED, (f"arrived {err:.3f} from goal "
                            f"(tolerance {self._param('arrival_tolerance')}).")
        return OK, ""

    def _stop_arm(self):
        if not self.stop_client.service_is_ready():
            self.get_logger().error("Cannot stop arm: /stop not available.")
            return
        future = self.stop_client.call_async(Trigger.Request())
        if not self._wait_future(future, 2.0):
            self.get_logger().error("No response from /stop.")


def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceNode()
    executor = MultiThreadedExecutor(num_threads=4)
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
