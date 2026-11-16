#!/usr/bin/env python3
"""
gui_node.py  —  ROS side of the operator GUI + process entry point.

Threading model (the key design point)
--------------------------------------
    Qt main thread                       ROS thread ("ros_executor", daemon)
    ──────────────                       ───────────────────────────────────
    QApplication.exec_()                 SingleThreadedExecutor.spin()
    QTimer 30 Hz ─ buffer.snapshot() ◄── /joint_states cb ─ buffer.append()
    slots ◄─── queued Qt signals ◄────── status / gripper cbs, service and
                                         action futures (done callbacks)
    buttons ─ send_*() ─ create_task ──► call_async / send_goal_async
                                         (run ON the ROS thread)

* Qt must own the main thread, so ROS spins in a background thread.
* 50 Hz joint data goes through a lock-protected ring buffer that the GUI
  pulls at its own render rate. It is never emitted as a Qt signal per
  message, so the message rate and the render rate are decoupled.
* Low-rate discrete events (results, feedback, status, gripper) are emitted
  as pyqtSignals. The emitting thread differs from the receiver's, so Qt
  queues them onto the GUI thread: widgets are only touched from Qt's thread.
* The reverse direction: button handlers never call rclpy themselves. They
  post a task to the executor (Executor.create_task is thread-safe and wakes
  the executor), so every rclpy call happens on the ROS thread. Nothing in
  the Qt thread ever waits on ROS.
"""

import signal
import sys
import threading

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from PyQt5 import QtCore, QtWidgets

from planar_arm_interfaces.action import PickPlace
from planar_arm_interfaces.msg import ControllerStatus, TrackingError
from planar_arm_interfaces.srv import MoveToTarget

from planar_arm_control.gui_window import ArmWindow
from planar_arm_control.planar_arm import PlanarArm
from planar_arm_control.telemetry import TelemetryBuffer

LINK_LENGTHS = [3.0, 2.0, 1.5]

_STATUS_NAMES = {
    GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
    GoalStatus.STATUS_CANCELED: "CANCELED",
    GoalStatus.STATUS_ABORTED: "ABORTED",
}


class QtBridge(QtCore.QObject):
    """Signals emitted from the ROS thread, received in the Qt thread."""

    move_result = QtCore.pyqtSignal(object)   # dict
    status = QtCore.pyqtSignal(object)        # dict
    error = QtCore.pyqtSignal(str)
    pp_state = QtCore.pyqtSignal(str)         # "accepted" | "rejected"
    pp_feedback = QtCore.pyqtSignal(object)   # dict
    pp_result = QtCore.pyqtSignal(object)     # dict
    gripper = QtCore.pyqtSignal(bool)


class GuiNode(Node):

    def __init__(self, bridge, buffer, err_buffer, ref_buffer):
        super().__init__("gui_node")
        self.arm = PlanarArm(LINK_LENGTHS)
        self.bridge = bridge
        self.buffer = buffer            # measured q, qd
        self.err_buffer = err_buffer    # tracking error, effort
        self.ref_buffer = ref_buffer    # reference q (ghost arm)
        self._pp_goal = None        # active ClientGoalHandle (ROS thread only)

        self.create_subscription(JointState, "joint_states", self._on_joint_state, 10)
        self.create_subscription(TrackingError, "tracking_error", self._on_tracking, 10)
        self.create_subscription(ControllerStatus, "controller_status",
                                 self._on_status, 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, "gripper",
                                 lambda m: self.bridge.gripper.emit(m.data), latched)
        self.move_client = self.create_client(MoveToTarget, "move_to_target")
        self.pp_client = ActionClient(self, PickPlace, "pick_place")
        self.get_logger().info("gui_node started.")

    # ==================================================================
    # Commands — called from the Qt thread, executed on the ROS thread
    # ==================================================================
    def _post(self, fn, *args):
        """Run fn(*args) on the ROS executor thread."""
        self.executor.create_task(fn, *args)

    def send_target(self, x, y):
        if not self.move_client.service_is_ready():
            self.bridge.error.emit(
                "Controller not available: /move_to_target service not found.")
            return False
        self._post(self._do_send_target, float(x), float(y))
        return True

    def send_pick_place(self, pick, place):
        if not self.pp_client.server_is_ready():
            self.bridge.error.emit("Sequencer not available: /pick_place not found.")
            return False
        self._post(self._do_send_pick_place, tuple(pick), tuple(place))
        return True

    def cancel_pick_place(self):
        self._post(self._do_cancel_pick_place)

    def pick_place_available(self):
        return self.pp_client.server_is_ready()

    # ==================================================================
    # ROS thread
    # ==================================================================
    def _on_joint_state(self, msg):
        if len(msg.position) != self.arm.n_joints:
            return
        self.buffer.append(msg.position, msg.velocity)

    def _on_tracking(self, msg):
        if len(msg.error) != self.arm.n_joints:
            return
        self.err_buffer.append(msg.error, msg.effort)
        self.ref_buffer.append(msg.reference)

    def _on_status(self, msg):
        self.bridge.status.emit({
            "state": msg.state,
            "control_mode": msg.control_mode,
            "goal_x": msg.goal_x,
            "goal_y": msg.goal_y,
            "projected": msg.projected,
            "progress": msg.progress,
            "message": msg.message,
        })

    # ---- move_to_target ----
    def _do_send_target(self, x, y):
        req = MoveToTarget.Request()
        req.x, req.y, req.duration = x, y, 0.0
        future = self.move_client.call_async(req)
        future.add_done_callback(lambda f: self._on_move_done(f, x, y))
        self.get_logger().info(f"Sent target ({x:.2f}, {y:.2f})")

    def _on_move_done(self, future, x, y):
        try:
            res = future.result()
        except Exception as exc:     # noqa: BLE001 - surface any failure in the UI
            self.bridge.error.emit(f"move_to_target failed: {exc}")
            return
        self.bridge.move_result.emit({
            "x": x, "y": y,
            "success": res.success,
            "projected": res.projected,
            "goal_x": res.goal_x,
            "goal_y": res.goal_y,
            "duration": res.duration,
            "message": res.message,
        })

    # ---- pick_place action ----
    def _do_send_pick_place(self, pick, place):
        goal = PickPlace.Goal()
        goal.pick_x, goal.pick_y = float(pick[0]), float(pick[1])
        goal.place_x, goal.place_y = float(place[0]), float(place[1])
        future = self.pp_client.send_goal_async(goal, feedback_callback=self._on_pp_feedback)
        future.add_done_callback(self._on_pp_goal_response)
        self.get_logger().info(f"Sent pick-and-place goal {pick} -> {place}")

    def _on_pp_goal_response(self, future):
        try:
            handle = future.result()
        except Exception as exc:     # noqa: BLE001
            self.bridge.error.emit(f"pick_place goal failed: {exc}")
            self.bridge.pp_state.emit("rejected")
            return
        if not handle.accepted:
            self.bridge.pp_state.emit("rejected")
            return
        self._pp_goal = handle
        self.bridge.pp_state.emit("accepted")
        handle.get_result_async().add_done_callback(self._on_pp_result)

    def _on_pp_feedback(self, msg):
        fb = msg.feedback
        self.bridge.pp_feedback.emit({
            "phase": fb.phase,
            "step": fb.step,
            "total_steps": fb.total_steps,
            "holding_object": fb.holding_object,
            "message": fb.message,
        })

    def _on_pp_result(self, future):
        self._pp_goal = None
        try:
            wrapped = future.result()
        except Exception as exc:     # noqa: BLE001
            self.bridge.pp_result.emit({"status": "ERROR", "message": str(exc),
                                        "elapsed": 0.0, "holding_object": False})
            return
        r = wrapped.result
        self.bridge.pp_result.emit({
            "status": _STATUS_NAMES.get(wrapped.status, f"STATUS_{wrapped.status}"),
            "success": r.success,
            "message": r.message,
            "elapsed": r.elapsed,
            "holding_object": r.holding_object,
        })

    def _do_cancel_pick_place(self):
        if self._pp_goal is not None:
            self.get_logger().info("Cancelling pick-and-place goal")
            self._pp_goal.cancel_goal_async()


def _spin(executor, node):
    try:
        executor.spin()
    except ExternalShutdownException:
        pass
    except Exception as exc:          # noqa: BLE001
        node.get_logger().error(f"ROS executor thread crashed: {exc}")


def main(args=None):
    rclpy.init(args=args)
    app = QtWidgets.QApplication(remove_ros_args(sys.argv))

    bridge = QtBridge()
    buffer = TelemetryBuffer(window_s=20.0)
    err_buffer = TelemetryBuffer(window_s=20.0)
    ref_buffer = TelemetryBuffer(window_s=1.0)
    node = GuiNode(bridge, buffer, err_buffer, ref_buffer)

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    ros_thread = threading.Thread(target=_spin, args=(executor, node),
                                  name="ros_executor", daemon=True)
    ros_thread.start()

    window = ArmWindow(node.arm, buffer, bridge, node,
                       err_buffer=err_buffer, ref_buffer=ref_buffer)
    window.show()

    # Ctrl+C / launch shutdown: Qt's C++ loop does not run Python signal
    # handlers on its own, so wake the interpreter periodically.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    wake = QtCore.QTimer()
    wake.timeout.connect(lambda: None)
    wake.start(100)

    rc = 0
    try:
        rc = app.exec_()
    finally:
        executor.shutdown(timeout_sec=1.0)
        ros_thread.join(timeout=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(rc)


if __name__ == "__main__":
    main()
