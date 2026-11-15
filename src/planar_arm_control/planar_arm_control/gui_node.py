#!/usr/bin/env python3
"""
gui_node.py  —  ROS side of the operator GUI + process entry point.

Threading model (the key design point)
--------------------------------------
    Qt main thread                       ROS thread ("ros_executor", daemon)
    ──────────────                       ───────────────────────────────────
    QApplication.exec_()                 SingleThreadedExecutor.spin()
    QTimer 30 Hz ─ buffer.snapshot() ◄── /joint_states cb ─ buffer.append()
    slots ◄─── queued Qt signals ◄────── /controller_status cb, service futures
    buttons ─ send_target() ──────────►  client.call_async()  (never blocks)

* Qt must own the main thread, so ROS spins in a background thread.
* 50 Hz joint data goes through a lock-protected ring buffer that the GUI
  pulls at its own render rate. It is never emitted as a Qt signal per
  message, so the message rate and the render rate are decoupled.
* Low-rate discrete events (service results, controller status) are emitted
  as pyqtSignals. Because the emitting thread differs from the receiver's
  thread, Qt queues them onto the GUI thread: widgets are only ever touched
  from the GUI thread.
* Service calls use call_async + add_done_callback, so a slow or missing
  controller can never freeze the window.
"""

import signal
import sys
import threading

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState

from PyQt5 import QtCore, QtWidgets

from planar_arm_interfaces.msg import ControllerStatus
from planar_arm_interfaces.srv import MoveToTarget

from planar_arm_control.gui_window import ArmWindow
from planar_arm_control.planar_arm import PlanarArm
from planar_arm_control.telemetry import TelemetryBuffer

LINK_LENGTHS = [3.0, 2.0, 1.5]


class QtBridge(QtCore.QObject):
    """Signals emitted from the ROS thread, received in the Qt thread."""

    move_result = QtCore.pyqtSignal(object)   # dict
    status = QtCore.pyqtSignal(object)        # dict
    error = QtCore.pyqtSignal(str)


class GuiNode(Node):

    def __init__(self, bridge, buffer):
        super().__init__("gui_node")
        self.arm = PlanarArm(LINK_LENGTHS)
        self.bridge = bridge
        self.buffer = buffer

        self.create_subscription(JointState, "joint_states", self._on_joint_state, 10)
        self.create_subscription(ControllerStatus, "controller_status",
                                 self._on_status, 10)
        self.move_client = self.create_client(MoveToTarget, "move_to_target")
        self.get_logger().info("gui_node started.")

    # ---------------- ROS thread ----------------
    def _on_joint_state(self, msg):
        if len(msg.position) != self.arm.n_joints:
            return
        self.buffer.append(msg.position, msg.velocity)

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

    # ---------------- called from the Qt thread ----------------
    def send_target(self, x, y):
        """Non-blocking: returns immediately; the result arrives as a signal."""
        if not self.move_client.service_is_ready():
            self.bridge.error.emit(
                "Controller not available: /move_to_target service not found.")
            return False
        req = MoveToTarget.Request()
        req.x, req.y, req.duration = float(x), float(y), 0.0
        future = self.move_client.call_async(req)
        future.add_done_callback(lambda f: self._on_move_done(f, x, y))
        self.get_logger().info(f"Sent target ({x:.2f}, {y:.2f})")
        return True


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
    node = GuiNode(bridge, buffer)

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    ros_thread = threading.Thread(target=_spin, args=(executor, node),
                                  name="ros_executor", daemon=True)
    ros_thread.start()

    window = ArmWindow(node.arm, buffer, bridge, node.send_target)
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
