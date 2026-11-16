"""
gui_window.py  —  the operator window (Qt only; no rclpy in this file).

The window never touches ROS directly. It is given:
    buffer     TelemetryBuffer filled by the ROS thread (pulled at 30 Hz)
    signals    QObject whose signals (move_result, status, error, pp_state,
               pp_feedback, pp_result, gripper) are emitted from the ROS
               thread and delivered here as queued (thread-safe) connections
    commands   object with non-blocking methods
                   send_target(x, y) -> bool
                   send_pick_place((x, y), (x, y)) -> bool
                   cancel_pick_place()
                   pick_place_available() -> bool

Keeping ROS out of this file means the threading boundary is explicit and the
window could be driven by a different transport (or a test) unchanged.
"""

import time

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

RENDER_HZ = 30
PLOT_WINDOW_S = 10.0
TRAIL_S = 4.0
STALE_S = 0.5
HOME_XY = (0.0, 6.5)
UNREACHABLE_XY = (7.0, 3.0)

JOINT_COLORS = ["#4fc3f7", "#ffb74d", "#ba68c8"]
BANNER_STYLES = {
    "info": "background:#1b3a4b; color:#e0f7fa;",
    "warn": "background:#5d4037; color:#ffe0b2; font-weight:bold;",
    "error": "background:#5c1a1a; color:#ffcdd2; font-weight:bold;",
    "ok": "background:#1b5e20; color:#c8e6c9; font-weight:bold;",
}
OBJECT_COLOR = "#ff9800"


def _fmt(value, decimals):
    """Format without a '-0.00' for tiny negative values."""
    return f"{round(float(value), decimals) + 0.0:.{decimals}f}"


class ArmWindow(QtWidgets.QMainWindow):

    def __init__(self, arm, buffer, signals, commands, err_buffer=None, ref_buffer=None):
        super().__init__()
        self.arm = arm
        self.buffer = buffer
        self.err_buffer = err_buffer    # (tracking error, effort) from /tracking_error
        self.ref_buffer = ref_buffer    # reference q, drawn as a ghost arm
        self.commands = commands
        self.reach = float(sum(arm.link_lengths))
        self.status = {}
        self.move_buttons = []
        self.pp_running = False
        self.obj_xy = None          # object position when not held
        self.holding = False        # simulated gripper closed
        self._pending_pick = None
        self._last_ee = None
        self._frames = 0
        self._fps = 0.0
        self._fps_t0 = time.monotonic()

        self.setWindowTitle("Planar Arm — Telemetry & Control")
        self.resize(1400, 860)
        pg.setConfigOptions(antialias=True)

        central = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(central)
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self._build_arm_view(), stretch=5)
        left.addWidget(self._build_joint_plots(), stretch=4)
        root.addLayout(left, stretch=1)
        root.addWidget(self._build_side_panel())
        self.setCentralWidget(central)

        self.status_label = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.status_label)

        signals.move_result.connect(self._on_move_result)
        signals.status.connect(self._on_status)
        signals.error.connect(lambda text: self._banner(text, "error"))
        signals.pp_state.connect(self._on_pp_state)
        signals.pp_feedback.connect(self._on_pp_feedback)
        signals.pp_result.connect(self._on_pp_result)
        signals.gripper.connect(self._on_gripper)

        self._update_pick_place_markers()
        self.render_timer = QtCore.QTimer(self)
        self.render_timer.timeout.connect(self._render)
        self.render_timer.start(int(1000 / RENDER_HZ))
        # Low-rate check whether the pick-and-place action server is up.
        self.avail_timer = QtCore.QTimer(self)
        self.avail_timer.timeout.connect(self._update_pp_availability)
        self.avail_timer.start(1000)
        self._update_pp_availability()

    # ==================================================================
    # Layout
    # ==================================================================
    def _build_arm_view(self):
        view = pg.PlotWidget(title="Arm (base frame)")
        view.setAspectLocked(True)
        view.showGrid(x=True, y=True, alpha=0.2)
        view.setXRange(-7.5, 7.5)
        view.setYRange(-1.0, 7.5)
        view.setLabel("bottom", "x")
        view.setLabel("left", "y")

        # Ground: line at y = 0 and a shaded forbidden region below it.
        ground = QtWidgets.QGraphicsRectItem(-20, -5, 40, 5)
        ground.setPen(pg.mkPen(None))
        ground.setBrush(QtGui.QBrush(QtGui.QColor(229, 115, 115, 45)))
        view.addItem(ground)
        view.addItem(pg.InfiniteLine(pos=0, angle=0,
                                     pen=pg.mkPen("#e57373", width=2)))

        # Reachable workspace boundary.
        th = np.linspace(0, np.pi, 181)
        view.plot(self.reach * np.cos(th), self.reach * np.sin(th),
                  pen=pg.mkPen("#78909c", width=1, style=QtCore.Qt.DashLine))

        self.trail = view.plot(pen=pg.mkPen(255, 255, 255, 70, width=1.5))
        # Reference ("ghost") arm: where the trajectory wants the arm to be.
        # Only visible when it differs from the measured pose (pid mode).
        self.ghost = view.plot(pen=pg.mkPen(129, 212, 250, 150, width=3,
                                            style=QtCore.Qt.DashLine))
        self.links = view.plot(
            pen=pg.mkPen("#eceff1", width=7), symbol="o", symbolSize=13,
            symbolBrush="#263238", symbolPen=pg.mkPen("#eceff1", width=2))
        self.ee_marker = pg.ScatterPlotItem(size=12, brush="#ffee58",
                                            pen=pg.mkPen("k"))
        view.addItem(self.ee_marker)
        view.addItem(pg.ScatterPlotItem([0], [0], symbol="s", size=20,
                                        brush="#546e7a", pen=pg.mkPen("w")))

        # Pick / place markers (follow the spin boxes).
        self.pick_marker = pg.ScatterPlotItem(symbol="s", size=14,
                                              brush=(66, 165, 245, 160),
                                              pen=pg.mkPen("#42a5f5"))
        self.place_marker = pg.ScatterPlotItem(symbol="t", size=16,
                                               brush=(171, 71, 188, 160),
                                               pen=pg.mkPen("#ab47bc"))
        self.pick_text = pg.TextItem("pick", color="#90caf9", anchor=(0.5, 1.6))
        self.place_text = pg.TextItem("place", color="#ce93d8", anchor=(0.5, 1.6))
        for item in (self.pick_marker, self.place_marker,
                     self.pick_text, self.place_text):
            view.addItem(item)

        # Requested target (red x) vs. goal actually used (green o).
        self.req_marker = pg.ScatterPlotItem(symbol="x", size=18,
                                             brush="#ef5350", pen=pg.mkPen("#ef5350"))
        self.goal_marker = pg.ScatterPlotItem(symbol="o", size=18,
                                              brush=(0, 0, 0, 0),
                                              pen=pg.mkPen("#66bb6a", width=3))
        self.proj_line = view.plot(pen=pg.mkPen("#ef5350", width=1.5,
                                                style=QtCore.Qt.DashLine))
        view.addItem(self.req_marker)
        view.addItem(self.goal_marker)

        # The object being picked (appears when a pick-and-place starts).
        self.obj_marker = pg.ScatterPlotItem(symbol="s", size=18,
                                             brush=OBJECT_COLOR, pen=pg.mkPen("k"))
        view.addItem(self.obj_marker)

        view.scene().sigMouseClicked.connect(self._on_view_clicked)
        self.arm_view = view
        return view

    def _build_joint_plots(self):
        glw = pg.GraphicsLayoutWidget()
        self.joint_curves = []
        first = None
        for i in range(self.arm.n_joints):
            p = glw.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True, alpha=0.2)
            p.setLabel("left", f"q{i + 1}°")
            p.setXRange(-PLOT_WINDOW_S, 0)
            lo, hi = np.degrees(self.arm.joint_limits[i])
            for lim in (lo, hi):
                p.addItem(pg.InfiniteLine(pos=lim, angle=0, pen=pg.mkPen(
                    "#e57373", width=1, style=QtCore.Qt.DashLine)))
            p.setYRange(lo - 10, hi + 10)
            if first is None:
                first = p
            else:
                p.setXLink(first)
            p.getAxis("bottom").setStyle(showValues=False)
            self.joint_curves.append(p.plot(pen=pg.mkPen(JOINT_COLORS[i], width=2)))

        # Tracking error (reference - measured): zero in position mode.
        ep = glw.addPlot(row=self.arm.n_joints, col=0)
        ep.showGrid(x=True, y=True, alpha=0.2)
        ep.setLabel("left", "err°")
        ep.setLabel("bottom", "time [s]")
        ep.setXLink(first)
        ep.addLegend(offset=(5, 5))
        ep.enableAutoRange(axis="y")
        self.err_curves = [ep.plot(pen=pg.mkPen(JOINT_COLORS[i], width=2), name=f"e{i + 1}")
                           for i in range(self.arm.n_joints)]
        return glw

    def _build_side_panel(self):
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(370)
        lay = QtWidgets.QVBoxLayout(panel)

        self.banner = QtWidgets.QLabel()
        self.banner.setWordWrap(True)
        self.banner.setMargin(8)
        self.banner.hide()
        lay.addWidget(self.banner)

        # ---- telemetry ----
        tele = QtWidgets.QGroupBox("Telemetry")
        grid = QtWidgets.QGridLayout(tele)
        self.lbl = {}
        rows = [("ee", "EE (x, y)"), ("q", "q [deg]"),
                ("qd", "vel [deg/s]"), ("err", "error [deg]"),
                ("peak", "peak |e| 10s"), ("tau", "torque [N·m]"),
                ("state", "Controller"), ("goal", "Goal (x, y)")]
        for r, (key, name) in enumerate(rows):
            grid.addWidget(QtWidgets.QLabel(name), r, 0)
            value = QtWidgets.QLabel("—")
            value.setFont(QtGui.QFont("Monospace"))
            value.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            grid.addWidget(value, r, 1)
            self.lbl[key] = value
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        grid.addWidget(self.progress, len(rows), 0, 1, 2)
        grid.setColumnStretch(1, 1)
        self.msg_label = QtWidgets.QLabel("")
        self.msg_label.setWordWrap(True)
        self.msg_label.setMinimumHeight(60)
        self.msg_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.msg_label.setStyleSheet("color:#90a4ae;")
        grid.addWidget(self.msg_label, len(rows) + 1, 0, 1, 2)
        lay.addWidget(tele)

        # ---- targets ----
        tgt = QtWidgets.QGroupBox("Targets")
        form = QtWidgets.QGridLayout(tgt)
        self.pick_x, self.pick_y = self._spin(4.0), self._spin(2.0)
        self.place_x, self.place_y = self._spin(-3.0), self._spin(3.0)
        for w in (self.pick_x, self.pick_y, self.place_x, self.place_y):
            w.valueChanged.connect(self._update_pick_place_markers)
        form.addWidget(QtWidgets.QLabel("Pick"), 0, 0)
        form.addWidget(self.pick_x, 0, 1)
        form.addWidget(self.pick_y, 0, 2)
        form.addWidget(QtWidgets.QLabel("Place"), 1, 0)
        form.addWidget(self.place_x, 1, 1)
        form.addWidget(self.place_y, 1, 2)
        hint = QtWidgets.QLabel("Click the arm view to set pick, "
                                "Shift+click to set place.")
        hint.setStyleSheet("color:#90a4ae;")
        hint.setWordWrap(True)
        form.addWidget(hint, 2, 0, 1, 3)

        buttons = [
            ("Move to pick", lambda: self._send(self.pick_x.value(), self.pick_y.value())),
            ("Move to place", lambda: self._send(self.place_x.value(), self.place_y.value())),
            ("Home", lambda: self._send(*HOME_XY)),
            ("Test unreachable (7, 3)", lambda: self._send(*UNREACHABLE_XY)),
        ]
        btn_grid = QtWidgets.QGridLayout()
        for i, (text, slot) in enumerate(buttons):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            btn_grid.addWidget(b, i // 2, i % 2)
            self.move_buttons.append(b)
        form.addLayout(btn_grid, 3, 0, 1, 3)
        lay.addWidget(tgt)

        # ---- pick & place sequence (ROS 2 action /pick_place) ----
        self.pp_group = QtWidgets.QGroupBox("Pick && place sequence (action)")
        pp = QtWidgets.QVBoxLayout(self.pp_group)
        row = QtWidgets.QHBoxLayout()
        self.pp_run = QtWidgets.QPushButton("Run pick && place")
        self.pp_cancel = QtWidgets.QPushButton("Cancel")
        self.pp_run.clicked.connect(self._run_pick_place)
        self.pp_cancel.clicked.connect(self.commands.cancel_pick_place)
        row.addWidget(self.pp_run)
        row.addWidget(self.pp_cancel)
        pp.addLayout(row)
        self.pp_progress = QtWidgets.QProgressBar()
        self.pp_progress.setFormat("%v / %m")
        self.pp_progress.setRange(0, 1)
        self.pp_progress.setValue(0)
        pp.addWidget(self.pp_progress)
        self.pp_label = QtWidgets.QLabel("Waiting for /pick_place action server…")
        self.pp_label.setWordWrap(True)
        self.pp_label.setStyleSheet("color:#90a4ae;")
        pp.addWidget(self.pp_label)
        self.pp_cancel.setEnabled(False)
        self.pp_run.setEnabled(False)
        lay.addWidget(self.pp_group)

        lay.addStretch(1)
        return panel

    @staticmethod
    def _spin(value):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(-10.0, 10.0)
        s.setDecimals(2)
        s.setSingleStep(0.1)
        s.setValue(value)
        return s

    # ==================================================================
    # Operator actions
    # ==================================================================
    def _send(self, x, y):
        self.req_marker.setData([x], [y])
        self.goal_marker.setData([], [])
        self.proj_line.setData([], [])
        if self.commands.send_target(x, y):
            self._banner(f"Sent target ({x:.2f}, {y:.2f})…", "info")

    def _run_pick_place(self):
        pick = (self.pick_x.value(), self.pick_y.value())
        place = (self.place_x.value(), self.place_y.value())
        self.req_marker.setData([], [])
        self.goal_marker.setData([], [])
        self.proj_line.setData([], [])
        if self.commands.send_pick_place(pick, place):
            self._pending_pick = pick
            self.pp_run.setEnabled(False)
            self.pp_label.setText("Sending goal…")

    def _set_pp_running(self, running):
        self.pp_running = running
        self.pp_run.setEnabled(not running and self.commands.pick_place_available())
        self.pp_cancel.setEnabled(running)
        for b in self.move_buttons:
            b.setEnabled(not running)

    def _update_pp_availability(self):
        if self.pp_running:
            return
        available = self.commands.pick_place_available()
        self.pp_run.setEnabled(available)
        if not available:
            self.pp_label.setText("Waiting for /pick_place action server…")
        elif self.pp_label.text().startswith("Waiting"):
            self.pp_label.setText("Ready.")

    def _on_view_clicked(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            return
        vb = self.arm_view.getPlotItem().getViewBox()
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        p = vb.mapSceneToView(ev.scenePos())
        if ev.modifiers() & QtCore.Qt.ShiftModifier:
            self.place_x.setValue(p.x())
            self.place_y.setValue(p.y())
        else:
            self.pick_x.setValue(p.x())
            self.pick_y.setValue(p.y())

    def _update_pick_place_markers(self):
        px, py = self.pick_x.value(), self.pick_y.value()
        qx, qy = self.place_x.value(), self.place_y.value()
        self.pick_marker.setData([px], [py])
        self.place_marker.setData([qx], [qy])
        self.pick_text.setPos(px, py)
        self.place_text.setPos(qx, qy)

    # ==================================================================
    # Slots for events coming from the ROS thread (queued connections)
    # ==================================================================
    def _on_move_result(self, r):
        x, y = r["x"], r["y"]
        if not r["success"]:
            self.goal_marker.setData([], [])
            self.proj_line.setData([], [])
            self._banner(f"Target ({x:.2f}, {y:.2f}) rejected: {r['message']}", "error")
            return
        gx, gy = r["goal_x"], r["goal_y"]
        self.goal_marker.setData([gx], [gy])
        if r["projected"]:
            self.proj_line.setData([x, gx], [y, gy])
            self._banner(
                f"⚠ Target ({x:.2f}, {y:.2f}) is outside the workspace "
                f"(reach {self.reach:.2f}). Moving to the closest reachable "
                f"point ({gx:.2f}, {gy:.2f}).", "warn")
        else:
            self.req_marker.setData([], [])
            self.proj_line.setData([], [])
            self._banner(f"Moving to ({gx:.2f}, {gy:.2f}) — "
                         f"{r['duration']:.2f} s. {r['message']}", "info")

    def _render_tracking(self, t_now, q_meas):
        """Error plot, error/torque readouts and the ghost reference arm."""
        if self.err_buffer is None:
            return
        te, e, tau = self.err_buffer.snapshot()
        if te.size:
            sel = te >= t_now - PLOT_WINDOW_S
            e_deg = np.degrees(e[sel])
            for i, curve in enumerate(self.err_curves):
                curve.setData(te[sel] - t_now, e_deg[:, i])
            self.lbl["err"].setText(", ".join(_fmt(v, 2) for v in e_deg[-1]))
            self.lbl["peak"].setText(f"{_fmt(np.max(np.abs(e_deg)), 2)} deg")
            self.lbl["tau"].setText(", ".join(_fmt(v, 1) for v in tau[-1])
                                    if np.any(tau[-1]) else "— (ideal servo)")
        latest = self.ref_buffer.latest() if self.ref_buffer is not None else None
        if latest is not None and np.max(np.abs(latest[1] - q_meas)) > 1e-4:
            pts = np.array(self.arm.forward_kinematics(latest[1]))
            self.ghost.setData(pts[:, 0], pts[:, 1])
        else:
            self.ghost.setData([], [])

    def _on_status(self, s):
        self.status = s

    def _on_pp_state(self, state):
        if state == "accepted":
            self._set_pp_running(True)
            if not self.holding and self._pending_pick is not None:
                self.obj_xy = self._pending_pick   # object appears at pick
            self.pp_label.setText("Goal accepted.")
            self._banner("Pick-and-place started.", "info")
        elif state == "rejected":
            self._set_pp_running(False)
            self._banner("Pick-and-place goal rejected (sequencer busy?).", "error")

    def _on_pp_feedback(self, fb):
        self.pp_progress.setRange(0, max(1, fb["total_steps"]))
        self.pp_progress.setValue(fb["step"])
        self.pp_label.setText(f"{fb['phase']}: {fb['message']}")

    def _on_pp_result(self, r):
        self._set_pp_running(False)
        status = r["status"]
        if status == "SUCCEEDED":
            self.pp_progress.setValue(self.pp_progress.maximum())
            self._banner(f"✔ Pick-and-place done in {r['elapsed']:.1f} s. "
                         f"{r['message']}", "ok")
        elif status == "CANCELED":
            self._banner(f"Pick-and-place cancelled: {r['message']}"
                         + (" Object still held." if r["holding_object"] else ""), "warn")
        else:
            self._banner(f"Pick-and-place {status.lower()}: {r['message']}", "error")
        self.pp_label.setText(f"{status}: {r['message']}")

    def _on_gripper(self, closed):
        if closed and not self.holding:
            self.holding = True
        elif not closed and self.holding:
            self.holding = False
            self.obj_xy = self._last_ee          # dropped where the tool is

    def _banner(self, text, level):
        self.banner.setText(text)
        self.banner.setStyleSheet(BANNER_STYLES[level] + "border-radius:4px;")
        self.banner.show()

    # ==================================================================
    # Render loop (Qt thread, 30 Hz) — pulls from the buffer
    # ==================================================================
    def _render(self):
        self._frames += 1
        now = time.monotonic()
        if now - self._fps_t0 >= 1.0:
            self._fps = self._frames / (now - self._fps_t0)
            self._frames, self._fps_t0 = 0, now

        t, q, qd = self.buffer.snapshot()
        age = self.buffer.age()
        rate = self.buffer.rate_hz()
        stale = age > STALE_S
        age_txt = "no data" if not np.isfinite(age) else f"{age * 1000:.0f} ms"
        self.status_label.setText(
            f"/joint_states {rate:5.1f} Hz   |   render {self._fps:4.1f} fps   |   "
            f"data age {age_txt}" + ("   |   STALE" if stale else ""))
        self.status_label.setStyleSheet("color:#ef5350;" if stale else "")

        if t.size == 0:
            return

        # Arm
        pts = np.array(self.arm.forward_kinematics(q[-1]))
        self.links.setData(pts[:, 0], pts[:, 1])
        ee = pts[-1]
        self._last_ee = (float(ee[0]), float(ee[1]))
        self.ee_marker.setData([ee[0]], [ee[1]])
        if self.holding:
            self.obj_marker.setData([ee[0]], [ee[1]])
        elif self.obj_xy is not None:
            self.obj_marker.setData([self.obj_xy[0]], [self.obj_xy[1]])

        # End-effector trail (last TRAIL_S seconds, subsampled)
        sel = q[t >= t[-1] - TRAIL_S][::2]
        trail = np.array([self.arm.end_effector(qi) for qi in sel])
        self.trail.setData(trail[:, 0], trail[:, 1])

        # Joint angle plots
        sel = t >= t[-1] - PLOT_WINDOW_S
        t_rel = t[sel] - t[-1]
        q_deg = np.degrees(q[sel])
        for i, curve in enumerate(self.joint_curves):
            curve.setData(t_rel, q_deg[:, i])

        # Telemetry
        self.lbl["ee"].setText(f"({_fmt(ee[0], 3)}, {_fmt(ee[1], 3)})")
        self.lbl["q"].setText(", ".join(_fmt(v, 1) for v in np.degrees(q[-1])))
        self.lbl["qd"].setText(", ".join(_fmt(v, 1) for v in np.degrees(qd[-1])))
        self._render_tracking(t[-1], q[-1])
        s = self.status
        if s:
            self.lbl["state"].setText(f"{s['state']} ({s['control_mode']})")
            self.lbl["goal"].setText(f"({_fmt(s['goal_x'], 3)}, {_fmt(s['goal_y'], 3)})"
                                     + (" ⚠" if s["projected"] else ""))
            self.progress.setValue(int(round(100 * s["progress"])))
            self.msg_label.setText(s["message"])
