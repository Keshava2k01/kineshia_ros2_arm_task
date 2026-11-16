# 3-DoF Planar Arm — ROS 2 Control, Telemetry GUI & Pick-and-Place

A small but complete ROS 2 (Humble) system for a simulated 3-DoF planar arm:

- **`controller_node`**: accepts Cartesian targets, runs IK, plans smooth
  quintic joint trajectories that respect joint limits and the ground constraint
  along the *whole path*, and streams `sensor_msgs/JointState` at 50 Hz through a
  swappable hardware backend.
- **`pick_place_node`**: the full pick-and-place sequence as a **ROS 2 action**
  with feedback and cancel. Cancelling brings the arm to a smooth stop.
- **`gui_node`**: a PyQt5/PyQtGraph operator GUI with a live arm view, joint-angle
  plots, end-effector telemetry and pick/place controls. The Qt event loop and the
  ROS 2 executor run side by side without blocking each other.

**Demo video:** [watch on Google Drive](https://drive.google.com/file/d/1nSMy3dZHzbfrHhmIwJg9qR3Zk1jBBdgy/view?usp=sharing).
It shows the fixed test scenario: pick (4, 2) → place (−3, 3), and the unreachable (7, 3) target.

| Brief item | Status |
|---|---|
| Controller node (IK, smooth trajectory, fixed-rate JointState, limits + ground, swappable backend) | ✅ |
| Telemetry GUI node (live arm, joint plots, EE telemetry, pick/place targets, non-blocking Qt + ROS) | ✅ |
| Pick-and-place sequence | ✅ |
| Package hygiene (ament packages, launch file, README) | ✅ |
| Stretch: second control mode, a PID trajectory-tracking loop on simulated dynamics with a live error plot (`control_mode:=pid`) | ✅ |
| Stretch: ROS 2 action for pick-and-place with feedback and cancel | ✅ |
| Stretch: tests (35 pytest unit tests) | ✅ |

The design note and the written sim-to-real answer (Part 2) are in
[docs/DESIGN.md](docs/DESIGN.md).

---

## Quick start

Tested on **Ubuntu 22.04 (WSL2) + ROS 2 Humble + Python 3.10**.

```bash
# dependencies (once)
sudo apt install ros-humble-desktop ros-dev-tools \
    python3-numpy python3-pytest python3-pyqt5 python3-pyqtgraph

# build (from the repo root = colcon workspace root)
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash

# run controller + pick-and-place sequencer + GUI
ros2 launch planar_arm_control bringup.launch.py

# ...or with the PID trajectory-tracking control mode
ros2 launch planar_arm_control bringup.launch.py control_mode:=pid
```

> On WSLg, if the window appears blank, run `export QT_X11_NO_MITSHM=1` before launching.

---

## Fixed test scenario

| Case | How to run it in the GUI | Expected result |
|---|---|---|
| **Pick (4.0, 2.0) → place (−3.0, 3.0)** | Targets are pre-filled. Press **Run pick & place**. | 8 phases shown as *k / 8*. The orange object is carried from (4, 2) and left at (−3, 3). Green banner "Pick-and-place done in ≈14 s". |
| **Unreachable (7.0, 3.0)** | Press **Test unreachable (7, 3)**. | Orange banner: target is outside the workspace (reach 6.50), moving to the closest reachable point **(5.974, 2.560)**. A red ✕ marks the request and a green ● the projected goal, joined by a dashed line. The arm stretches out to the boundary. |
| Unreachable as a *pick* target | Set pick to (7, 3) and press **Run pick & place**. | Rejected before moving. Grasping at a projected point would grasp nothing, so pick/place targets are never projected. |
| Cancel | Press **Cancel** during a sequence. | The arm decelerates smoothly to rest (≤ 2 rad/s²). The banner reports the phase and whether the object is still held. |

The same cases from the command line:

```bash
ros2 action send_goal /pick_place planar_arm_interfaces/action/PickPlace \
  "{pick_x: 4.0, pick_y: 2.0, place_x: -3.0, place_y: 3.0}" --feedback

ros2 service call /move_to_target planar_arm_interfaces/srv/MoveToTarget "{x: 7.0, y: 3.0}"
# -> success: true, projected: true, goal_x: 5.974, goal_y: 2.560
```

---

## Control modes

The mode is selected with the `control_mode` parameter or launch argument. Both
modes use the same planner and quintic reference trajectory; only the backend
behind the `ArmBackend` interface changes.

| `control_mode` | What runs | Hardware equivalent |
|---|---|---|
| `position` (default) | An ideal position servo: the measured state equals the reference. | Dynamixel in position mode, where the servo closes its own loop |
| `pid` | A host-side PID computes joint **torques** at 50 Hz and drives a simulated plant with inertia, viscous friction, **gravity** (the arm moves in a vertical plane), torque saturation and hard joint stops. | Dynamixel in current (torque) control mode, with the loop on the host |

**PID law:** τ = J·q̈_ref + Kp·e + Ki·∫e + Kd·(q̇_ref − q̇)

- **Acceleration feed-forward** comes from the quintic trajectory.
- **Gravity is not fed forward.** It is treated as an unmodelled load (an
  unknown payload), so the **integral term** has to reject it.
- **Gains come from pole placement on each joint's inertia:** Kp = J·ω²,
  Kd = 2ζω·J, Ki = Kp·ω·r. The defaults are ω = 10 rad/s, ζ = 1, r = 0.3. The
  stability bound at a 50 Hz control rate is roughly ω·Ts ≲ 0.3.
- **Anti-windup:** the integrator is clamped, and frozen while the output is
  saturated.
- **A move counts as done only when the joints have settled.** The trajectory
  must have ended *and* the error must be below `settle_tolerance`
  (0.001 rad ≈ 0.06°), like a servo's in-position flag. The controller reports
  `SETTLING` in between. This keeps the sequencer's end-effector arrival check
  meaningful.

**Measured** on the pick-and-place scenario: the peak tracking error is
**about 2.1°** (while accelerating against gravity), every move settles below
0.06°, and the peak torques of [131, 96, 11] N·m stay under the limits of
[400, 150, 30] N·m.

The GUI shows a **live tracking-error plot** (e1–e3), the current error, the
peak error over 10 s, the applied torques, and a dashed **reference "ghost"
arm** whenever the measured arm lags the reference. All controller output is
also published on `/tracking_error`.

---

## Architecture

```
                 ┌──────────────────────── gui_node ─────────────────────────┐
                 │ Qt main thread: window, plots, buttons                     │
                 │ ROS thread: subscriptions, service + action clients        │
                 └────┬───────────────┬────────────────────┬──────────────────┘
       /move_to_target│   /pick_place │       /joint_states│ /controller_status
          (service)   │    (action)   ▼        (50 Hz)     │ (10 Hz)   /gripper
                      │      ┌─────────────────┐           │
                      │      │ pick_place_node │── /move_to_target, /stop
                      │      └─────────────────┘           │
                      ▼                                    │
                 ┌──────────────────────── controller_node ┴─────────────────┐
                 │ MotionPlanner: IK → validate → quintic (whole path checked)│
                 │ 50 Hz timer: sample trajectory → ArmBackend → JointState   │
                 └────────────────────────────────────────────────────────────┘
```

| Node | Role |
|---|---|
| `controller_node` | Owns the arm state. Plans and executes trajectories, publishes joint states and status, and provides a smooth stop. |
| `pick_place_node` | Sequencer and action server. Drives the controller step by step, simulates the gripper, and handles cancel and pre-emption. |
| `gui_node` | Operator GUI: telemetry, plots, move commands and the pick-and-place action client. |

### ROS interfaces

| Name | Type | Direction | Notes |
|---|---|---|---|
| `/joint_states` | `sensor_msgs/JointState` | controller → all | 50 Hz, measured position and velocity (plus effort in pid mode) |
| `/tracking_error` | `planar_arm_interfaces/TrackingError` | controller → all | 50 Hz: reference, measured, error, effort |
| `/controller_status` | `planar_arm_interfaces/ControllerStatus` | controller → all | 10 Hz. IDLE/MOVING/SETTLING, control mode, progress, goal, `projected`, message, `command_id` |
| `/move_to_target` | `planar_arm_interfaces/srv/MoveToTarget` | → controller | `x, y, duration` (0 = auto). Returns success, projected, goal, joint goal, duration, message, `command_id` |
| `/stop` | `std_srvs/Trigger` | → controller | Smooth deceleration to rest |
| `/pick_place` | `planar_arm_interfaces/action/PickPlace` | → sequencer | Feedback: phase, step k/N, holding. Result: success, message, elapsed, holding |
| `/gripper` | `std_msgs/Bool` | sequencer → all | Simulated gripper, transient-local (late joiners get the state) |

The custom service, message and action live in a separate **ament_cmake**
package (`planar_arm_interfaces`), because ament_python packages cannot generate
interfaces.

### Parameters

**controller_node**

| Parameter | Default | Meaning |
|---|---|---|
| `publish_rate_hz` | 50.0 | Control loop and `/joint_states` rate |
| `max_joint_velocity` | 1.0 rad/s | Per-joint velocity limit used to time trajectories |
| `max_joint_acceleration` | 2.0 rad/s² | Per-joint acceleration limit |
| `min_duration` | 0.5 s | Shortest allowed move |
| `initial_joint_positions` | [π/2, 0, 0] | Start pose (arm straight up) |
| `joint_names` | [joint1, joint2, joint3] | Names in `JointState` |
| `backend` | `sim` | Hardware backend, see `backends.py` |
| `control_mode` | `position` | `position` (ideal servo) or `pid` (torque PID on simulated dynamics) |
| `pid_bandwidth` | 10.0 rad/s | Closed-loop natural frequency ω used to derive the gains |
| `pid_damping` | 1.0 | Damping ratio ζ |
| `pid_integral_ratio` | 0.3 | Ki = Kp·ω·ratio |
| `settle_tolerance` | 0.001 rad | Max joint error for a move to count as done |
| `settle_timeout` | 2.0 s | Stop waiting for settling after this (logged as a warning) |

**pick_place_node**

| Parameter | Default | Meaning |
|---|---|---|
| `approach_offset` | 0.5 | Height of the approach/retreat points above pick and place. 0 gives the plain 4-step sequence |
| `grip_time` | 0.5 s | Dwell for grip and release |
| `move_timeout_margin` | 5.0 s | Allowed on top of the planned move duration |
| `arrival_tolerance` | 0.01 | Max end-effector error accepted after each move |

Launch arguments: `publish_rate_hz`, `max_joint_velocity`,
`max_joint_acceleration`, `min_duration`, `backend`, `control_mode`,
`pid_bandwidth`, `approach_offset`, e.g.

```bash
ros2 launch planar_arm_control bringup.launch.py control_mode:=pid pid_bandwidth:=8.0
ros2 launch planar_arm_control bringup.launch.py publish_rate_hz:=100.0 approach_offset:=0.0
```

### Other useful commands

```bash
ros2 service call /move_to_target planar_arm_interfaces/srv/MoveToTarget "{x: 4.0, y: 2.0}"
ros2 service call /stop std_srvs/srv/Trigger
ros2 topic hz /joint_states          # ~50 Hz
ros2 topic echo /controller_status
```

---

## Package layout

```
src/
├── planar_arm_interfaces/              ament_cmake: interface definitions
│   ├── msg/ControllerStatus.msg
│   ├── msg/TrackingError.msg
│   ├── srv/MoveToTarget.srv
│   └── action/PickPlace.action
└── planar_arm_control/                 ament_python
    ├── launch/bringup.launch.py        controller + sequencer + GUI
    ├── planar_arm_control/
    │   ├── planar_arm.py               PROVIDED kinematics (unmodified)
    │   ├── trajectory.py               quintic + segmented trajectories       (no ROS)
    │   ├── planner.py                  IK validation, path checks, stop plan  (no ROS)
    │   ├── backends.py                 ArmBackend interface, position + pid backends (no ROS)
    │   ├── control.py                  JointPid + ArmDynamics plant model     (no ROS)
    │   ├── sequence.py                 pick-and-place step list + validation  (no ROS)
    │   ├── telemetry.py                thread-safe joint-state ring buffer    (no ROS)
    │   ├── controller_node.py          controller ROS node
    │   ├── pick_place_node.py          action server / sequencer ROS node
    │   ├── gui_node.py                 GUI ROS side + Qt/ROS threading
    │   └── gui_window.py               Qt window (no rclpy)
    └── test/                           pytest unit tests
```

All the logic modules are plain Python and unit-tested without ROS. The nodes
are thin shells around them.

---

## Key design points

Each point is covered in more depth in [docs/DESIGN.md](docs/DESIGN.md).

- **Quintic trajectories.** Each move starts and ends with zero velocity and
  acceleration. A new target can arrive mid-move: the new trajectory starts from
  the currently commanded position *and velocity*, so there is no jump.
- **The whole path is checked, not just the endpoints.** Joint limits form a box,
  so they hold along a straight joint-space path. The ground constraint does not:
  two valid poses can have a path between them that dips below y = 0. The planner
  samples the full trajectory and, if it fails, re-plans through a safe
  straight-up pose.
- **Hardware seam.** The controller only talks to an `ArmBackend`
  (`write_command` / `read_state`). A Dynamixel backend is one new subclass,
  selected by the `backend` parameter. The planner, nodes and GUI are unchanged.
- **Qt + ROS threading.** ROS spins in a background thread and Qt owns the main
  thread.
  - 50 Hz joint data goes into a lock-protected ring buffer that the GUI reads
    at 30 Hz, so the message rate and the render rate are independent.
  - Discrete events reach the GUI as queued Qt signals.
  - Button presses post work to the ROS executor (`create_task`), so rclpy is
    only ever called from the ROS thread. The GUI shows the measured message
    rate, render fps and data age.
- **Race-free completion.** Every accepted controller command gets a
  `command_id`. The sequencer waits for *its* id to report IDLE, then checks
  the end effector actually arrived. A newer id means someone else commanded
  the arm, so the sequence aborts as pre-empted.
- **Timing.** The control timer runs on ROS time; on Humble it measured
  50.000 Hz, while a steady-clock timer measured 46.6 Hz. Trajectory sampling
  uses a steady clock, so a wall-clock step (NTP, WSL resync) cannot make the arm
  jump along its path. Under `use_sim_time` it follows the simulation clock.

---

## Tests

```bash
python3 -m pytest src/planar_arm_control/test -q      # 35 passed
```

| File | Covers |
|---|---|
| `test_trajectory.py` | boundary conditions, peak velocity/acceleration against the closed form, limit-respecting auto duration, non-zero start velocity, segmented trajectories |
| `test_planner.py` | the test scenario (4,2) → (−3,3), (7,3) projection, below-ground rejection, mid-motion retargeting continuity, duration stretching, smooth stop, via-pose rerouting over 300 random pose pairs |
| `test_sequence.py` | 8-step and 4-step sequences, approach skipped near the reach boundary, invalid targets rejected |
| `test_telemetry.py` | windowing, rate and age estimates, concurrent append and snapshot |
| `test_control.py` | gravity-torque signs, hard stops, the integral term rejecting gravity while holding, PID tracking of the scenario moves (peak error < 3°, settles < 0.001 rad, torques within limits), backend selection |

The full system was also exercised headless (GUI offscreen) against the
running nodes, in both control modes: the test scenario, cancel, an
unreachable pick, and pre-emption.

---

## Notes on the provided `planar_arm.py`

The file is **unmodified**. Three behaviours worth flagging, all handled in
`planner.py`:

1. **The Jacobian fallback is unconstrained.** When the analytical IK finds no
   solution, `inverse_kinematics` falls back to `jacobian_ik`, which does not
   check joint limits or the ground constraint. Its result is therefore
   validated before use.
2. **The ground constraint is not convex in joint space.** The IK guarantees a
   valid *goal pose*, but not a valid *path* to it, so the whole trajectory is
   checked.
3. **Projection is reported only through `print()`.** `reachable_target` writes to
   stdout, so the caller cannot tell a target was projected. The planner repeats
   the same projection rule and reports it in the service response.

---

## Known limitations

- **Python GIL.** The control loop and the planner run on separate executor
  threads, but they share the GIL, so a long IK call can delay a control tick by
  a few ms. Measured steady-state jitter is about 0.75 ms (std).
- **Simplified dynamics.** In `position` mode the servos are ideal. In `pid`
  mode each joint has a constant effective inertia, so the real
  configuration-dependent, coupled mass matrix M(q) is not modelled and there
  is no motor or sensor noise. The gripper is a boolean.
- **Step completion is detected from `/controller_status` at 10 Hz,** which adds
  up to about 0.1 s per sequence step.
- **One pick-and-place goal at a time.** New goals are rejected while one runs.
- **Pre-emption has a small race window.** A command sent in the few
  milliseconds between the sequencer's check and its own next command is
  overridden rather than detected.
