# Design Note

## Architecture

I built three ROS 2 (Humble) nodes. Build and run steps are in the README.

- **`controller_node`** owns the arm state. A target comes in on `/move_to_target`, I run
  IK, and plan a checked quintic trajectory. A 50 Hz loop samples it, sends the reference
  to an `ArmBackend`, and publishes `/joint_states`, `/tracking_error` and
  `/controller_status`.
- **`pick_place_node`** runs pick-and-place as a `PickPlace` action with feedback and
  cancel. It drives the controller one step at a time.
- **`gui_node`** is the PyQt5/PyQtGraph operator screen: live arm, joint and error plots,
  telemetry, and the move and pick-and-place buttons.

All the logic (trajectory, planner, backends, PID, sequence, telemetry buffer) is plain
Python with no ROS imports, covered by 35 unit tests. The nodes are thin wrappers. The
**`ArmBackend` interface** (`write_command` / `advance` / `read_state`) is where hardware
plugs in: `position` and `pid` are two backends today, and Dynamixel would be a third.

**Stretch goals done:** PID control mode with a live error plot, pick-and-place as an
action with feedback and cancel, and unit tests.

## Key trade-offs

| Decision | Why | Cost |
|---|---|---|
| **Quintic** trajectories, duration set from velocity and acceleration limits | Zero acceleration at both ends, and it can start from a non-zero velocity, so re-targeting mid-move stays smooth | Slower than a trapezoid for the same peak velocity (peak is 1.875× the average) |
| **Check the whole path**, and re-plan through a straight-up pose if it dips below ground | Two valid end poses don't guarantee a valid path between them | The detour is longer than needed; a proper planner would find a shorter one |
| **Pick-and-place as an action in its own node** | Feedback and cancel come with the action; the controller stays a simple "move" primitive | Step completion is read from a 10 Hz status topic (up to 0.1 s late per step) and needs command ids to avoid races |
| **Pick/place targets are never projected** | Grasping at a projected point grasps nothing | An unreachable pick or place is rejected, not attempted |
| **Threads:** controller on a MultiThreadedExecutor; GUI runs ROS in a background thread and pulls a ring buffer at 30 Hz | Planning never stalls the 50 Hz loop, and Qt and ROS never block each other | Python's GIL still couples the threads (a long IK call can delay a tick by a few ms) |
| **PID mode:** constant decoupled inertia, no gravity feed-forward | Simple to explain, and the integral term visibly rejects gravity | Ignores the coupled mass matrix; 50 Hz limits the loop to about 10 rad/s |

**Issues I found in `planar_arm.py`** (left unmodified, handled in `planner.py`): the
Jacobian IK fallback ignores joint limits and the ground, the ground constraint can be
violated between two valid poses, and projection is only reported through `print()`.

## Next steps with more time

1. A **Dynamixel backend** (`dynamixel_sdk`, SyncRead/SyncWrite), starting the controller
   from the measured pose. Later, port it to a `ros2_control` hardware interface.
2. **Computed-torque control** with the full mass matrix and gravity feed-forward.
3. **Gazebo (`ros_gz`)** as one more backend.
4. **`launch_testing`** tests for the full system, and lifecycle nodes for clean bring-up.

---

# Part 2: From simulation to physical hardware

In simulation the arm does exactly what I tell it, instantly, and the timing is perfect.
On a real Dynamixel arm none of that holds. These are the bottlenecks I expect and how I
would handle them.

**1. Timing.** My 50 Hz loop runs in Python. Even in simulation I measured about 0.75 ms
of jitter, with some ticks around 7 ms late, most likely from the GIL. On WSL I also saw
the wall clock and the monotonic clock disagree by 2.5 s in one run. On hardware, jitter
turns into uneven motion. *Mitigation:* let the servo close the fast loop and only stream
setpoints from the host. If the host must close a loop, I would do it in C++
(`ros2_control`) on a real-time kernel. I already time trajectories on a monotonic clock.

**2. Communication.** Every read and write goes over a serial bus. Talking to each servo
separately at a low baud rate can eat most of a 20 ms cycle, and the delay adds lag to any
loop closed on the host. My PID mode already shows this limit: keeping ω·Ts below about
0.2 gives roughly 10 rad/s at 50 Hz. *Mitigation:* SyncRead/SyncWrite for all joints in
one packet, a higher baud rate, and velocity feed-forward from the quintic.

**3. Model mismatch.** Real joints have backlash, friction, compliance and unknown
payloads. My `position` backend hides all of this. The `pid` mode gives a hint: with no
gravity feed-forward, the integral term carries the load and the peak error is about
2–3° during moves. *Mitigation:* identify link masses and friction, feed gravity forward,
tune gains per joint on the real arm, and keep the integral term with anti-windup.

**4. Calibration and faults.** Zero offsets, direction signs, encoder resolution (about
0.09° per tick on a 4096-count servo) and link-length errors all show up as end-effector
error. Servos also overheat, overload and lose connection. *Mitigation:* calibrate, and
always act on measured state. My sequencer already checks the measured end-effector
position after every move and has timeouts, so a stalled joint aborts the sequence
cleanly. I would also publish servo error bits, temperature and voltage.

**5. Safety.** In simulation a bad command only costs a bad plot. *Mitigation:* soft
limits inside the hard limits, velocity and current caps, a watchdog that disables torque
if commands stop, and an e-stop. The first move should be slow and start from the
measured pose. Cancel already decelerates smoothly.

**What changes in my code.** A `DynamixelBackend` would implement `write_command` (goal
position, or goal current in `pid` mode) and `read_state` (present position, velocity and
current). The planner, sequencer and GUI logic stay the same. The controller would add
start-up from the measured pose and per-joint offset and direction parameters. The GUI
would add servo health, connection state and an e-stop button. It already shows data
age, measured rate and a reference-vs-measured "ghost" arm, which matter more on
hardware than in simulation.
