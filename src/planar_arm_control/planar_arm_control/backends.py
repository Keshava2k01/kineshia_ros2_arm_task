"""
backends.py  —  the hardware abstraction seam (no ROS).

The controller node streams reference setpoints to an ArmBackend and reads the
measured state back from it. The planner never touches a backend. To run on a
real arm (e.g. Dynamixel XM430s via dynamixel_sdk), write one new subclass that
implements write_command()/read_state(), register it in make_backend(), and
select it with the `backend` / `control_mode` ROS parameters. Planner, node and
GUI are unchanged.

Control modes (simulated backend)
    position  SimulatedBackend: ideal position servo (measured == reference).
              Equivalent to a Dynamixel in position mode, which closes its
              own loop on the servo.
    pid       PidSimBackend: torque-level PID tracking on a simulated plant
              with inertia, friction, gravity and torque limits. Equivalent to
              running the loop on the host and sending current/torque
              commands (Dynamixel current-control mode).
"""

from abc import ABC, abstractmethod

import numpy as np

from planar_arm_control.control import ArmDynamics, JointPid


class ArmBackend(ABC):
    """Interface every backend (simulated or physical) implements."""

    name = "abstract"

    @abstractmethod
    def write_command(self, q, qd, qdd=None):
        """Send the reference: position [rad], velocity [rad/s], accel [rad/s^2]."""

    def advance(self, dt):
        """Advance a simulated plant by dt [s]. Real hardware: no-op."""

    @abstractmethod
    def read_state(self):
        """Return measured (q, qd, effort) as numpy arrays (effort may be empty)."""

    def close(self):
        """Release hardware resources (torque off, close port, ...)."""


class SimulatedBackend(ArmBackend):
    """
    Ideal position-controlled joints: the measured state equals the last
    command (the "perfect servo" case).
    """

    name = "sim/position"

    def __init__(self, q_init):
        self._q = np.asarray(q_init, dtype=float).copy()
        self._qd = np.zeros_like(self._q)

    def write_command(self, q, qd, qdd=None):
        self._q = np.asarray(q, dtype=float).copy()
        self._qd = np.asarray(qd, dtype=float).copy()

    def read_state(self):
        return self._q.copy(), self._qd.copy(), np.array([])


class PidSimBackend(ArmBackend):
    """
    Host-side PID computing joint torques at the control rate (zero-order
    hold between ticks), driving the ArmDynamics plant.
    """

    name = "sim/pid"

    def __init__(self, arm, q_init, bandwidth=10.0, damping=1.0, integral_ratio=0.3):
        self.plant = ArmDynamics(arm, q_init)
        self.pid = JointPid(self.plant.J, self.plant.tau_max, bandwidth=bandwidth,
                            damping=damping, integral_ratio=integral_ratio)
        q0 = np.asarray(q_init, dtype=float)
        self._ref = (q0.copy(), np.zeros_like(q0), np.zeros_like(q0))
        self.last_error = np.zeros_like(q0)

    def write_command(self, q, qd, qdd=None):
        q = np.asarray(q, dtype=float)
        self._ref = (q, np.asarray(qd, dtype=float),
                     np.zeros_like(q) if qdd is None else np.asarray(qdd, dtype=float))

    def advance(self, dt):
        if dt <= 0.0:
            return
        q_ref, qd_ref, qdd_ref = self._ref
        tau, self.last_error = self.pid.update(q_ref, qd_ref, qdd_ref,
                                               self.plant.q, self.plant.qd, dt)
        self.plant.step(tau, dt)

    def read_state(self):
        return self.plant.q.copy(), self.plant.qd.copy(), self.plant.tau.copy()


CONTROL_MODES = ("position", "pid")


def make_backend(backend, control_mode, q_init, arm=None, **pid_kwargs):
    if backend != "sim":
        raise ValueError(f"Unknown backend '{backend}'. Available: ['sim'] "
                         "(a hardware backend would be registered here).")
    if control_mode == "position":
        return SimulatedBackend(q_init)
    if control_mode == "pid":
        if arm is None:
            raise ValueError("pid control mode needs the arm model")
        return PidSimBackend(arm, q_init, **pid_kwargs)
    raise ValueError(f"Unknown control_mode '{control_mode}'. "
                     f"Available: {list(CONTROL_MODES)}")
