"""
backends.py  —  the hardware abstraction seam (no ROS).

The controller node streams setpoints to an ArmBackend and reads the measured
state back from it. The planner never touches a backend. To run on a real arm
(e.g. Dynamixel XM430s via dynamixel_sdk), write one new subclass that
implements write_command()/read_state(), register it in make_backend(), and
select it with the `backend` ROS parameter. Planner, node and GUI are
unchanged.
"""

from abc import ABC, abstractmethod

import numpy as np


class ArmBackend(ABC):
    """Interface every backend (simulated or physical) implements."""

    name = "abstract"

    @abstractmethod
    def write_command(self, q, qd, effort=None):
        """Send a position setpoint (with velocity feed-forward) [rad, rad/s]."""

    @abstractmethod
    def read_state(self):
        """Return measured (q, qd, effort) as numpy arrays (effort may be empty)."""

    def close(self):
        """Release hardware resources (torque off, close port, ...)."""


class SimulatedBackend(ArmBackend):
    """
    Ideal position-controlled joints: the measured state equals the last
    command. This is the "perfect servo" case; tracking error and dynamics
    are added by later control modes.
    """

    name = "sim"

    def __init__(self, q_init):
        self._q = np.asarray(q_init, dtype=float).copy()
        self._qd = np.zeros_like(self._q)

    def write_command(self, q, qd, effort=None):
        self._q = np.asarray(q, dtype=float).copy()
        self._qd = np.asarray(qd, dtype=float).copy()

    def read_state(self):
        return self._q.copy(), self._qd.copy(), np.array([])


_BACKENDS = {
    SimulatedBackend.name: SimulatedBackend,
}


def make_backend(name, q_init):
    try:
        cls = _BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {sorted(_BACKENDS)}") from None
    return cls(q_init)
