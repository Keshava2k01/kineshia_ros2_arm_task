import numpy as np
import pytest

from planar_arm_control.backends import (
    PidSimBackend, SimulatedBackend, make_backend,
)
from planar_arm_control.control import ArmDynamics
from planar_arm_control.planar_arm import PlanarArm
from planar_arm_control.planner import MotionPlanner

ARM = PlanarArm([3.0, 2.0, 1.5])
READY = np.array([np.pi / 2, 0.0, 0.0])
DT = 0.02          # 50 Hz control rate


def _hold(backend, q, seconds):
    for _ in range(int(seconds / DT)):
        backend.write_command(q, np.zeros(3), np.zeros(3))
        backend.advance(DT)


def test_gravity_torque_signs():
    dyn = ArmDynamics(ARM, READY)
    # Straight up: no lever arm, no gravity torque.
    np.testing.assert_allclose(dyn.gravity_torque(READY), 0.0, atol=1e-9)
    # Horizontal to the right: gravity pulls the arm down (negative torque).
    tau = dyn.gravity_torque(np.radians([1e-9, 0.0, 0.0]))
    assert np.all(tau < 0)
    # Horizontal to the left: gravity torque on joint 1 is positive.
    assert dyn.gravity_torque(np.array([np.pi - 1e-9, 0.0, 0.0]))[0] > 0


def test_plant_respects_hard_stops():
    dyn = ArmDynamics(ARM, np.array([0.05, 0.0, 0.0]))
    for _ in range(100):
        dyn.step(np.array([-400.0, 0.0, 0.0]), DT)     # drive into the limit
    assert dyn.q[0] >= ARM.joint_limits[0, 0] - 1e-12


def test_integral_rejects_gravity_when_holding():
    q = np.radians([30.0, 20.0, -10.0])                 # loaded pose
    backend = PidSimBackend(ARM, q)
    _hold(backend, q, 4.0)
    assert np.max(np.abs(backend.last_error)) < 1e-3
    # Holding torque matches gravity (the I term found it).
    np.testing.assert_allclose(backend.plant.tau, -backend.plant.gravity_torque(q),
                               rtol=0.02, atol=0.5)


@pytest.mark.parametrize("targets", [[(4.0, 2.0), (-3.0, 3.0)], [(7.0, 3.0)]])
def test_tracks_scenario_moves(targets):
    planner = MotionPlanner(ARM)
    backend = PidSimBackend(ARM, READY)
    _hold(backend, READY, 1.0)
    q = READY
    for target in targets:
        traj = planner.plan(q, None, target).trajectory
        peak, t = 0.0, 0.0
        while t < traj.duration:
            backend.write_command(*traj.sample(t))
            backend.advance(DT)
            peak = max(peak, np.max(np.abs(backend.last_error)))
            assert np.all(np.abs(backend.plant.tau) <= backend.plant.tau_max)
            t += DT
        assert peak < np.radians(3.0), f"peak tracking error {np.degrees(peak):.2f} deg"
        _hold(backend, traj.end, 1.0)
        assert np.max(np.abs(backend.last_error)) < 1e-3
        assert ARM.arm_above_base(backend.plant.q)
        q = traj.end


def test_make_backend_modes():
    assert isinstance(make_backend("sim", "position", READY), SimulatedBackend)
    assert isinstance(make_backend("sim", "pid", READY, arm=ARM), PidSimBackend)
    with pytest.raises(ValueError):
        make_backend("sim", "velocity", READY)
    with pytest.raises(ValueError):
        make_backend("dynamixel", "position", READY)
