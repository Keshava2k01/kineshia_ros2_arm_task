import numpy as np
import pytest

from planar_arm_control.planar_arm import PlanarArm
from planar_arm_control.planner import MotionPlanner

READY = np.array([np.pi / 2, 0.0, 0.0])


@pytest.fixture
def planner():
    return MotionPlanner(PlanarArm([3.0, 2.0, 1.5]), v_max=1.0, a_max=2.0, t_min=0.5)


def _assert_path_safe(planner, traj):
    _, qs, qd, _ = traj.sample_many(400)
    for q in qs:
        assert planner.arm.within_joint_limits(q)
        assert planner.arm.arm_above_base(q)
    assert np.max(np.abs(qd)) <= planner.v_max * 1.02


def test_pick_target_reachable(planner):
    res = planner.plan(READY, None, (4.0, 2.0))
    assert res.success, res.message
    assert not res.projected
    np.testing.assert_allclose(planner.arm.end_effector(res.trajectory.end),
                               (4.0, 2.0), atol=1e-3)
    _assert_path_safe(planner, res.trajectory)


def test_pick_then_place(planner):
    pick = planner.plan(READY, None, (4.0, 2.0))
    place = planner.plan(pick.trajectory.end, None, (-3.0, 3.0))
    assert place.success, place.message
    np.testing.assert_allclose(planner.arm.end_effector(place.trajectory.end),
                               (-3.0, 3.0), atol=1e-3)
    _assert_path_safe(planner, place.trajectory)


def test_unreachable_target_is_projected_and_reported(planner):
    res = planner.plan(READY, None, (7.0, 3.0))
    assert res.success, res.message
    assert res.projected
    expected = np.array([7.0, 3.0]) * 6.5 / np.hypot(7.0, 3.0)
    np.testing.assert_allclose(res.goal_xy, expected, atol=1e-9)
    np.testing.assert_allclose(planner.arm.end_effector(res.trajectory.end),
                               expected, atol=1e-3)
    assert "outside the workspace" in res.message


def test_target_below_ground_rejected(planner):
    res = planner.plan(READY, None, (3.0, -1.0))
    assert not res.success


def test_retarget_mid_motion_is_continuous(planner):
    first = planner.plan(READY, None, (4.0, 2.0))
    t_mid = first.trajectory.duration / 2
    q_mid, qd_mid, _ = first.trajectory.sample(t_mid)
    second = planner.plan(q_mid, qd_mid, (-3.0, 3.0))
    assert second.success, second.message
    q, qd, _ = second.trajectory.sample(0.0)
    np.testing.assert_allclose(q, q_mid)
    np.testing.assert_allclose(qd, qd_mid)
    _assert_path_safe(planner, second.trajectory)


def test_short_requested_duration_is_stretched(planner):
    res = planner.plan(READY, None, (4.0, 2.0), duration=0.05)
    assert res.success
    assert res.duration > 0.05
    assert "exceeds joint limits" in res.message


def test_ground_violating_direct_path_uses_ready_pose(planner):
    """Random valid pose pairs whose straight joint path dips below y = 0."""
    rng = np.random.default_rng(0)
    lo, hi = planner.arm.joint_limits[:, 0], planner.arm.joint_limits[:, 1]

    def random_valid():
        while True:
            q = rng.uniform(lo, hi)
            if planner.pose_valid(q):
                return q

    rerouted = 0
    for _ in range(300):
        qa, qb = random_valid(), random_valid()
        res = planner.plan_joint(qa, None, qb)
        if res.success:
            _assert_path_safe(planner, res.trajectory)
            rerouted += res.via_ready_pose
    assert rerouted > 0, "expected at least one pair to need the ready pose"
