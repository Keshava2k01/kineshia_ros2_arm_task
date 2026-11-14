import numpy as np
import pytest

from planar_arm_control.trajectory import (
    A_PEAK, V_PEAK, QuinticTrajectory, SegmentedTrajectory, auto_duration,
)

Q0 = np.array([0.5, -0.3, 0.2])
Q1 = np.array([1.5, 0.4, -0.6])


def test_rest_to_rest_boundary_conditions():
    traj = QuinticTrajectory(Q0, Q1, 2.0)
    q, qd, qdd = traj.sample(0.0)
    np.testing.assert_allclose(q, Q0)
    np.testing.assert_allclose(qd, 0.0, atol=1e-12)
    np.testing.assert_allclose(qdd, 0.0, atol=1e-12)
    q, qd, qdd = traj.sample(2.0)
    np.testing.assert_allclose(q, Q1)
    np.testing.assert_allclose(qd, 0.0, atol=1e-12)
    np.testing.assert_allclose(qdd, 0.0, atol=1e-12)


def test_sample_is_clamped_outside_duration():
    traj = QuinticTrajectory(Q0, Q1, 1.0)
    np.testing.assert_allclose(traj.sample(-1.0)[0], Q0)
    np.testing.assert_allclose(traj.sample(5.0)[0], Q1)


def test_peak_values_match_closed_form():
    T = 1.7
    traj = QuinticTrajectory(Q0, Q1, T)
    _, _, qd, qdd = traj.sample_many(5001)
    h = np.abs(Q1 - Q0)
    np.testing.assert_allclose(np.max(np.abs(qd), axis=0), V_PEAK * h / T, rtol=1e-4)
    np.testing.assert_allclose(np.max(np.abs(qdd), axis=0), A_PEAK * h / T ** 2, rtol=1e-3)


def test_auto_duration_respects_limits():
    v_max, a_max = 0.8, 1.5
    T = auto_duration(Q0, Q1, v_max, a_max)
    _, _, qd, qdd = QuinticTrajectory(Q0, Q1, T).sample_many(5001)
    assert np.max(np.abs(qd)) <= v_max * 1.0001
    assert np.max(np.abs(qdd)) <= a_max * 1.0001
    assert auto_duration(Q0, Q0, v_max, a_max, t_min=0.5) == 0.5


def test_nonzero_start_velocity_is_continuous():
    v0 = np.array([0.4, -0.2, 0.1])
    traj = QuinticTrajectory(Q0, Q1, 2.0, qd0=v0)
    q, qd, _ = traj.sample(0.0)
    np.testing.assert_allclose(q, Q0)
    np.testing.assert_allclose(qd, v0)
    np.testing.assert_allclose(traj.sample(2.0)[1], 0.0, atol=1e-12)


def test_segmented_trajectory_joins_segments():
    mid = np.array([1.0, 0.0, 0.0])
    a = QuinticTrajectory(Q0, mid, 1.0)
    b = QuinticTrajectory(mid, Q1, 2.0)
    seg = SegmentedTrajectory([a, b])
    assert seg.duration == pytest.approx(3.0)
    np.testing.assert_allclose(seg.sample(0.0)[0], Q0)
    np.testing.assert_allclose(seg.sample(1.0)[0], mid)
    np.testing.assert_allclose(seg.sample(3.0)[0], Q1)
    np.testing.assert_allclose(seg.sample(2.0)[0], b.sample(1.0)[0])


def test_invalid_duration_rejected():
    with pytest.raises(ValueError):
        QuinticTrajectory(Q0, Q1, 0.0)
