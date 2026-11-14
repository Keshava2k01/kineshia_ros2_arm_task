"""
trajectory.py  —  time-parameterised joint-space trajectories (no ROS).

Why quintic?
    A quintic polynomial lets us fix position, velocity AND acceleration at
    both ends of a segment. Rest-to-rest moves therefore start and stop with
    zero acceleration (no jerk spike), which is kinder to real gearboxes than a
    trapezoidal profile whose acceleration steps instantly. It also lets a new
    segment start from a non-zero velocity, so re-targeting mid-motion is
    smooth (position and velocity are continuous).

All joints share the same duration T, so they start and finish together and
the path is a straight line in joint space for rest-to-rest moves.
"""

import numpy as np

# Peak |velocity| and |acceleration| of a rest-to-rest quintic of length h
# over duration T are  V_PEAK * h / T  and  A_PEAK * h / T^2.
V_PEAK = 1.875               # 15/8, reached at t = T/2
A_PEAK = 10.0 / np.sqrt(3.0)  # ~5.7735, reached at t = T/2 -/+ T/(2*sqrt(3))


class Trajectory:
    """Common interface: a duration and a sample(t) -> (q, qd, qdd)."""

    duration = 0.0

    def sample(self, t):
        raise NotImplementedError

    @property
    def start(self):
        return self.sample(0.0)[0]

    @property
    def end(self):
        return self.sample(self.duration)[0]

    def sample_many(self, n):
        """Return n evenly spaced samples as arrays (t, q, qd, qdd)."""
        ts = np.linspace(0.0, self.duration, max(int(n), 2))
        q, qd, qdd = zip(*(self.sample(t) for t in ts))
        return ts, np.array(q), np.array(qd), np.array(qdd)


class QuinticTrajectory(Trajectory):
    """
    Per-joint quintic from (q0, qd0, qdd0=0) to (q1, qd1=0, qdd1=0) over T.

        q(t) = q0 + qd0 t + c3 t^3 + c4 t^4 + c5 t^5
    """

    def __init__(self, q0, q1, duration, qd0=None):
        self.q0 = np.asarray(q0, dtype=float).copy()
        self.q1 = np.asarray(q1, dtype=float).copy()
        if self.q0.shape != self.q1.shape:
            raise ValueError("q0 and q1 must have the same shape")
        self.qd0 = (np.zeros_like(self.q0) if qd0 is None
                    else np.asarray(qd0, dtype=float).copy())
        if duration <= 0.0:
            raise ValueError("duration must be > 0")
        self.duration = float(duration)

        T = self.duration
        h = self.q1 - self.q0
        v = self.qd0
        self.c3 = (10.0 * h - 6.0 * v * T) / T ** 3
        self.c4 = (-15.0 * h + 8.0 * v * T) / T ** 4
        self.c5 = (6.0 * h - 3.0 * v * T) / T ** 5

    def sample(self, t):
        t = min(max(float(t), 0.0), self.duration)
        q = (self.q0 + self.qd0 * t
             + self.c3 * t ** 3 + self.c4 * t ** 4 + self.c5 * t ** 5)
        qd = (self.qd0
              + 3.0 * self.c3 * t ** 2 + 4.0 * self.c4 * t ** 3
              + 5.0 * self.c5 * t ** 4)
        qdd = 6.0 * self.c3 * t + 12.0 * self.c4 * t ** 2 + 20.0 * self.c5 * t ** 3
        return q, qd, qdd


class SegmentedTrajectory(Trajectory):
    """Several trajectories played back-to-back (e.g. via a safe waypoint)."""

    def __init__(self, segments):
        if not segments:
            raise ValueError("need at least one segment")
        self.segments = list(segments)
        self._starts = np.cumsum([0.0] + [s.duration for s in self.segments[:-1]])
        self.duration = float(sum(s.duration for s in self.segments))

    def sample(self, t):
        t = min(max(float(t), 0.0), self.duration)
        i = int(np.searchsorted(self._starts, t, side="right")) - 1
        i = min(max(i, 0), len(self.segments) - 1)
        return self.segments[i].sample(t - self._starts[i])


def auto_duration(q0, q1, v_max, a_max, t_min=0.0):
    """
    Shortest duration for a rest-to-rest quintic that keeps every joint within
    v_max [rad/s] and a_max [rad/s^2].
    """
    h = float(np.max(np.abs(np.asarray(q1, float) - np.asarray(q0, float))))
    t_vel = V_PEAK * h / v_max if v_max > 0 else 0.0
    t_acc = np.sqrt(A_PEAK * h / a_max) if a_max > 0 else 0.0
    return max(t_min, t_vel, t_acc, 1e-3)
