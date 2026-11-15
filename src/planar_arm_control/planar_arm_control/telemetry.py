"""
telemetry.py  —  thread-safe rolling buffer of joint states (no ROS, no Qt).

The ROS executor thread appends at the message rate (50 Hz); the Qt thread
takes snapshots at the render rate (30 Hz). The two rates are independent:
high-rate data never goes through the Qt event queue, so a faster publisher
cannot flood the GUI, and a slow redraw cannot back-pressure ROS.
"""

import threading
import time
from collections import deque

import numpy as np


class TelemetryBuffer:

    def __init__(self, window_s=20.0, clock=time.monotonic):
        self.window_s = float(window_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._t = deque()       # receive time (local monotonic clock) [s]
        self._q = deque()
        self._qd = deque()

    def append(self, q, qd=None, t=None):
        t = self._clock() if t is None else float(t)
        q = np.asarray(q, dtype=float)
        qd = np.zeros_like(q) if qd is None or len(qd) != len(q) else np.asarray(qd, float)
        with self._lock:
            self._t.append(t)
            self._q.append(q)
            self._qd.append(qd)
            cutoff = t - self.window_s
            while self._t and self._t[0] < cutoff:
                self._t.popleft()
                self._q.popleft()
                self._qd.popleft()

    def snapshot(self):
        """Copy of the buffer as arrays (t [N], q [N, n], qd [N, n])."""
        with self._lock:
            if not self._t:
                return np.empty(0), np.empty((0, 0)), np.empty((0, 0))
            return np.array(self._t), np.array(self._q), np.array(self._qd)

    def latest(self):
        """(t, q, qd) of the newest sample, or None if empty."""
        with self._lock:
            if not self._t:
                return None
            return self._t[-1], self._q[-1].copy(), self._qd[-1].copy()

    def age(self):
        """Seconds since the newest sample (inf if empty)."""
        with self._lock:
            return self._clock() - self._t[-1] if self._t else float("inf")

    def rate_hz(self, horizon_s=1.0):
        """Measured message rate over the last horizon_s seconds."""
        with self._lock:
            if len(self._t) < 2:
                return 0.0
            now = self._t[-1]
            n = 0
            for t in reversed(self._t):
                if now - t > horizon_s:
                    break
                n += 1
            first = self._t[-n]
            span = now - first
            return (n - 1) / span if span > 0 else 0.0
