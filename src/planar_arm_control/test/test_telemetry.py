import threading

import numpy as np
import pytest

from planar_arm_control.telemetry import TelemetryBuffer


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_empty_buffer():
    buf = TelemetryBuffer()
    t, q, qd = buf.snapshot()
    assert t.size == 0
    assert buf.latest() is None
    assert buf.rate_hz() == 0.0
    assert buf.age() == float("inf")


def test_window_drops_old_samples():
    clock = FakeClock()
    buf = TelemetryBuffer(window_s=1.0, clock=clock)
    for i in range(100):           # 2 s at 50 Hz
        clock.t = i * 0.02
        buf.append([i, 0, 0])
    t, q, _ = buf.snapshot()
    assert t[-1] - t[0] <= 1.0 + 1e-9
    assert q[-1, 0] == 99
    assert q.shape[1] == 3


def test_rate_and_age():
    clock = FakeClock()
    buf = TelemetryBuffer(clock=clock)
    for i in range(200):
        clock.t = i * 0.02
        buf.append([0, 0, 0], [0, 0, 0])
    assert buf.rate_hz() == pytest.approx(50.0, rel=0.03)
    clock.t += 0.3
    assert buf.age() == pytest.approx(0.3)


def test_missing_velocity_becomes_zeros():
    buf = TelemetryBuffer()
    buf.append([1.0, 2.0, 3.0], [])
    _, q, qd = buf.latest()
    np.testing.assert_allclose(qd, 0.0)


def test_concurrent_append_and_snapshot():
    buf = TelemetryBuffer(window_s=0.05)
    stop = threading.Event()
    errors = []

    def writer():
        i = 0
        while not stop.is_set():
            buf.append([i, i, i], [0, 0, 0])
            i += 1

    def reader():
        try:
            for _ in range(2000):
                t, q, qd = buf.snapshot()
                assert len(t) == len(q) == len(qd)
        except Exception as e:      # pragma: no cover - reported below
            errors.append(e)

    w = threading.Thread(target=writer)
    w.start()
    reader()
    stop.set()
    w.join()
    assert not errors
