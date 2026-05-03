from app.use_cases.fps import PerfCounterFpsMeter


def test_fps_meter_returns_zero_for_first_frame() -> None:
    times = iter([10.0])
    meter = PerfCounterFpsMeter(timer=lambda: next(times))

    assert meter.tick() == 0.0


def test_fps_meter_calculates_elapsed_frame_rate() -> None:
    times = iter([1.0, 1.1, 1.3])
    meter = PerfCounterFpsMeter(timer=lambda: next(times))

    assert meter.tick() == 0.0
    assert round(meter.tick(), 2) == 10.0
    assert round(meter.tick(), 2) == 5.0

