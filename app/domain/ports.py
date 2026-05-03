from __future__ import annotations

from typing import Protocol

from app.domain.entities import Frame, PredictionResult


class FrameSource(Protocol):
    @property
    def is_single_frame(self) -> bool:
        pass

    def read(self) -> Frame | None:
        pass

    def release(self) -> None:
        pass


class InferenceEngine(Protocol):
    def predict(self, frame: Frame) -> PredictionResult:
        pass


class FrameRenderer(Protocol):
    def render(self, prediction: PredictionResult, fps: float | None = None) -> None:
        pass

    def wait_key(self, delay_ms: int) -> int:
        pass

    def close(self) -> None:
        pass


class FpsMeter(Protocol):
    def tick(self) -> float:
        pass

    def reset(self) -> None:
        pass

