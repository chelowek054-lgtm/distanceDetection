from __future__ import annotations

import cv2

from app.domain.entities import InferenceConfig, PredictionResult


class OpenCVFrameRenderer:
    def __init__(self, config: InferenceConfig) -> None:
        self._config = config

    def render(self, prediction: PredictionResult, fps: float | None = None) -> None:
        annotated_frame = prediction.raw.plot(
            boxes=self._config.mode.show_boxes,
            masks=self._config.mode.show_masks,
        )

        if fps is not None:
            self._draw_fps(annotated_frame, fps)

        cv2.imshow(self._config.window_name, annotated_frame)

    def wait_key(self, delay_ms: int) -> int:
        return cv2.waitKey(delay_ms)

    def close(self) -> None:
        cv2.destroyAllWindows()

    def _draw_fps(self, frame, fps: float) -> None:
        label = "FPS: --" if fps <= 0 else f"FPS: {fps:.1f}"
        cv2.putText(
            frame,
            label,
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

