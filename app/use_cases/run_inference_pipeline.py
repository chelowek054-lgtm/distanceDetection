from __future__ import annotations

from app.domain.ports import FpsMeter, FrameRenderer, FrameSource, InferenceEngine


class RunInferencePipeline:
    def __init__(
        self,
        source: FrameSource,
        inference_engine: InferenceEngine,
        renderer: FrameRenderer,
        fps_meter: FpsMeter,
    ) -> None:
        self._source = source
        self._inference_engine = inference_engine
        self._renderer = renderer
        self._fps_meter = fps_meter

    def run(self) -> int:
        try:
            while True:
                frame = self._source.read()
                if frame is None:
                    break

                prediction = self._inference_engine.predict(frame)
                fps = self._fps_meter.tick()
                self._renderer.render(prediction, fps=fps)

                delay_ms = 0 if self._source.is_single_frame else 1
                if self._renderer.wait_key(delay_ms) & 0xFF == ord("q"):
                    break
        finally:
            self._source.release()
            self._renderer.close()

        return 0

