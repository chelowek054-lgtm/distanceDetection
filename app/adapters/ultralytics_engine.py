from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO

try:
    from ultralytics import YOLOE
except ImportError:  # pragma: no cover - depends on installed ultralytics version
    YOLOE = None

from app.domain.entities import Frame, InferenceConfig, PredictionResult


class UltralyticsYoloEEngine:
    def __init__(self, config: InferenceConfig) -> None:
        if config.visual_prompt.enabled:
            raise NotImplementedError(
                "Visual prompt requires a bbox provider for the reference image. "
                "The architecture supports this mode, but CLI/UI bbox selection is a later step."
            )

        self._config = config
        self._model = self._load_model()
        self._configure_text_prompt()

    def predict(self, frame: Frame) -> PredictionResult:
        results = self._model.predict(
            frame.image,
            conf=self._config.conf,
            imgsz=self._config.imgsz,
            device=self._config.device,
            verbose=False,
        )
        result = results[0]

        return PredictionResult(
            raw=result,
            boxes=getattr(result, "boxes", None),
            masks=getattr(result, "masks", None),
        )

    def _load_model(self) -> Any:
        model_path = str(self._config.model_path)
        if "yoloe" in self._config.model_path.name.lower() and YOLOE is not None:
            return YOLOE(model_path)

        return YOLO(model_path)

    def _configure_text_prompt(self) -> None:
        if not self._config.text_prompt.enabled:
            return

        if not hasattr(self._model, "set_classes"):
            raise RuntimeError("Text prompts require a YOLOE model with set_classes support.")

        text_model_path = self._config.text_prompt.model_path
        if not text_model_path.exists():
            raise RuntimeError(f"Text model file was not found: {text_model_path}")

        embeddings = self._build_text_embeddings(text_model_path)
        self._model.set_classes(list(self._config.text_prompt.classes), embeddings=embeddings)

    def _build_text_embeddings(self, text_model_path: Path) -> Any:
        import torch
        from ultralytics.nn.text_model import MobileCLIPTS

        yoloe_model = self._model.model
        device = next(yoloe_model.model.parameters()).device
        text_model = MobileCLIPTS(device=device, weight=str(text_model_path))
        text_tokens = text_model.tokenize(list(self._config.text_prompt.classes))

        with torch.inference_mode():
            text_features = [
                text_model.encode_text(token).detach()
                for token in text_tokens.split(80)
            ]
            text_features = (
                text_features[0]
                if len(text_features) == 1
                else torch.cat(text_features, dim=0)
            )
            text_features = text_features.reshape(
                -1,
                len(self._config.text_prompt.classes),
                text_features.shape[-1],
            )

            return yoloe_model.model[-1].get_tpe(text_features)

