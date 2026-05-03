from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class PredictionMode(str, Enum):
    SEGMENT = "segment"
    DETECT = "detect"
    BOTH = "both"

    @property
    def show_boxes(self) -> bool:
        return True

    @property
    def show_masks(self) -> bool:
        return self is not PredictionMode.DETECT


@dataclass(frozen=True)
class TextPromptConfig:
    classes: tuple[str, ...] = ()
    model_path: Path = Path("models/mobileclip2_b.ts")

    @property
    def enabled(self) -> bool:
        return bool(self.classes)


@dataclass(frozen=True)
class VisualPromptConfig:
    reference_image: Path | None = None
    bboxes: tuple[tuple[float, float, float, float], ...] = ()
    classes: tuple[int, ...] = ()

    @property
    def enabled(self) -> bool:
        return self.reference_image is not None or bool(self.bboxes)


@dataclass(frozen=True)
class InferenceConfig:
    model_path: Path
    source: str
    conf: float = 0.25
    imgsz: int = 640
    device: str | None = None
    mode: PredictionMode = PredictionMode.BOTH
    window_name: str = "YOLO Segmentation"
    text_prompt: TextPromptConfig = field(default_factory=TextPromptConfig)
    visual_prompt: VisualPromptConfig = field(default_factory=VisualPromptConfig)


@dataclass(frozen=True)
class Frame:
    image: Any
    index: int
    source_name: str


@dataclass(frozen=True)
class PredictionResult:
    raw: Any
    boxes: Any
    masks: Any

