from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from app.domain.entities import (
    InferenceConfig,
    PredictionMode,
    TextPromptConfig,
    VisualPromptConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


def load_env_file(env_file: Path = ENV_FILE) -> None:
    if not env_file.exists():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()


MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
DEFAULT_MODEL = MODELS_DIR / os.getenv("YOLOE_WEIGHTS_FILE", "yoloe-26x-seg.pt")
DEFAULT_TEXT_MODEL = MODELS_DIR / os.getenv("YOLOE_TEXT_MODEL_FILE", "mobileclip2_b.ts")
DEFAULT_WINDOW_NAME = "YOLO Segmentation"


def parse_text_prompts(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()

    return tuple(prompt.strip() for prompt in value.split(",") if prompt.strip())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an Ultralytics YOLO/YOLOE model on a camera, screen, image, video file, or URL."
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help=f"Path to model weights. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--source",
        default="0",
        help='Input source: camera index like "0", "screen", image/video path, or URL.',
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in PredictionMode],
        default=PredictionMode.BOTH.value,
        help="Visualization mode: bbox detection, instance segmentation, or both.",
    )
    parser.add_argument(
        "--text",
        default=None,
        help='Comma-separated YOLOE text prompts, for example "person,bus".',
    )
    parser.add_argument(
        "--text-model",
        default=str(DEFAULT_TEXT_MODEL),
        help=f"Path to YOLOE text encoder TorchScript weights. Default: {DEFAULT_TEXT_MODEL}",
    )
    parser.add_argument(
        "--prompt-image",
        default=None,
        help="Reference image for future YOLOE visual prompt support. Requires bbox provider in a later UI/API step.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Device for inference, for example "cpu", "0", or "cuda:0". Default: Ultralytics auto-select.',
    )
    parser.add_argument(
        "--window-name",
        default=DEFAULT_WINDOW_NAME,
        help=f'OpenCV output window name. Default: "{DEFAULT_WINDOW_NAME}".',
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> InferenceConfig:
    prompt_image = Path(args.prompt_image) if args.prompt_image else None

    return InferenceConfig(
        model_path=Path(args.model),
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        mode=PredictionMode(args.mode),
        window_name=args.window_name,
        text_prompt=TextPromptConfig(
            classes=parse_text_prompts(args.text),
            model_path=Path(args.text_model),
        ),
        visual_prompt=VisualPromptConfig(reference_image=prompt_image),
    )

