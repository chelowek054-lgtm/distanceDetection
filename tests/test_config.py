from pathlib import Path

from app.domain.entities import PredictionMode
from app.infrastructure.config import build_config, parse_args, parse_text_prompts


def test_parse_text_prompts_trims_empty_values() -> None:
    assert parse_text_prompts(" person, bus ,,traffic light ") == (
        "person",
        "bus",
        "traffic light",
    )


def test_build_config_maps_cli_values() -> None:
    args = parse_args(
        [
            "--model",
            "models/custom.pt",
            "--source",
            "screen",
            "--mode",
            "detect",
            "--text",
            "person,bus",
            "--text-model",
            "models/mobileclip2_b.ts",
            "--prompt-image",
            "reference.jpg",
            "--conf",
            "0.5",
            "--imgsz",
            "320",
            "--device",
            "cpu",
            "--window-name",
            "Demo",
        ]
    )

    config = build_config(args)

    assert config.model_path == Path("models/custom.pt")
    assert config.source == "screen"
    assert config.mode is PredictionMode.DETECT
    assert config.text_prompt.classes == ("person", "bus")
    assert config.text_prompt.model_path == Path("models/mobileclip2_b.ts")
    assert config.visual_prompt.reference_image == Path("reference.jpg")
    assert config.conf == 0.5
    assert config.imgsz == 320
    assert config.device == "cpu"
    assert config.window_name == "Demo"

