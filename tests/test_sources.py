from pathlib import Path

from app.adapters.sources import SourceKind, resolve_source_kind


def test_resolve_source_kind_for_screen_and_camera() -> None:
    assert resolve_source_kind("screen") is SourceKind.SCREEN
    assert resolve_source_kind("0") is SourceKind.CAMERA


def test_resolve_source_kind_for_existing_image(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"not a real image")

    assert resolve_source_kind(str(image_path)) is SourceKind.IMAGE


def test_resolve_source_kind_defaults_to_video() -> None:
    assert resolve_source_kind("video.mp4") is SourceKind.VIDEO
    assert resolve_source_kind("rtsp://example.com/media.mp4") is SourceKind.VIDEO

