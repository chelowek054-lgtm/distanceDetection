from __future__ import annotations

from enum import Enum
from pathlib import Path

from app.domain.entities import Frame
from app.domain.ports import FrameSource


class SourceKind(str, Enum):
    CAMERA = "camera"
    SCREEN = "screen"
    IMAGE = "image"
    VIDEO = "video"


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def resolve_source_kind(source: str) -> SourceKind:
    if source.lower() == "screen":
        return SourceKind.SCREEN

    if source.isdigit():
        return SourceKind.CAMERA

    path = Path(source)
    if path.exists() and path.suffix.lower() in IMAGE_SUFFIXES:
        return SourceKind.IMAGE

    return SourceKind.VIDEO


class OpenCVCameraSource:
    def __init__(self, camera_index: int, source_name: str | None = None) -> None:
        import cv2

        self._source_name = source_name or str(camera_index)
        self._capture = cv2.VideoCapture(camera_index)
        self._frame_index = 0

        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open camera source: {self._source_name}")

    @property
    def is_single_frame(self) -> bool:
        return False

    def read(self) -> Frame | None:
        ok, image = self._capture.read()
        if not ok:
            return None

        frame = Frame(image=image, index=self._frame_index, source_name=self._source_name)
        self._frame_index += 1
        return frame

    def release(self) -> None:
        self._capture.release()


class OpenCVVideoSource:
    def __init__(self, source: str) -> None:
        import cv2

        self._source_name = source
        self._capture = cv2.VideoCapture(source)
        self._frame_index = 0

        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

    @property
    def is_single_frame(self) -> bool:
        return False

    def read(self) -> Frame | None:
        ok, image = self._capture.read()
        if not ok:
            return None

        frame = Frame(image=image, index=self._frame_index, source_name=self._source_name)
        self._frame_index += 1
        return frame

    def release(self) -> None:
        self._capture.release()


class ScreenSource:
    def __init__(self) -> None:
        self._frame_index = 0

    @property
    def is_single_frame(self) -> bool:
        return False

    def read(self) -> Frame | None:
        import cv2
        import numpy as np
        from PIL import ImageGrab

        screenshot = ImageGrab.grab()
        image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        frame = Frame(image=image, index=self._frame_index, source_name="screen")
        self._frame_index += 1
        return frame

    def release(self) -> None:
        return None


class ImageSource:
    def __init__(self, source: str) -> None:
        import cv2

        self._source_name = source
        self._image = cv2.imread(source)
        self._consumed = False

        if self._image is None:
            raise RuntimeError(f"Could not open image source: {source}")

    @property
    def is_single_frame(self) -> bool:
        return True

    def read(self) -> Frame | None:
        if self._consumed:
            return None

        self._consumed = True
        return Frame(image=self._image, index=0, source_name=self._source_name)

    def release(self) -> None:
        return None


def create_frame_source(source: str) -> FrameSource:
    source_kind = resolve_source_kind(source)

    if source_kind is SourceKind.SCREEN:
        return ScreenSource()
    if source_kind is SourceKind.CAMERA:
        return OpenCVCameraSource(int(source), source_name=source)
    if source_kind is SourceKind.IMAGE:
        return ImageSource(source)

    return OpenCVVideoSource(source)

