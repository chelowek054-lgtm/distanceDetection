import argparse
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import ImageGrab
from ultralytics import YOLO


DEFAULT_MODEL = Path("models/yoloe-26x-seg.pt")
WINDOW_NAME = "YOLO Segmentation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an Ultralytics segmentation model on a camera, screen, video file, or video URL."
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help=f"Path to segmentation model weights. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--source",
        default="0",
        help='Input source: camera index like "0", "screen", video file path, or video URL.',
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
    return parser.parse_args()


def source_to_capture(source: str) -> cv2.VideoCapture:
    capture_source: int | str = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(capture_source)

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    return capture


def infer_and_show(model: YOLO, frame: np.ndarray, args: argparse.Namespace, previous_time: float) -> float:
    results = model.predict(
        frame,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )
    annotated_frame = results[0].plot()

    current_time = perf_counter()
    fps = 1.0 / max(current_time - previous_time, 1e-6)
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow(WINDOW_NAME, annotated_frame)
    return current_time


def run_screen(model: YOLO, args: argparse.Namespace) -> None:
    previous_time = perf_counter()

    while True:
        screenshot = ImageGrab.grab()
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        previous_time = infer_and_show(model, frame, args, previous_time)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def run_video_source(model: YOLO, args: argparse.Namespace) -> None:
    capture = source_to_capture(args.source)
    previous_time = perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            previous_time = infer_and_show(model, frame, args, previous_time)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"Model file was not found: {model_path}", file=sys.stderr)
        print("Place the weights at models/yoloe-26x-seg.pt or pass --model PATH.", file=sys.stderr)
        return 1

    model = YOLO(str(model_path))

    try:
        if args.source.lower() == "screen":
            run_screen(model, args)
        else:
            run_video_source(model, args)
    finally:
        cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
