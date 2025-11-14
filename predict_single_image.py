#!/usr/bin/env python3
"""Run a trained YOLO model on a single image and visualize detections."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a trained YOLO model on a single image, draw detections, "
            "and save the annotated output."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to the trained YOLO weights (e.g., best.pt).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input image to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the annotated image. Defaults to <image>_pred.jpg.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum confidence threshold for displaying detections.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size passed to YOLO.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (e.g., 'cpu', '0'). Defaults to YOLO's auto selection.",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=2,
        help="Bounding box line thickness.",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=0.6,
        help="Font scale for label text.",
    )
    parser.add_argument(
        "--font-thickness",
        type=int,
        default=1,
        help="Font thickness for label text.",
    )
    parser.add_argument(
        "--line-merge-pixels",
        type=float,
        default=25.0,
        help="Vertical tolerance (in pixels) when grouping letters into lines.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the annotated image in a window.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )


def load_image(path: Path) -> NDArray[np.uint8]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def draw_detections(
    image: NDArray[np.uint8],
    boxes: NDArray[np.float32],
    confidences: Sequence[float],
    class_ids: Sequence[int],
    class_names: Sequence[str],
    line_thickness: int,
    font_scale: float,
    font_thickness: int,
) -> NDArray[np.uint8]:
    annotated = image.copy()
    for (x1, y1, x2, y2), conf, cls in zip(boxes, confidences, class_ids):
        color = (0, 255, 0)
        cv2.rectangle(
            annotated,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            thickness=line_thickness,
        )

        label = f"{class_names[cls] if cls < len(class_names) else cls}: {conf:.2f}"
        text_size, _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        text_origin = (int(x1), max(int(y1) - 5, text_size[1] + 5))

        cv2.rectangle(
            annotated,
            (text_origin[0], text_origin[1] - text_size[1] - 4),
            (text_origin[0] + text_size[0], text_origin[1] + 4),
            color,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            label,
            (text_origin[0], text_origin[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    return annotated


def extract_letter_detections(
    boxes: NDArray[np.float32],
    class_ids: Sequence[int],
    class_names: Sequence[str],
) -> list[tuple[str, float, float]]:
    """Return (char, x_center, y_center) for detected letter classes."""
    letters: list[tuple[str, float, float]] = []
    for (x1, y1, x2, y2), cls in zip(boxes, class_ids):
        if cls < 0 or cls >= len(class_names):
            continue
        name = class_names[cls]
        if not name.startswith("letter_") or len(name) <= len("letter_"):
            continue
        char = name.split("_", 1)[1]
        x_center = float((x1 + x2) / 2.0)
        y_center = float((y1 + y2) / 2.0)
        letters.append((char, x_center, y_center))
    return letters


def group_letters_into_lines(
    letters: Sequence[tuple[str, float, float]],
    vertical_tolerance: float,
) -> list[str]:
    """Group letters into lines by vertical proximity, then sort left-to-right."""
    if not letters:
        return []

    sorted_letters = sorted(letters, key=lambda item: item[2])
    line_entries: list[list[tuple[str, float, float]]] = []
    line_means: list[float] = []

    for char, x_center, y_center in sorted_letters:
        assigned = False
        for idx, mean_y in enumerate(line_means):
            if abs(y_center - mean_y) <= vertical_tolerance:
                entries = line_entries[idx]
                entries.append((char, x_center, y_center))
                line_means[idx] = ((mean_y * (len(entries) - 1)) + y_center) / len(entries)
                assigned = True
                break
        if not assigned:
            line_entries.append([(char, x_center, y_center)])
            line_means.append(y_center)

    output_lines: list[str] = []
    for idx in sorted(range(len(line_entries)), key=lambda i: line_means[i]):
        ordered = sorted(line_entries[idx], key=lambda entry: entry[1])
        output_lines.append("".join(char for char, _, _ in ordered))

    return output_lines


def predict_single_image(args: argparse.Namespace) -> Path:
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights file not found: {args.weights}")
    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    model = YOLO(str(args.weights))
    LOGGER.info("Loaded model from %s", args.weights)

    raw_results = model.predict(
        source=str(args.image),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=False,
        verbose=False,
    )
    if not raw_results:
        raise RuntimeError("YOLO returned no results.")

    result = raw_results[0]
    if result.boxes is None or result.boxes.xyxy is None:
        LOGGER.warning("No detections found above confidence threshold %.2f", args.conf)
        boxes_np = np.empty((0, 4), dtype=np.float32)
        confs_np: list[float] = []
        class_ids_np: list[int] = []
    else:
        boxes_np = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        confs_np = result.boxes.conf.cpu().numpy().astype(float).tolist()
        class_ids_np = result.boxes.cls.cpu().numpy().astype(int).tolist()

    image = load_image(args.image)
    class_names = tuple(result.names.values()) if result.names else tuple()

    letter_detections = extract_letter_detections(boxes_np, class_ids_np, class_names)
    letter_lines = group_letters_into_lines(letter_detections, args.line_merge_pixels)
    if letter_lines:
        LOGGER.info("Detected letters (top to bottom):")
        for line in letter_lines:
            LOGGER.info("  %s", line)
    else:
        LOGGER.info("No letter detections found.")

    annotated = draw_detections(
        image=image,
        boxes=boxes_np,
        confidences=confs_np,
        class_ids=class_ids_np,
        class_names=class_names,
        line_thickness=args.line_thickness,
        font_scale=args.font_scale,
        font_thickness=args.font_thickness,
    )

    if args.output is None:
        default_suffix = args.image.suffix if args.image.suffix else ".jpg"
        output_filename = f"{args.image.stem}_pred{default_suffix}"
        output_path = args.image.with_name(output_filename)
    else:
        output_path = args.output
    if output_path.suffix.lower() == "":
        output_path = output_path.with_suffix(".jpg")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Failed to write annotated image to {output_path}")
    LOGGER.info("Annotated image written to %s", output_path)

    if args.show:
        cv2.imshow("Detections", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return output_path


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    predict_single_image(args)


if __name__ == "__main__":
    main()

