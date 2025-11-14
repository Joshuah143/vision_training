#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

NDArrayUInt8 = NDArray[np.uint8]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = SCRIPT_DIR / "fizz_yolo_dataset"
DEFAULT_PLATES_DIR = SCRIPT_DIR / "plates_unmodified"
DEFAULT_BACKGROUNDS_DIR = SCRIPT_DIR / "backgrounds"
DEFAULT_IMAGE_HEIGHT = 720
DEFAULT_IMAGE_WIDTH = 1280
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_AUG_PER_PLATE = 20
DEFAULT_FIZZ_CLASS_INDEX = 3
DEFAULT_CLASS_NAMES = ("person", "car", "truck", "fizz_sign")


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Configuration for synthetic dataset generation."""

    dataset_root: Path = field(default=DEFAULT_DATASET_ROOT)
    plates_dir: Path = field(default=DEFAULT_PLATES_DIR)
    backgrounds_dir: Path = field(default=DEFAULT_BACKGROUNDS_DIR)
    image_height: int = field(default=DEFAULT_IMAGE_HEIGHT)
    image_width: int = field(default=DEFAULT_IMAGE_WIDTH)
    train_ratio: float = field(default=DEFAULT_TRAIN_RATIO)
    augmentations_per_plate: int = field(default=DEFAULT_AUG_PER_PLATE)
    fizz_sign_class_index: int = field(default=DEFAULT_FIZZ_CLASS_INDEX)
    class_names: tuple[str, ...] = field(default=DEFAULT_CLASS_NAMES)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", self.dataset_root.resolve())
        object.__setattr__(self, "plates_dir", self.plates_dir.resolve())
        object.__setattr__(self, "backgrounds_dir", self.backgrounds_dir.resolve())

        if not 0.0 < self.train_ratio < 1.0:
            raise ValueError("train_ratio must be between 0 and 1 (exclusive)")
        if self.augmentations_per_plate <= 0:
            raise ValueError("augmentations_per_plate must be a positive integer")
        if self.fizz_sign_class_index >= len(self.class_names):
            raise ValueError(
                "fizz_sign_class_index must be less than the number of class_names"
            )
        if not self.plates_dir.exists():
            LOGGER.warning(
                "Plate directory %s does not exist; no samples will be generated until it is populated.",
                self.plates_dir,
            )

    @property
    def images_train_dir(self) -> Path:
        return self.dataset_root / "images" / "train"

    @property
    def images_val_dir(self) -> Path:
        return self.dataset_root / "images" / "val"

    @property
    def labels_train_dir(self) -> Path:
        return self.dataset_root / "labels" / "train"

    @property
    def labels_val_dir(self) -> Path:
        return self.dataset_root / "labels" / "val"

    @property
    def yaml_path(self) -> Path:
        return self.dataset_root / "fizz_dataset.yaml"

    @property
    def cv_image_size(self) -> tuple[int, int]:
        return self.image_width, self.image_height


def ensure_directories(config: DatasetConfig) -> None:
    """Create the dataset directory structure."""
    for directory in (
        config.images_train_dir,
        config.images_val_dir,
        config.labels_train_dir,
        config.labels_val_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def collect_plate_paths(plates_dir: Path) -> list[Path]:
    """Collect plate image paths from the provided directory."""
    valid_suffixes = {".png", ".jpg", ".jpeg"}
    return sorted(
        path
        for path in plates_dir.iterdir()
        if path.is_file() and path.suffix.lower() in valid_suffixes
    )


def load_backgrounds(config: DatasetConfig) -> list[NDArrayUInt8]:
    """Load and resize background images to the target resolution."""
    if not config.backgrounds_dir.exists():
        LOGGER.info(
            "Background directory %s not found; using synthetic backgrounds.",
            config.backgrounds_dir,
        )
        return []

    image_paths = sorted(
        list(config.backgrounds_dir.glob("*.png"))
        + list(config.backgrounds_dir.glob("*.jpg"))
        + list(config.backgrounds_dir.glob("*.jpeg"))
    )

    backgrounds: list[NDArrayUInt8] = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            LOGGER.warning("Skipping unreadable background image: %s", path)
            continue
        resized = cv2.resize(image, config.cv_image_size, interpolation=cv2.INTER_AREA)
        backgrounds.append(resized)

    return backgrounds


def random_background(
    backgrounds: Sequence[NDArrayUInt8],
    config: DatasetConfig,
    np_rng: np.random.Generator,
) -> NDArrayUInt8:
    """Return a random background image, falling back to synthetic noise."""
    if backgrounds:
        index = int(np_rng.integers(0, len(backgrounds)))
        return backgrounds[index].copy()

    return np_rng.integers(
        150,
        220,
        size=(config.image_height, config.image_width, 3),
        dtype=np.uint8,
    )


def random_affine_on_sign(
    sign_img: NDArrayUInt8, rng: random.Random
) -> tuple[NDArrayUInt8, NDArrayUInt8]:
    """Apply random scaling, rotation, and optional perspective warp to the sign.

    Returns
    -------
    tuple
        A tuple containing the transformed sign image and a binary mask indicating
        which pixels are valid sign content (255) versus background (0).
    """
    height, width = sign_img.shape[:2]

    scale = rng.uniform(0.5, 1.2)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(sign_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    mask = np.full((new_height, new_width), 255, dtype=np.uint8)

    angle = rng.uniform(-25.0, 25.0)
    rotation_matrix = cv2.getRotationMatrix2D(
        (new_width / 2.0, new_height / 2.0), angle, 1.0
    )
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    bound_width = int(new_height * abs_sin + new_width * abs_cos)
    bound_height = int(new_height * abs_cos + new_width * abs_sin)
    rotation_matrix[0, 2] += bound_width / 2.0 - new_width / 2.0
    rotation_matrix[1, 2] += bound_height / 2.0 - new_height / 2.0
    transformed = cv2.warpAffine(
        resized,
        rotation_matrix,
        (bound_width, bound_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = cv2.warpAffine(
        mask,
        rotation_matrix,
        (bound_width, bound_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    if rng.random() < 0.5:
        cur_height, cur_width = transformed.shape[:2]
        pts1 = np.float32(
            [
                [0, 0],
                [cur_width - 1, 0],
                [0, cur_height - 1],
                [cur_width - 1, cur_height - 1],
            ]
        )
        shift = 0.1 * float(min(cur_width, cur_height))
        pts2 = pts1 + np.float32(
            [[rng.uniform(-shift, shift), rng.uniform(-shift, shift)] for _ in range(4)]
        )
        perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)
        transformed = cv2.warpPerspective(
            transformed,
            perspective_matrix,
            (cur_width, cur_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mask = cv2.warpPerspective(
            mask,
            perspective_matrix,
            (cur_width, cur_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    non_zero_y, non_zero_x = np.nonzero(mask)
    if non_zero_x.size == 0 or non_zero_y.size == 0:
        return transformed, mask

    min_x, max_x = non_zero_x.min(), non_zero_x.max()
    min_y, max_y = non_zero_y.min(), non_zero_y.max()

    cropped_image = transformed[min_y : max_y + 1, min_x : max_x + 1]
    cropped_mask = mask[min_y : max_y + 1, min_x : max_x + 1]

    return cropped_image, cropped_mask


def add_noise_blur(
    image: NDArrayUInt8, rng: random.Random, np_rng: np.random.Generator
) -> NDArrayUInt8:
    """Apply random noise, blur, and brightness/contrast adjustments to the sign."""
    result = image.copy()

    if rng.random() < 0.7:
        noise = np_rng.normal(loc=0.0, scale=10.0, size=result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.6:
        kernel_size = int(rng.choice((3, 5)))
        result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)

    if rng.random() < 0.7:
        alpha = rng.uniform(0.7, 1.3)
        beta = rng.randint(-30, 30)
        result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)

    return result


def compute_yolo_bbox(
    x: int,
    y: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Compute a normalized YOLO bounding box."""
    x_center = (x + width / 2.0) / float(image_width)
    y_center = (y + height / 2.0) / float(image_height)
    w_norm = width / float(image_width)
    h_norm = height / float(image_height)
    return x_center, y_center, w_norm, h_norm


def write_dataset_yaml(config: DatasetConfig) -> None:
    """Write the dataset YAML file for YOLO training."""
    yaml_content = "\n".join(
        [
            f"path: {config.dataset_root}",
            "train: images/train",
            "val: images/val",
            "",
            f"nc: {len(config.class_names)}",
            f"names: {list(config.class_names)}",
            "",
        ]
    )
    config.yaml_path.write_text(yaml_content, encoding="utf-8")


def _process_plate(
    plate_index: int,
    plate_path: Path,
    config: DatasetConfig,
    backgrounds: list[NDArrayUInt8],
    seed: int | None,
    progress: tqdm,
) -> list[Path]:
    """Worker function to generate augmentations for a single plate image."""
    plate_img = cv2.imread(str(plate_path), cv2.IMREAD_COLOR)
    if plate_img is None:
        LOGGER.warning("Skipping unreadable plate image: %s", plate_path)
        progress.update(config.augmentations_per_plate)
        return []

    rng_seed = seed + plate_index if seed is not None else None
    rng = random.Random(rng_seed)
    np_rng = np.random.default_rng(rng_seed)

    generated: list[Path] = []
    for idx in range(config.augmentations_per_plate):
        progress.update()

        background = random_background(backgrounds, config, np_rng)
        transformed_sign, transformed_mask = random_affine_on_sign(plate_img, rng)
        if transformed_mask.max() == 0:
            LOGGER.debug(
                "Skipping sample with empty mask (%s idx=%d)",
                plate_path.name,
                idx,
            )
            continue

        augmented_sign = add_noise_blur(transformed_sign, rng, np_rng)
        sign_height, sign_width = augmented_sign.shape[:2]

        if sign_width > config.image_width or sign_height > config.image_height:
            LOGGER.debug(
                "Skipping sample with oversized sign (%s idx=%d)",
                plate_path.name,
                idx,
            )
            continue

        max_x = config.image_width - sign_width
        max_y = config.image_height - sign_height
        x = rng.randint(0, max_x) if max_x > 0 else 0
        y = rng.randint(0, max_y) if max_y > 0 else 0

        composite = background.copy()
        roi = composite[y : y + sign_height, x : x + sign_width]
        cv2.copyTo(augmented_sign, transformed_mask, roi)

        bbox = compute_yolo_bbox(
            x,
            y,
            sign_width,
            sign_height,
            config.image_width,
            config.image_height,
        )

        if rng.random() < config.train_ratio:
            img_dir = config.images_train_dir
            lbl_dir = config.labels_train_dir
        else:
            img_dir = config.images_val_dir
            lbl_dir = config.labels_val_dir

        suffix = f"{plate_path.stem}_{idx}"
        img_path = img_dir / f"{suffix}.jpg"
        lbl_path = lbl_dir / f"{suffix}.txt"

        if not cv2.imwrite(str(img_path), composite):
            LOGGER.error("Failed to write image %s", img_path)
            continue

        label_content = (
            f"{config.fizz_sign_class_index} "
            f"{bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
        )
        lbl_path.write_text(label_content, encoding="utf-8")
        generated.append(img_path)

    return generated


def generate_dataset(
    config: DatasetConfig | None = None, seed: int | None = None
) -> list[Path]:
    """Generate the synthetic dataset based on the provided configuration."""
    if config is None:
        config = DatasetConfig()

    ensure_directories(config)

    backgrounds = load_backgrounds(config)
    plate_paths = collect_plate_paths(config.plates_dir)
    if not plate_paths:
        LOGGER.warning("No plate images found in %s", config.plates_dir)
        return []

    generated_images: list[Path] = []

    total_attempts = len(plate_paths) * config.augmentations_per_plate
    max_workers = os.cpu_count() or 1
    with tqdm(
        total=total_attempts,
        desc="Generating synthetic images",
        unit="img",
    ) as progress:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _process_plate,
                    index,
                    plate_path,
                    config,
                    backgrounds,
                    seed,
                    progress,
                )
                for index, plate_path in enumerate(plate_paths)
            ]
            for future in as_completed(futures):
                generated_images.extend(future.result())

    write_dataset_yaml(config)
    LOGGER.info("Dataset created at %s", config.dataset_root)
    LOGGER.info("YAML written to %s", config.yaml_path)

    return generated_images


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a synthetic dataset for YOLO training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Directory where the dataset will be created.",
    )
    parser.add_argument(
        "--plates-dir",
        type=Path,
        default=DEFAULT_PLATES_DIR,
        help="Directory containing the base plate images.",
    )
    parser.add_argument(
        "--background-dir",
        type=Path,
        default=DEFAULT_BACKGROUNDS_DIR,
        help="Directory containing optional background images.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=DEFAULT_IMAGE_WIDTH,
        help="Width of the generated images in pixels.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=DEFAULT_IMAGE_HEIGHT,
        help="Height of the generated images in pixels.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Probability that a sample is assigned to the training split.",
    )
    parser.add_argument(
        "--augmentations-per-plate",
        type=int,
        default=DEFAULT_AUG_PER_PLATE,
        help="Number of augmented images generated per base plate.",
    )
    parser.add_argument(
        "--fizz-sign-class-index",
        type=int,
        default=DEFAULT_FIZZ_CLASS_INDEX,
        help="Class index to use for the fizz_sign label.",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=list(DEFAULT_CLASS_NAMES),
        help="Class names for the dataset in order.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible dataset generation.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the dataset generation script."""
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    config = DatasetConfig(
        dataset_root=args.dataset_root.resolve(),
        plates_dir=args.plates_dir.resolve(),
        backgrounds_dir=args.background_dir.resolve(),
        image_height=args.image_height,
        image_width=args.image_width,
        train_ratio=args.train_ratio,
        augmentations_per_plate=args.augmentations_per_plate,
        fizz_sign_class_index=args.fizz_sign_class_index,
        class_names=tuple(args.class_names),
    )

    generate_dataset(config, seed=args.seed)


if __name__ == "__main__":
    main()

