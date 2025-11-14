#!/usr/bin/env python3
import os
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, get_args

import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

ClueType = Literal["SIZE", "VICTIM", "CRIME", "TIME", "PLACE", "MOTIVE", "WEAPON", "BANDIT"]
TYPES = list(get_args(ClueType))

banner_path = Path("gen_tools/clue_banner.png")
output_path = Path("plates_unmodified")
output_path.mkdir(parents=True, exist_ok=True)

banner_canvas = cv2.imread(str(banner_path))
PLATE_HEIGHT = 600
PLATE_WIDTH = 400


def save_banner(clue_type: ClueType, clue_text: str, save_path: Path):
    font_size = 90
    font_path = Path("gen_tools/UbuntuMono-Regular.ttf")
    blank_plate_pil = Image.fromarray(banner_canvas)
    draw = ImageDraw.Draw(blank_plate_pil)
    monospace = ImageFont.truetype(font_path, font_size)
    font_color = (255, 0, 0)

    # TYPE
    draw.text((250, 30), clue_type, font_color, font=monospace)
    # CLUE TEXT
    draw.text((30, 250), clue_text, font_color, font=monospace)

    populated_banner = np.array(blank_plate_pil)
    cv2.imwrite(str(save_path), populated_banner)


def random_clue_text():
    # either sample from the CSV list, or just roll your own
    # single word
    if random.random() < 0.5:
        length = random.randint(3, 8)
        return "".join(random.choice(string.ascii_uppercase) for _ in range(length))
    else:
        # two words
        length1 = random.randint(3, 7)
        length2 = min(11 - length1, random.randint(3, 7))

        w1 = "".join(random.choice(string.ascii_uppercase) for _ in range(length1))
        w2 = "".join(random.choice(string.ascii_uppercase) for _ in range(length2))
        return f"{w1} {w2}"


def _generate_plate(task: tuple[int, str, str]):
    index, key, value = task
    save_banner(key, value, output_path / f"{index}_{key}_{value}.png")


def main():
    NUM_PLATES = 10000  # or more
    print(f"Generating {NUM_PLATES} plates")
    tasks = [(i, random.choice(TYPES), random_clue_text()) for i in range(NUM_PLATES)]
    max_workers = os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_generate_plate, task) for task in tasks]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Generating plates",
            unit="plate",
        ):
            future.result()

if __name__ == "__main__":
    main()
