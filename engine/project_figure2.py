#!/usr/bin/env python3
"""
Generate qualitative samples that show how the training data looks before and
after the augmentation pipeline. For N randomly selected train samples we save:

1. Raw RGB image (straight from disk, before resize/augment/standardize)
2. Augmented model input (grayscale, destandardized for visualization)
3. Augmented 2D heatmap produced from the transformed label
"""

import os
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import yaml

# Allow local package imports when the script is called directly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_loader.guidewire_data_loader import (  # noqa: E402
    GuidewireDataPreprocessor,
    GuidewireDataSet,
)


def parse_args():
    default_config = os.path.join(PROJECT_ROOT, 'config', 'ver3.yaml')
    default_dataset = os.path.join(PROJECT_ROOT, 'datasets', 'guidewire')
    default_output = os.path.join(PROJECT_ROOT, 'results', 'figure2_assets')

    parser = ArgumentParser(description="Export raw vs augmented samples + heatmaps from the train split.")
    parser.add_argument('--config', type=str, default=default_config,
                        help='Path to the YAML config file.')
    parser.add_argument('--dataset-root', type=str, default=default_dataset,
                        help='Root directory of the guidewire dataset.')
    parser.add_argument('--output-dir', type=str, default=default_output,
                        help='Directory where the generated images will be stored.')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='How many random train samples to export.')
    parser.add_argument('--seed', type=int, default=13,
                        help='Random seed for sampling/augmentation reproducibility.')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    return config


def load_rgb_image(image_path: str) -> np.ndarray:
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return image_rgb


def save_rgb_image(image_rgb: np.ndarray, path: Path):
    image_uint8 = (np.clip(image_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)


def save_grayscale_image(image_gray: np.ndarray, path: Path):
    gray_uint8 = (np.clip(image_gray, 0.0, 1.0) * 255.0).astype(np.uint8)
    cv2.imwrite(str(path), gray_uint8)


def save_heatmap(heatmap: np.ndarray, path: Path):
    normalized = heatmap
    max_val = float(np.max(heatmap))
    if max_val > 0:
        normalized = heatmap / max_val
    heatmap_uint8 = (np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(str(path), colored)


def choose_indices(total: int, count: int, rng: np.random.Generator) -> np.ndarray:
    if total <= 0:
        raise RuntimeError("Dataset does not contain any samples.")
    count = max(1, min(total, count))
    if count == total:
        return np.arange(total, dtype=np.int64)
    return rng.choice(total, size=count, replace=False)


def main():
    args = parse_args()
    config = load_config(args.config)

    dataset_root = os.path.abspath(args.dataset_root)
    if not os.path.isdir(dataset_root):
        raise NotADirectoryError(f"Dataset directory not found: {dataset_root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility: affect Python's random, NumPy, and OpenCV augmentations.
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    preprocessor = GuidewireDataPreprocessor(
        dir_dataset=dataset_root,
        split_ratio=config['dataset']['split_ratio'],
    )
    train_samples = preprocessor.get_data_sample_names('train')
    train_dataset = GuidewireDataSet(
        train_samples,
        apply_augmentation=True,
        config=config,
    )

    sample_indices = choose_indices(len(train_samples), args.num_samples, rng)
    print(f"Exporting {len(sample_indices)} samples to {output_dir}")

    for export_rank, dataset_index in enumerate(sample_indices, start=1):
        image_path, label_path = train_samples[dataset_index]
        base_name = Path(image_path).stem
        prefix = f"{export_rank:02d}_{dataset_index:05d}_{base_name}"

        # Load and save the raw RGB image
        raw_rgb = load_rgb_image(image_path)
        raw_rgb_path = output_dir / f"{prefix}_raw_rgb.png"
        save_rgb_image(raw_rgb, raw_rgb_path)

        # Fetch augmented tensors used for training
        aug_image, aug_heatmap = train_dataset.__getitem__(dataset_index)
        aug_image = train_dataset.destandardize_image(aug_image.copy())
        aug_image = np.squeeze(aug_image, axis=-1)

        augmented_path = output_dir / f"{prefix}_augmented_input.png"
        save_grayscale_image(aug_image, augmented_path)

        heatmap_path = output_dir / f"{prefix}_augmented_heatmap.png"
        save_heatmap(aug_heatmap, heatmap_path)

        print(f"[{export_rank}/{len(sample_indices)}] index={dataset_index} -> "
              f"{raw_rgb_path.name}, {augmented_path.name}, {heatmap_path.name}")


if __name__ == "__main__":
    main()

