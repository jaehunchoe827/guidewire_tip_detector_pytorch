#!/usr/bin/env python3
"""
Utility script used to reproduce the figure assets described in the project
spec. It loads one sample from the test split, creates several visual
representations (raw RGB, grayscale+noise, Gaussian heatmap, and a grayscale
noise image with the ground-truth marked), and saves them to disk.

No network checkpoints are loaded – we only read from the dataset.
"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import yaml

# Ensure project root is on sys.path for local imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_loader.guidewire_data_loader import GuidewireDataPreprocessor


def parse_args():
    default_config = os.path.join(PROJECT_ROOT, 'config', 'ver3.yaml')
    default_dataset = os.path.join(PROJECT_ROOT, 'datasets', 'guidewire')
    default_output = os.path.join(PROJECT_ROOT, 'results', 'figure1_assets')

    parser = ArgumentParser(description="Generate figure assets from a single test sample.")
    parser.add_argument('--config', type=str, default=default_config,
                        help='Path to the YAML config file.')
    parser.add_argument('--dataset-root', type=str, default=default_dataset,
                        help='Root directory of the guidewire dataset.')
    parser.add_argument('--sample-index', type=int, default=0,
                        help='Index of the test sample to visualize.')
    parser.add_argument('--output-dir', type=str, default=default_output,
                        help='Directory where the generated images will be saved.')
    parser.add_argument('--noise-sigma', type=float, default=None,
                        help='Standard deviation for Gaussian noise (overrides config).')
    parser.add_argument('--heatmap-sigma', type=float, default=None,
                        help='Sigma (in pixels) for the 2D Gaussian heatmap (overrides config).')
    parser.add_argument('--dot-radius', type=int, default=3,
                        help='Radius (in pixels) for the ground-truth dot overlay.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed used for Gaussian noise.')
    parser.add_argument('--target-pos', type=int, nargs=2, metavar=('X', 'Y'), default=[304, 252],
                        help='Desired (x, y) location of the ground-truth after cropping.')
    parser.add_argument('--crop-size', type=int, nargs=2, metavar=('H', 'W'), default=None,
                        help='Crop size (height width). Defaults to the network input shape.')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    config.setdefault('config_name', Path(config_path).stem)
    return config


def load_rgb_image(image_path: str) -> np.ndarray:
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return image_rgb


def load_normalized_label(label_path: str, width: int, height: int) -> np.ndarray:
    with open(label_path, 'r', encoding='utf-8') as handle:
        line = handle.readline().strip()
    if not line:
        raise ValueError(f"Label file is empty: {label_path}")
    parts = line.split()
    if len(parts) < 3:
        raise ValueError(f"Label file does not contain x and y: {label_path}")
    x_raw = float(parts[1])
    y_raw = float(parts[2])
    if width <= 1 or height <= 1:
        raise ValueError("Invalid image dimensions when decoding labels.")
    return np.array([x_raw / (width - 1), y_raw / (height - 1)], dtype=np.float32)


def normalized_to_pixel_coords(label: np.ndarray, width: int, height: int) -> Tuple[float, float]:
    x = float(label[0]) * (width - 1)
    y = float(label[1]) * (height - 1)
    return x, y


def convert_to_gray_with_noise(image_rgb: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    noise = rng.normal(loc=0.0, scale=sigma, size=gray.shape).astype(np.float32)
    noisy_gray = np.clip(gray + noise, 0.0, 1.0)
    return noisy_gray


def generate_heatmap(width: int, height: int, center: Tuple[float, float], sigma: float) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("heatmap sigma must be positive.")
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    cx, cy = center
    exponent = -0.5 * (((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / (sigma ** 2))
    heatmap = np.exp(exponent)
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    return heatmap


def save_rgb_image(image_rgb: np.ndarray, path: Path):
    image_bgr = cv2.cvtColor((image_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)


def save_grayscale_image(image_gray: np.ndarray, path: Path):
    gray_uint8 = (np.clip(image_gray, 0.0, 1.0) * 255.0).astype(np.uint8)
    cv2.imwrite(str(path), gray_uint8)


def save_heatmap_image(heatmap: np.ndarray, path: Path):
    heatmap_uint8 = (np.clip(heatmap, 0.0, 1.0) * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(str(path), colored)


def overlay_dot_on_gray(gray: np.ndarray, center: Tuple[float, float], radius: int) -> np.ndarray:
    color = cv2.cvtColor((np.clip(gray, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    h, w = color.shape[:2]
    cx = int(np.clip(round(center[0]), 0, w - 1))
    cy = int(np.clip(round(center[1]), 0, h - 1))
    radius = max(1, radius)
    cv2.circle(color, (cx, cy), radius, (0, 255, 0), thickness=-1)
    return color


def crop_image_to_target(image: np.ndarray,
                         center: Tuple[float, float],
                         target_size: Tuple[int, int],
                         target_pos: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """
    Crop the image so that the ground-truth pixel lands at target_pos inside the cropped patch.
    target_size is provided as (height, width). target_pos is (x, y).
    """
    target_h, target_w = target_size
    desired_x, desired_y = target_pos
    height, width = image.shape[:2]

    if target_h > height or target_w > width:
        raise ValueError(f"Requested crop {target_h}x{target_w} larger than image {height}x{width}.")

    if not (0 <= desired_x < target_w and 0 <= desired_y < target_h):
        raise ValueError("target_pos must lie within the crop dimensions.")

    x0 = int(round(center[0]) - desired_x)
    y0 = int(round(center[1]) - desired_y)
    x0 = int(np.clip(x0, 0, max(0, width - target_w)))
    y0 = int(np.clip(y0, 0, max(0, height - target_h)))

    cropped = image[y0:y0 + target_h, x0:x0 + target_w]
    new_center = (center[0] - x0, center[1] - y0)
    return cropped, new_center, (x0, y0)


def main():
    args = parse_args()
    config = load_config(args.config)

    dataset_root = os.path.abspath(args.dataset_root)
    if not os.path.isdir(dataset_root):
        raise NotADirectoryError(f"Dataset directory not found: {dataset_root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_dims = tuple(args.crop_size) if args.crop_size else tuple(config['network']['input_image_shape'])
    if len(crop_dims) != 2:
        raise ValueError("crop_size must contain exactly two values: height and width.")
    crop_height, crop_width = int(crop_dims[0]), int(crop_dims[1])
    target_pos = (int(args.target_pos[0]), int(args.target_pos[1]))

    split_ratio = config['dataset']['split_ratio']
    preprocessor = GuidewireDataPreprocessor(dir_dataset=dataset_root, split_ratio=split_ratio)
    test_samples = preprocessor.get_data_sample_names('test')
    if not test_samples:
        raise RuntimeError("No samples found in the test split.")

    sample_index = max(0, min(args.sample_index, len(test_samples) - 1))
    image_path, label_path = test_samples[sample_index]

    rgb_image = load_rgb_image(image_path)
    height, width = rgb_image.shape[:2]
    label_norm = load_normalized_label(label_path, width, height)
    center = normalized_to_pixel_coords(label_norm, width, height)

    rgb_image, center, crop_origin = crop_image_to_target(
        rgb_image,
        center,
        target_size=(crop_height, crop_width),
        target_pos=target_pos,
    )
    height, width = rgb_image.shape[:2]

    noise_sigma = args.noise_sigma if args.noise_sigma is not None else float(config['dataset']['sigma_xray_noise'])
    heatmap_sigma = args.heatmap_sigma if args.heatmap_sigma is not None else float(config['dataset']['heatmap_sigma'])

    rng = np.random.default_rng(args.seed)
    noisy_gray = convert_to_gray_with_noise(rgb_image, noise_sigma, rng)
    heatmap = generate_heatmap(width, height, center, heatmap_sigma)
    overlay = overlay_dot_on_gray(noisy_gray, center, args.dot_radius)

    base_name = Path(image_path).stem
    raw_rgb_path = output_dir / f"{base_name}_raw_rgb.png"
    gray_noise_path = output_dir / f"{base_name}_gray_gaussian_noise.png"
    heatmap_path = output_dir / f"{base_name}_gaussian_heatmap.png"
    overlay_path = output_dir / f"{base_name}_gray_gaussian_noise_with_gt.png"

    save_rgb_image(rgb_image, raw_rgb_path)
    save_grayscale_image(noisy_gray, gray_noise_path)
    save_heatmap_image(heatmap, heatmap_path)
    cv2.imwrite(str(overlay_path), overlay)

    print(f"Processed sample index {sample_index}:")
    print(f"  Image path: {image_path}")
    print(f"  Label path: {label_path}")
    print(f"  Crop origin (x0, y0): {crop_origin}")
    print(f"  Crop size (H, W): {height} x {width}")
    print(f"  Desired GT position inside crop (x, y): {target_pos}")
    print(f"  Actual GT position after crop (x, y): ({center[0]:.2f}, {center[1]:.2f})")
    print(f"Saved outputs to {output_dir}:")
    print(f"  1) Raw RGB image -> {raw_rgb_path}")
    print(f"  2) Grayscale + Gaussian noise -> {gray_noise_path}")
    print(f"  3) Gaussian heatmap -> {heatmap_path}")
    print(f"  4) Grayscale noise + GT dot -> {overlay_path}")


if __name__ == "__main__":
    main()


# python3 /home/jaehun/workspace/gwtd/engine/project_figure1.py --config /home/jaehun/workspace/gwtd/config/ver3.yaml --sample-index 0