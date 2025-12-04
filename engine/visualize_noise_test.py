#!/usr/bin/env python3
"""
Utility to visualize how guidewire tip detections behave under incremental
Gaussian noise injections.

For a given training config, this script:
1. Loads the corresponding checkpoint from results/<config_name>.
2. Samples N deterministic items from the test split.
3. Runs inference for the clean input and for inputs with 1×, 2×, 3× Gaussian noise
   (sigma taken from the config).
4. Saves the raw input plus overlays for each noise level.
"""

import argparse
import os
import random
import sys
from contextlib import nullcontext

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_loader.guidewire_data_loader import GuidewireDataPreprocessor, GuidewireDataSet
from nets import nn
from augmentation import pixel_coords as aug_functions


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize guidewire detection predictions.")
    parser.add_argument('--config', type=str, default='coords_amplifier_100',
                        help='Name or path of the results directory inside project_root/results.')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of test samples to visualize.')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Seed that controls the deterministic random sampler.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to store the exported figures. Defaults to results/<config_name>/detection_noise_test.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint file to load. Defaults to results/<config_name>/best.pt.')
    parser.add_argument('--marker_radius', type=float, default=12.0,
                        help='Radius (in pixels) for the visualization circle.')
    parser.add_argument('--dpi', type=int, default=200,
                        help='Resolution used when saving matplotlib figures.')
    return parser.parse_args()


def resolve_results_dir(config_arg: str) -> str:
    """
    Return absolute path to the results directory containing config.yaml and checkpoint files.
    """
    candidate_dirs = []
    if os.path.isabs(config_arg):
        candidate_dirs.append(config_arg)
    else:
        candidate_dirs.append(os.path.join(project_root, 'results', config_arg))
    for path in candidate_dirs:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f'Could not locate results directory for "{config_arg}". Tried: {candidate_dirs}')


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def resolve_checkpoint_path(checkpoint_arg: str, results_dir: str) -> str:
    if checkpoint_arg:
        cp = checkpoint_arg if os.path.isabs(checkpoint_arg) else os.path.join(results_dir, checkpoint_arg)
    else:
        cp = os.path.join(results_dir, 'best.pt')
    if not os.path.isfile(cp):
        raise FileNotFoundError(f'Checkpoint not found: {cp}')
    return cp


def build_model(config: dict, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    name_backbone = config['backbone']
    # Use the checkpoint itself to bootstrap backbone weights when dedicated backbone weights
    # are not available separately.
    pretrained_weights_path = checkpoint_path
    if not os.path.isfile(pretrained_weights_path):
        pretrained_weights_path = os.path.join(project_root, 'weights', f'{name_backbone}.pt')
    model = nn.YOLOwithCustomHead(
        name_backbone,
        pretrained_weights_path,
        tuple(config['network']['input_image_shape']),
        config['network']['head'],
        from_logits=config['from_logits']
    )
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def select_sample_indices(total: int, num_samples: int, seed: int) -> list:
    rng = random.Random(seed)
    num = min(num_samples, total)
    return sorted(rng.sample(range(total), num))


def tensor_from_image(image_np: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor.to(device=device, dtype=torch.float32)


def extract_tip_coordinates(prediction: torch.Tensor) -> np.ndarray:
    """
    Convert model output to normalized (x, y) coordinates in [0, 1].
    Handles both coordinate regression (B, 2) and heatmap (B, H, W) outputs.
    """
    pred = prediction.detach().cpu()
    if pred.ndim == 2 and pred.shape[1] == 2:
        coords = pred[0].numpy()
    else:
        heatmap = pred[0]
        flat_idx = torch.argmax(heatmap.view(-1)).item()
        height, width = heatmap.shape
        y = flat_idx // width
        x = flat_idx % width
        coords = np.array([x / max(1, width - 1), y / max(1, height - 1)], dtype=np.float32)
    return np.clip(coords, 0.0, 1.0)


def normalized_to_pixels(coords: np.ndarray, image_shape: tuple) -> tuple:
    height, width = image_shape
    x = coords[0] * (width - 1)
    y = coords[1] * (height - 1)
    return float(x), float(y)


def save_input_image(image_display: np.ndarray, save_path: str):
    plt.imsave(save_path, image_display, cmap='gray', vmin=0.0, vmax=1.0)


def save_overlay(image_display: np.ndarray,
                 save_path: str,
                 circles: list,
                 radius: float,
                 dpi: int):
    fig, ax = plt.subplots(figsize=(image_display.shape[1] / dpi, image_display.shape[0] / dpi), dpi=dpi)
    ax.imshow(image_display, cmap='gray', vmin=0.0, vmax=1.0)
    for circle_cfg in circles:
        center = circle_cfg.get('center')
        color = circle_cfg.get('color', 'red')
        linewidth = circle_cfg.get('linewidth', 1.0)
        circle_radius = circle_cfg.get('radius', radius)
        circle = plt.Circle(center, radius=circle_radius, color=color, fill=False, linewidth=linewidth)
        ax.add_patch(circle)
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def autocast_context(device: torch.device):
    """
    Return an autocast context manager compatible with both legacy torch.cuda.amp.autocast
    and newer torch.amp.autocast APIs. Falls back to a no-op on CPU.
    """
    if device.type != 'cuda':
        return nullcontext()
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast('cuda')
    return torch.cuda.amp.autocast()


def apply_repeated_gaussian_noise(image_std: np.ndarray,
                                  dataset: GuidewireDataSet,
                                  sigma: float,
                                  repeats: int) -> np.ndarray:
    """
    Destandardize image, apply Gaussian noise `repeats` times (sigma from config), then re-standardize.
    """
    if repeats <= 0 or sigma <= 0:
        return image_std.copy()

    image = dataset.destandardize_image(image_std.copy())
    dummy_coords = [0.0, 0.0]
    for _ in range(repeats):
        image, _ = aug_functions.augment_gaussian_noise(image, dummy_coords, sigma)
    image = np.clip(image, 0.0, 1.0)
    noisy_std = dataset.standardize_image(image)
    return noisy_std.astype(np.float32)


def main():
    args = parse_args()

    results_dir = resolve_results_dir(args.config)
    config_path = os.path.join(results_dir, 'config.yaml')
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f'Config file not found at {config_path}')
    config = load_config(config_path)
    config_name = os.path.basename(os.path.normpath(results_dir))
    save_dir = args.save_dir or os.path.join(results_dir, 'detection_noise_test')
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = resolve_checkpoint_path(args.checkpoint, results_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config, checkpoint_path, device)

    dataset_path = os.path.join(project_root, 'datasets', 'guidewire')
    data_preprocessor = GuidewireDataPreprocessor(dataset_path, config['dataset']['split_ratio'])
    test_samples = data_preprocessor.get_data_sample_names('test')
    if not test_samples:
        raise RuntimeError('No test samples available.')
    dataset = GuidewireDataSet(test_samples, apply_augmentation=False, config=config)

    sample_indices = select_sample_indices(len(dataset), args.num_samples, args.seed)
    print(f"Selected sample indices (deterministic): {sample_indices}")

    sigma_noise = float(config['dataset'].get('sigma_xray_noise', 0.0))
    noise_variants = [
        ('prediction_clean', 0),
        ('prediction_noise2', 2),
        ('prediction_noise4', 4),
    ]
    pred_radius_ratio = 0.9

    for order, sample_idx in enumerate(sample_indices, start=1):
        image_np, gt_coords = dataset.__getitem__(sample_idx, get_heatmap=False, apply_default_noise=False)

        height, width = image_np.shape[0], image_np.shape[1]
        gt_pixels = normalized_to_pixels(gt_coords, (height, width))

        image_path = dataset.data_sample_names[sample_idx][0]
        sample_name = os.path.splitext(os.path.basename(image_path))[0]
        base_name = f"{order:02d}_{sample_name}"
        input_path = os.path.join(save_dir, f"{base_name}_input.png")

        base_display = dataset.destandardize_image(image_np.copy()).squeeze(-1)
        base_display = np.clip(base_display, 0.0, 1.0)
        save_input_image(base_display, input_path)

        print(f"[{order}/{len(sample_indices)}] {sample_name}")
        print(f"    Saved input     : {input_path}")

        for variant_name, repeat_count in noise_variants:
            if repeat_count == 0:
                variant_image = image_np.copy()
            else:
                variant_image = apply_repeated_gaussian_noise(image_np, dataset, sigma_noise, repeat_count)

            variant_display = dataset.destandardize_image(variant_image.copy()).squeeze(-1)
            variant_display = np.clip(variant_display, 0.0, 1.0)

            image_tensor = tensor_from_image(variant_image, device)
            with torch.no_grad():
                with autocast_context(device):
                    prediction = model(image_tensor)
            pred_coords = extract_tip_coordinates(prediction)
            pred_pixels = normalized_to_pixels(pred_coords, (height, width))

            overlay_path = os.path.join(save_dir, f"{base_name}_{variant_name}.png")
            pred_radius = args.marker_radius * pred_radius_ratio
            save_overlay(
                variant_display,
                overlay_path,
                circles=[
                    {'center': gt_pixels, 'color': 'lime', 'linewidth': 1.0, 'radius': args.marker_radius},
                    {'center': pred_pixels, 'color': 'red', 'linewidth': 1.0, 'radius': pred_radius},
                ],
                radius=args.marker_radius,
                dpi=args.dpi,
            )

            print(f"    Saved {variant_name}: {overlay_path}")
            print(f"    GT (norm)  : ({gt_coords[0]:.4f}, {gt_coords[1]:.4f})")
            print(f"    Pred (norm): ({pred_coords[0]:.4f}, {pred_coords[1]:.4f})")

    print("\nVisualization complete!")
    print(f"Artifacts saved under: {save_dir}")


if __name__ == "__main__":
    main()
