import argparse
import os

import numpy as np
from PIL import Image


EPS = 1e-7


def load_lap_depth_png(depth_path, min_depth=0.9, max_depth=1.1):
    depth_png = np.asarray(Image.open(depth_path))
    if depth_png.ndim == 3:
        depth_png = depth_png[..., 0]

    if depth_png.dtype == np.uint16:
        depth_01 = depth_png.astype(np.float32) / 65535.0
    else:
        depth_01 = depth_png.astype(np.float32) / 255.0

    return depth_01 * (max_depth - min_depth) + min_depth


def depth_to_3d_grid(depth, fov=10.0):
    h, w = depth.shape
    fx = (w - 1) / 2 / np.tan(fov / 2 * np.pi / 180)
    fy = (h - 1) / 2 / np.tan(fov / 2 * np.pi / 180)
    cx = (w - 1) / 2
    cy = (h - 1) / 2

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    z = depth
    x3 = (x.astype(np.float32) - cx) / fx * z
    y3 = (y.astype(np.float32) - cy) / fy * z
    return np.stack([x3, y3, z], axis=-1)


def depth_to_normal(depth, fov=10.0):
    grid_3d = depth_to_3d_grid(depth, fov=fov)
    h, w = depth.shape

    tu = grid_3d[1:-1, 2:] - grid_3d[1:-1, :-2]
    tv = grid_3d[2:, 1:-1] - grid_3d[:-2, 1:-1]
    inner_normal = np.cross(tu, tv)

    normal = np.zeros((h, w, 3), dtype=np.float32)
    normal[..., 2] = 1.0
    normal[1:-1, 1:-1] = inner_normal
    normal /= np.sqrt(np.sum(normal ** 2, axis=-1, keepdims=True)) + EPS
    return normal


def save_lap_normal_png(normal, output_path):
    normal_rgb = (normal / 2.0 + 0.5) * 255.0
    normal_rgb = np.rint(normal_rgb).clip(0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    Image.fromarray(normal_rgb, mode='RGB').save(output_path)


def convert_depth_png_to_normal_png(depth_path, output_path, min_depth=0.9, max_depth=1.1, fov=10.0):
    depth = load_lap_depth_png(depth_path, min_depth=min_depth, max_depth=max_depth)
    normal = depth_to_normal(depth, fov=fov)
    save_lap_normal_png(normal, output_path)
    return normal


def main():
    parser = argparse.ArgumentParser(description='Convert LAP canonical_depth_lap.png to canonical_normal_lap.png.')
    parser.add_argument('--depth', default='canonical_depth_lap.png', help='Path to canonical_depth_lap.png.')
    parser.add_argument('--output', default='canonical_normal_lap_from_depth.png', help='Output normal map path.')
    parser.add_argument('--min_depth', type=float, default=0.9)
    parser.add_argument('--max_depth', type=float, default=1.1)
    parser.add_argument('--fov', type=float, default=10.0)
    args = parser.parse_args()

    normal = convert_depth_png_to_normal_png(
        depth_path=args.depth,
        output_path=args.output,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        fov=args.fov,
    )
    print(f'Saved normal map: {args.output}')
    print(f'Normal shape: {normal.shape}, range: [{normal.min():.4f}, {normal.max():.4f}]')


if __name__ == '__main__':
    main()
