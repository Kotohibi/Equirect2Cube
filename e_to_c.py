#!/usr/bin/env python3
"""
equirect_to_cubemap.py

OpenCV-based converter: Equirectangular panorama -> cubemap faces
Supports specifying overlap angle (degree) for SfM-friendly overlapping cube faces.
Processes all images in input directory with parallel processing.

Usage:
    python equirect_to_cubemap.py --input-dir inputs --out-dir cubemaps --face-size 1024 --cross --overlap 10

Dependencies:
    pip install opencv-python numpy

"""

import os
import argparse
import math
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor

FACE_NAMES = [
    ("posx", "+X"),
    ("negx", "-X"),
    ("posy", "+Y"),
    ("negy", "-Y"),
    ("posz", "+Z"),
    ("negz", "-Z"),
]

def generate_face_map(face: str, size: int, in_w: int, in_h: int, overlap_deg: float = 0.0):
    # 拡大視野角 = 90 + オーバーラップ角度
    fov_deg = 90.0 + overlap_deg
    half_fov_rad = math.radians(fov_deg / 2.0)
    tan_half_fov = math.tan(half_fov_rad)

    u = (np.arange(size) + 0.5) / size * 2.0 - 1.0  # [-1,1]
    v = (np.arange(size) + 0.5) / size * 2.0 - 1.0  # [-1,1]
    xv, yv = np.meshgrid(u, v)

    # スケール調整して視線方向を計算
    xv = xv * tan_half_fov
    yv = yv * tan_half_fov

    if face == 'posx':
        X = np.ones_like(xv)
        Y = -yv
        Z = -xv
    elif face == 'negx':
        X = -np.ones_like(xv)
        Y = -yv
        Z = xv
    elif face == 'posy':
        X = xv
        Y = np.ones_like(xv)
        Z = yv
    elif face == 'negy':
        X = xv
        Y = -np.ones_like(xv)
        Z = -yv
    elif face == 'posz':
        X = xv
        Y = -yv
        Z = np.ones_like(xv)
    elif face == 'negz':
        X = -xv
        Y = -yv
        Z = -np.ones_like(xv)
    else:
        raise ValueError('Unknown face: ' + face)

    norm = np.sqrt(X*X + Y*Y + Z*Z)
    Xn = X / norm
    Yn = Y / norm
    Zn = Z / norm

    theta = np.arctan2(Zn, Xn)
    phi = np.arcsin(Yn)

    map_x = (theta + math.pi) / (2 * math.pi) * (in_w - 1)
    map_y = (math.pi / 2 - phi) / math.pi * (in_h - 1)

    return map_x.astype(np.float32), map_y.astype(np.float32)

def face_from_equirect(in_img, face: str, size: int, overlap_deg: float):
    h, w = in_img.shape[:2]
    map_x, map_y = generate_face_map(face, size, w, h, overlap_deg)
    face_img = cv2.remap(in_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return face_img

def make_vertical_cross(faces_dict, face_size):
    W = face_size * 3
    H = face_size * 4
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    canvas[0:face_size, face_size:face_size * 2] = faces_dict['posy']
    canvas[face_size:face_size * 2, 0:face_size] = faces_dict['negx']
    canvas[face_size:face_size * 2, face_size:face_size * 2] = faces_dict['posz']
    canvas[face_size:face_size * 2, face_size * 2:face_size * 3] = faces_dict['posx']
    canvas[face_size * 2:face_size * 3, face_size:face_size * 2] = faces_dict['negy']
    canvas[face_size * 3:face_size * 4, face_size:face_size * 2] = faces_dict['negz']

    return canvas

def process_image(args):
    input_path, out_dir, face_size, out_format, do_cross, overlap_deg = args
    img_name = os.path.splitext(os.path.basename(input_path))[0]
    print(f'Processing {img_name} with overlap {overlap_deg} deg')

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f'Failed to load image: {input_path}')
        return

    if face_size is None:
        fs = img.shape[1] // 4
        print(f'--face-size 未指定のため元画像幅に基づき自動設定: {fs}')
    else:
        fs = face_size

    faces = {}
    for shortname, _ in FACE_NAMES:
        faces[shortname] = face_from_equirect(img, shortname, fs, overlap_deg)
        out_path = os.path.join(out_dir, f"{img_name}_{shortname}.{out_format}")
        cv2.imwrite(out_path, faces[shortname])
        print(f'Saved {out_path}')

    if do_cross:
        cross_img = make_vertical_cross(faces, fs)
        out_cross = os.path.join(out_dir, f"{img_name}_vertical_cross.{out_format}")
        cv2.imwrite(out_cross, cross_img)
        print(f'Saved {out_cross}')

def main():
    parser = argparse.ArgumentParser(description='Convert equirectangular panoramas to cubemap faces with overlap for SfM.')
    parser.add_argument('--input-dir', required=True, help='Input directory containing equirectangular images')
    parser.add_argument('--face-size', type=int, default=None, help='Output cube face size (pixels). If not set, auto set from input width / 6')
    parser.add_argument('--out-dir', default='cubemap_out', help='Output directory')
    parser.add_argument('--cross', action='store_true', help='Also create a vertical-cross stitched image')
    parser.add_argument('--format', default='png', choices=['png', 'jpg', 'jpeg'], help='Output image format')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel worker processes. Defaults to CPU core count.')
    parser.add_argument('--overlap', type=float, default=0.0, help='Overlap angle in degrees added to 90° FOV per face (e.g., 10 means 100° FOV)')
    args = parser.parse_args()

    exts = ['.jpg', '.jpeg', '.png']

    files = [f for f in os.listdir(args.input_dir) if os.path.splitext(f.lower())[1] in exts]
    if not files:
        print('No images found in input directory.')
        return

    os.makedirs(args.out_dir, exist_ok=True)

    task_args = [
        (os.path.join(args.input_dir, f), args.out_dir, args.face_size, args.format, args.cross, args.overlap)
        for f in files
    ]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(process_image, task_args))

    print('All done.')

if __name__ == '__main__':
    main()
