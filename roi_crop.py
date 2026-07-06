#!/usr/bin/env python3
"""ROI crop script

Scans a directory tree for images, performs a centered square crop
around the center (crop_ratio of shortest side), resizes to 224x224.

By default the processed images are written under `processed_images` and
mirror the input subfolder structure. Use `--inplace` to write an ROI
version next to each original image (same folder) using a filename
suffix (default: ``_roi``). The original images are never deleted.
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import Tuple

try:
    from PIL import Image
except Exception as e:  # pragma: no cover - user will install Pillow if missing
    raise


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


def is_image_file(name: str) -> bool:
    return name.lower().endswith(IMAGE_EXTS)


def center_square_crop(image_size: Tuple[int, int], crop_ratio: float) -> Tuple[int, int, int, int]:
    w, h = image_size
    side = int(min(w, h) * crop_ratio)
    left = (w - side) // 2
    top = (h - side) // 2
    return (left, top, left + side, top + side)


def process_file(src: str, dst: str, crop_ratio: float, size: int, force: bool) -> bool:
    if os.path.exists(dst) and not force:
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        with Image.open(src) as im:
            im = im.convert('RGB')
            bbox = center_square_crop(im.size, crop_ratio)
            cropped = im.crop(bbox)
            resized = cropped.resize((size, size), Image.LANCZOS)
            resized.save(dst)
        return True
    except Exception as exc:
        print(f"Failed processing {src}: {exc}", file=sys.stderr)
        return False


def main() -> None:
    p = argparse.ArgumentParser(description='ROI center-crop and resize images')
    p.add_argument('--root', '-r', default='.', help='dataset root to scan')
    p.add_argument('--output', '-o', default='processed_images', help='output directory')
    p.add_argument('--inplace', action='store_true', help='save ROI images next to originals with a suffix (do not overwrite originals)')
    p.add_argument('--suffix', default='_roi', help='suffix to append to filename when using --inplace')
    p.add_argument('--crop-ratio', type=float, default=0.7, help='fraction of shortest side to crop')
    p.add_argument('--size', type=int, default=224, help='output size (square)')
    p.add_argument('--force', action='store_true', help='overwrite existing outputs')
    p.add_argument('--dry-run', action='store_true', help='only list files to be processed')
    args = p.parse_args()

    root = os.path.abspath(args.root)
    outdir = os.path.abspath(args.output)

    total = 0
    processed = 0
    skipped = 0

    for dirpath, dirs, files in os.walk(root):
        # avoid descending into output directory or typical virtualenv/__pycache__
        base_out = os.path.basename(outdir)
        dirs[:] = [d for d in dirs if d != base_out and not d.startswith('.') and d != '__pycache__']
        for fname in files:
            if not is_image_file(fname):
                continue
            total += 1
            src = os.path.join(dirpath, fname)

            # If inplace mode, write next to source with a suffix and skip any
            # files that already look like ROI outputs (avoid reprocessing ROI files)
            if args.inplace:
                base, ext = os.path.splitext(fname)
                if base.endswith(args.suffix):
                    # already an ROI file
                    skipped += 1
                    continue
                dst = os.path.join(dirpath, base + args.suffix + ext)
            else:
                rel_dir = os.path.relpath(dirpath, root)
                if rel_dir == os.curdir:
                    rel_dir = ''
                dst_dir = os.path.join(outdir, rel_dir)
                dst = os.path.join(dst_dir, fname)

            if args.dry_run:
                print(f"Would process: {src} -> {dst}")
                continue

            ok = process_file(src, dst, args.crop_ratio, args.size, args.force)
            if ok:
                processed += 1
            else:
                skipped += 1

    print(f"Total images found: {total}")
    print(f"Processed: {processed}, Skipped: {skipped}")
    print(f"Output directory: {outdir}")


if __name__ == '__main__':
    main()
