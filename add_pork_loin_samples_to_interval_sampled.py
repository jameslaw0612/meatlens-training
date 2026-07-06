#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path.cwd()
INTERVAL_SAMPLED_ROOT = PROJECT_ROOT / "interval sampled"

SOURCE_TO_DEST = {
    Path(r"E:\Pork Loin - sample 7"): {
        "dest_folder_name": "Pork Loin - Sample 7",
        "subdir_map": {
            "Fresh": "Fresh 0-7 hours",
            "Not Fresh": "Not Fresh 7-14 hours",
            "Spoiled": "all spoiled 14-40 hours",
        },
    },
    Path(r"E:\Pork Loin - sample 8"): {
        "dest_folder_name": "Pork Loin - Sample 8",
        "subdir_map": {
            "Fresh 0-7 hours": "Fresh 0-7 hours",
            "Not Fresh 7-14 hours": "Not Fresh 7-14 hours",
            "all spoiled 14-40 hours": "all spoiled 14-40 hours",
        },
    },
}


TARGET_COUNT_PER_CLASS = 200
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


def evenly_spaced_indices(total_count: int, target_count: int) -> list[int]:
    if total_count <= 0:
        return []
    if total_count <= target_count:
        return list(range(total_count))
    if target_count <= 1:
        return [total_count // 2]

    indices = []
    for idx in range(target_count):
        position = round(idx * (total_count - 1) / (target_count - 1))
        indices.append(int(position))
    return indices


def sorted_images(folder_path: Path) -> list[Path]:
    images = [path for path in folder_path.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS]
    return sorted(images, key=lambda path: (path.stat().st_mtime, path.name.lower()))


def copy_interval_sampled_subset(src_dir: Path, dst_dir: Path, target_count: int = TARGET_COUNT_PER_CLASS) -> tuple[int, int]:
    images = sorted_images(src_dir)
    selected_indices = evenly_spaced_indices(len(images), target_count)
    selected_images = [images[idx] for idx in selected_indices]

    copied_files = 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_file in selected_images:
        shutil.copy2(src_file, dst_dir / src_file.name)
        copied_files += 1
    return copied_files, len(images)


def main() -> None:
    INTERVAL_SAMPLED_ROOT.mkdir(parents=True, exist_ok=True)

    total_files = 0
    for source_root, config in SOURCE_TO_DEST.items():
        if not source_root.exists():
            raise FileNotFoundError(f"Missing source folder: {source_root}")

        dest_root = INTERVAL_SAMPLED_ROOT / str(config["dest_folder_name"])
        dest_root.mkdir(parents=True, exist_ok=True)

        subdir_map: dict[str, str] = dict(config["subdir_map"])
        for src_subdir_name, dest_subdir_name in subdir_map.items():
            src_subdir = source_root / src_subdir_name
            if not src_subdir.exists():
                raise FileNotFoundError(f"Missing source subfolder: {src_subdir}")

            dest_subdir = dest_root / dest_subdir_name
            copied, original_count = copy_interval_sampled_subset(src_subdir, dest_subdir)
            total_files += copied
            print(
                f"Copied {copied} sampled files from {original_count} originals: "
                f"{src_subdir} -> {dest_subdir}"
            )

    print(f"Total copied files: {total_files}")


if __name__ == "__main__":
    main()
