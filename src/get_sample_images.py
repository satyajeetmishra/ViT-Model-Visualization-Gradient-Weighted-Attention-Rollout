"""
One-time helper: download Imagenette (a small, standard 10-class subset of
ImageNet -- a mix of animals and objects, one clear subject per image:
tench, English springer spaniel, cassette player, chainsaw, church, French
horn, garbage truck, gas pump, golf ball, parachute) at 320px resolution
and dump ~100 sample images straight into images/, ready for main.py to
run on.

This uses the "320px" size variant (~300MB download, real JPEG photos --
high enough resolution that the heatmap overlay is clearly visible, unlike
CIFAR-100's native 32x32 images). The tradeoff versus something like
Caltech-256 is fewer distinct classes (10, not 100) -- this gives 10
images from each of those 10 classes instead.

This is a one-off dataset-preparation script, kept separate from the actual
visualization pipeline (model.py / rollout.py / visualize.py / main.py) so
it's obvious it's not part of the assignment's core deliverable -- just a
convenience for generating test images.

Usage:
    python src/get_sample_images.py
    python src/get_sample_images.py --per-class 5 --output images
"""

import argparse
import os

from torchvision.datasets import Imagenette

# Official Imagenette class order (index -> readable name), used only for
# naming the saved files.
CLASS_NAMES = [
    "tench",
    "english_springer",
    "cassette_player",
    "chain_saw",
    "church",
    "french_horn",
    "garbage_truck",
    "gas_pump",
    "golf_ball",
    "parachute",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Download ~100 Imagenette sample images")
    parser.add_argument("--data-root", default="data/imagenette", help="Where to cache the raw dataset")
    parser.add_argument("--output", default="images", help="Folder to save sample .jpg files into")
    parser.add_argument("--per-class", type=int, default=10, help="How many images per class (10 classes total)")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="Imagenette split to pull from")
    return parser.parse_args()


def download_dataset(data_root, split):
    """
    Downloads Imagenette at 320px resolution on first call (~300MB) and
    caches it under data_root for any future runs.
    """
    os.makedirs(data_root, exist_ok=True)
    return Imagenette(root=data_root, split=split, size="320px", download=True)


def save_samples(dataset, output_dir, per_class):
    """
    Walk the dataset once, and for each of the 10 classes save the first
    `per_class` images encountered as plain .jpg files named
    <class_name>_<index>.jpg into output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    counts = {label: 0 for label in range(len(CLASS_NAMES))}
    saved_paths = []

    for image, label in dataset:
        if counts[label] >= per_class:
            if all(count >= per_class for count in counts.values()):
                break
            continue

        class_name = CLASS_NAMES[label]
        file_name = f"{class_name}_{counts[label]}.jpg"
        file_path = os.path.join(output_dir, file_name)

        image.convert("RGB").save(file_path)
        saved_paths.append(file_path)
        counts[label] += 1

    return saved_paths


def main():
    args = parse_args()

    dataset = download_dataset(args.data_root, args.split)
    saved_paths = save_samples(dataset, args.output, args.per_class)

    print(f"Saved {len(saved_paths)} images to {args.output}/")
    print(f"({args.per_class} per class x {len(CLASS_NAMES)} classes)")


if __name__ == "__main__":
    main()
