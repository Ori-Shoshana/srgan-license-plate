"""Build paired HR (128x128) / LR (32x32) grayscale training images from a folder of images.

Usage:
    python prepare_dataset.py --src data/original --out data/train
"""

import argparse
import os

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="folder with the original images")
    parser.add_argument("--out", required=True, help="output folder (hr/ and lr/ are created inside)")
    args = parser.parse_args()

    hr_dir = os.path.join(args.out, "hr")
    lr_dir = os.path.join(args.out, "lr")
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)

    count = 0
    for name in sorted(os.listdir(args.src)):
        if name.startswith("."):
            continue

        img_array = cv2.imread(os.path.join(args.src, name))
        if img_array is None:
            continue

        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        hr_img_array = cv2.resize(img_array, (128, 128))
        lr_img_array = cv2.resize(hr_img_array, (32, 32))

        cv2.imwrite(os.path.join(hr_dir, name), hr_img_array)
        cv2.imwrite(os.path.join(lr_dir, name), lr_img_array)
        count += 1

    print(f"Prepared {count} image pairs in {args.out}")


if __name__ == "__main__":
    main()
