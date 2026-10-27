"""Super-resolve a license plate image with the trained SRGAN generator and read it with OCR.

Usage:
    python pipeline.py path/to/plate.jpg            # prints the plate text, saves result.png
    python pipeline.py path/to/plate.jpg -o out.png
"""

import argparse
import functools
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from matplotlib.backends.backend_agg import FigureCanvasAgg

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "generator.h5")


@functools.lru_cache(maxsize=None)
def load_generator(model_path=DEFAULT_MODEL):
    from keras.models import load_model

    return load_model(model_path, compile=False)


# ------------------------------------------------------------- OCR helpers
def info_detection(image):
    """Run Tesseract on the cleaned binary plate image."""
    img = cv2.resize(image, (252, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return pytesseract.image_to_string(img, lang="eng", config="--psm 6")


def find_largest_objects(binary_image, num_objects=7):
    """Keep only the largest connected components (the plate characters), dropping noise."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    largest_indices = np.argsort(areas)[::-1][1 : num_objects + 1]  # skip the background
    result_image = np.zeros_like(binary_image)
    for index in largest_indices:
        result_image[labels == index] = 255
    return result_image


def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = np.uint8(cv2.GaussianBlur(gray, (5, 5), 0).astype(int))
    return cv2.Canny(blurred, 80, 150)


def prepare_for_ocr(image_255_hr):
    """Turn the SRGAN output into a clean binary image Tesseract can read."""
    edge = detect_edges(image_255_hr)

    test = np.uint8(cv2.cvtColor(image_255_hr, cv2.COLOR_RGB2GRAY) + edge)
    _, test = cv2.threshold(test, 80, 255, cv2.THRESH_BINARY_INV)
    test = np.uint8(cv2.GaussianBlur(test, (3, 3), 0).astype(int))
    _, test = cv2.threshold(test, 80, 255, cv2.THRESH_BINARY)
    test = test + edge
    test = np.uint8(cv2.GaussianBlur(test, (3, 3), 0).astype(int))
    _, test = cv2.threshold(test, 80, 255, cv2.THRESH_BINARY)
    return find_largest_objects(test, num_objects=8)


# ------------------------------------------------------------- full pipeline
def run(path, model_path=DEFAULT_MODEL):
    """Returns (plate_text, original_128, low_res_32, super_res_128)."""
    generator = load_generator(model_path)

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # The input is downscaled to 32x32 to simulate an extremely low-resolution capture.
    hr = cv2.cvtColor(cv2.resize(gray_img, (128, 128)), cv2.COLOR_BGR2RGB) / 255.0
    lr = cv2.cvtColor(cv2.resize(gray_img, (32, 32)), cv2.COLOR_BGR2RGB) / 255.0

    generated_hr = generator.predict(np.expand_dims(lr, axis=0), verbose=0)

    image_255_hr = np.float32(np.clip(generated_hr[0] * 255, 0, 255).astype(int))
    image_255_lr = np.float32(np.clip(lr * 255, 0, 255).astype(int))

    text = info_detection(prepare_for_ocr(image_255_hr)).strip()
    return text, hr, image_255_lr, image_255_hr


def render_result(text, hr, lr, sr):
    """Side-by-side figure (original / low-res / super-res + detected text) as an RGB array."""
    px = 1 / plt.rcParams["figure.dpi"]
    fig = plt.figure(layout="constrained", figsize=(720 * px, 960 * px))
    ax = fig.subplot_mosaic([["left", "center", "right"]])

    ax["left"].imshow(hr)
    ax["left"].set_title("Orig. HR image")
    ax["center"].imshow(lr.astype(int))
    ax["center"].set_title("LR Image")
    ax["right"].imshow(sr.astype(int))
    ax["right"].set_title("Super resolution")

    fig.suptitle("\n \nresult: " + text, fontsize="40", color="blue")

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    im = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return im


def project(path, model_path=DEFAULT_MODEL):
    """Full pipeline, returns the rendered result image (used by the GUI)."""
    return render_result(*run(path, model_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("-o", "--output", default="result.png")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    text, hr, lr, sr = run(args.image, args.model)
    print("Detected plate:", text)
    cv2.imwrite(args.output, cv2.cvtColor(render_result(text, hr, lr, sr), cv2.COLOR_RGB2BGR))
    print("Saved", args.output)


if __name__ == "__main__":
    main()
