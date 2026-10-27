"""Train the SRGAN on paired LR/HR images produced by prepare_dataset.py.

Usage:
    python train.py --data data/train --epochs 10
"""

import argparse
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from srgan.model import build_vgg, create_comb, create_disc, create_gen


def load_images(folder, names):
    images = []
    for name in names:
        img = cv2.imread(os.path.join(folder, name))
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.array(images)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="folder containing hr/ and lr/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=60000)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    names = sorted(n for n in os.listdir(os.path.join(args.data, "lr")) if not n.startswith("."))
    names = names[: args.max_images]

    lr_images = load_images(os.path.join(args.data, "lr"), names) / 255.0
    hr_images = load_images(os.path.join(args.data, "hr"), names) / 255.0

    lr_train, lr_test, hr_train, hr_test = train_test_split(
        lr_images, hr_images, test_size=0.33, random_state=42
    )

    lr_ip = Input(shape=lr_train.shape[1:])
    hr_ip = Input(shape=hr_train.shape[1:])

    generator = create_gen(lr_ip, num_res_block=16)

    discriminator = create_disc(hr_ip)
    discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    vgg = build_vgg(hr_train.shape[1:])
    vgg.trainable = False

    # Two losses: adversarial (binary cross-entropy on the discriminator's output)
    # and content (MSE between VGG19 feature maps of generated and real HR images).
    gan_model = create_comb(generator, discriminator, vgg, lr_ip, hr_ip)
    gan_model.compile(loss=["binary_crossentropy", "mse"], loss_weights=[1e-3, 1], optimizer="adam")

    batch_size = 1
    fake_label = np.zeros((batch_size, 1))
    real_label = np.ones((batch_size, 1))

    os.makedirs(args.out, exist_ok=True)
    g_loss_list, d_loss_list = [], []

    for e in range(args.epochs):
        g_losses, d_losses = [], []

        for start in tqdm(range(0, len(hr_train), batch_size)):
            lr_imgs = lr_train[start : start + batch_size]
            hr_imgs = hr_train[start : start + batch_size]

            # 1. Train the discriminator on generated and real HR images.
            fake_imgs = generator.predict_on_batch(lr_imgs)
            discriminator.trainable = True
            d_loss_gen = discriminator.train_on_batch(fake_imgs, fake_label)
            d_loss_real = discriminator.train_on_batch(hr_imgs, real_label)
            d_losses.append(0.5 * np.add(d_loss_gen, d_loss_real))

            # 2. Train the generator through the combined model, discriminator frozen.
            discriminator.trainable = False
            image_features = vgg.predict(hr_imgs, verbose=0)
            g_loss, _, _ = gan_model.train_on_batch([lr_imgs, hr_imgs], [real_label, image_features])
            g_losses.append(g_loss)

        g_loss = np.mean(g_losses, axis=0)
        d_loss = np.mean(d_losses, axis=0)
        g_loss_list.append(g_loss)
        d_loss_list.append(d_loss)
        print("epoch:", e + 1, "g_loss:", g_loss, "d_loss:", d_loss)

        if (e + 1) % args.save_every == 0:
            generator.save(os.path.join(args.out, f"gen_e_{e + 1}.h5"))

    plt.plot(np.arange(args.epochs), g_loss_list, label="g_loss")
    plt.plot(np.arange(args.epochs), np.array(d_loss_list)[:, 0], label="d_loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("training")
    plt.savefig(os.path.join(args.out, "training_loss.png"))


if __name__ == "__main__":
    main()
