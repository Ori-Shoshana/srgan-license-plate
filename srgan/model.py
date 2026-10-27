"""SRGAN architecture (Ledig et al., 2017) used for 4x super-resolution (32x32 -> 128x128)."""

from keras import Model
from keras.applications import VGG19
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    LeakyReLU,
    PReLU,
    UpSampling2D,
    add,
)


# ---------------------------------------------------------------- generator
def res_block(ip):
    res_model = Conv2D(64, (3, 3), padding="same")(ip)
    res_model = BatchNormalization(momentum=0.5)(res_model)
    res_model = PReLU(shared_axes=[1, 2])(res_model)

    res_model = Conv2D(64, (3, 3), padding="same")(res_model)
    res_model = BatchNormalization(momentum=0.5)(res_model)

    return add([ip, res_model])


def upscale_block(ip):
    up_model = Conv2D(256, (3, 3), padding="same")(ip)
    up_model = UpSampling2D(size=2)(up_model)
    up_model = PReLU(shared_axes=[1, 2])(up_model)

    return up_model


def create_gen(gen_ip, num_res_block=16):
    """Generator: residual blocks + skip connection + two 2x upscale blocks (4x total)."""
    layers = Conv2D(64, (9, 9), padding="same")(gen_ip)
    layers = PReLU(shared_axes=[1, 2])(layers)

    temp = layers

    for _ in range(num_res_block):
        layers = res_block(layers)

    layers = Conv2D(64, (3, 3), padding="same")(layers)
    layers = BatchNormalization(momentum=0.5)(layers)
    layers = add([layers, temp])

    layers = upscale_block(layers)
    layers = upscale_block(layers)

    op = Conv2D(3, (9, 9), padding="same")(layers)

    return Model(inputs=gen_ip, outputs=op)


# ------------------------------------------------------------ discriminator
def discriminator_block(ip, filters, strides=1, bn=True):
    disc_model = Conv2D(filters, (3, 3), strides=strides, padding="same")(ip)

    if bn:
        disc_model = BatchNormalization(momentum=0.8)(disc_model)

    disc_model = LeakyReLU(negative_slope=0.2)(disc_model)

    return disc_model


def create_disc(disc_ip):
    """Discriminator: 8 conv blocks (64 -> 512 filters) + dense head, real/fake output."""
    df = 64

    d1 = discriminator_block(disc_ip, df, bn=False)
    d2 = discriminator_block(d1, df, strides=2)
    d3 = discriminator_block(d2, df * 2)
    d4 = discriminator_block(d3, df * 2, strides=2)
    d5 = discriminator_block(d4, df * 4)
    d6 = discriminator_block(d5, df * 4, strides=2)
    d7 = discriminator_block(d6, df * 8)
    d8 = discriminator_block(d7, df * 8, strides=2)

    d8_5 = Flatten()(d8)
    d9 = Dense(df * 16)(d8_5)
    d10 = LeakyReLU(negative_slope=0.2)(d9)
    validity = Dense(1, activation="sigmoid")(d10)

    return Model(disc_ip, validity)


# ------------------------------------------------------------- content loss
def build_vgg(hr_shape):
    """Pre-trained VGG19 truncated at block 3 (last conv), used for the perceptual content loss."""
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=hr_shape)

    return Model(inputs=vgg.inputs, outputs=vgg.layers[10].output)


def create_comb(gen_model, disc_model, vgg, lr_ip, hr_ip):
    """Combined model for training the generator: adversarial output + VGG features."""
    gen_img = gen_model(lr_ip)

    gen_features = vgg(gen_img)

    disc_model.trainable = False
    validity = disc_model(gen_img)

    return Model(inputs=[lr_ip, hr_ip], outputs=[validity, gen_features])
