"""
Tensorflow keras implementation of U-Net
https://github.com/Bashirkazimi/BashirLearning
Author: Bashir Kazimi
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


def down_conv(x, filter_size,padding):
    """
    down convolution
    Args:
        x (): input tensor
        filter_size (int): filter size
        padding (str): padding

    Returns: keras tensor

    """

    for i in range(2):
        x = layers.Conv2D(
            filter_size,
            3,
            1,
            activation='relu',
            padding=padding
        )(x)

    maxpooling = layers.MaxPooling2D(
        2,
        2
    )(x)

    return maxpooling, x


def up_conv(x, skip, filter_size, similar=False):
    """
    up convolution
    Args:
        x (tensor): input tensor
        skip (tensor): residual connection
        filter_size (int): filter size
        similar (bool): similar size restoration?

    Returns: keras tensor

    """

    x = layers.Conv2DTranspose(
        filter_size,
        2,
        2,
        padding='same'
    )(x)

    image_size = tf.keras.backend.int_shape(x)[1]
    skip_size = tf.keras.backend.int_shape(skip)[1]

    crop_size = (skip_size - image_size) // 2
    cropped_tuple = (crop_size, crop_size)

    # make proper cropping to skip connections or zero padding to x's based
    # on whether similar or different input-ouput sizes are expected!
    if (skip_size - image_size) % 2:
        crop_size = crop_size
        cropped_tuple = ((crop_size, crop_size+1), (crop_size+1, crop_size))

    if not similar:  # just like original unet paper
        skip = layers.Cropping2D(
            cropped_tuple
        )(skip)
        padding='valid'
    else:  # zero padding to x
        x = layers.ZeroPadding2D(
            cropped_tuple
        )(x)
        padding='same'

    x = layers.Concatenate()([skip, x])

    for i in range(2):
        x = layers.Conv2D(
            filter_size,
            3,
            1,
            padding=padding,
            activation='relu'
        )(x)

    return x


def unet(input_shape=(572,572,1), num_classes=2, similar_output_size=False):
    """
    U-Net implementation based on https://arxiv.org/pdf/1505.04597.pdf
    Args:
        input_shape (tensor): input shape
        num_classes (int): number of categories
        similar_output_size (bool): similar or different output size

    Returns: U-Net tf.keras.Model

    """

    input = layers.Input(shape=input_shape)

    filter_size = 64
    x = input
    skips = []
    padding = 'same' if similar_output_size else 'valid'

    for i in range(4):
        x, skip = down_conv(x, filter_size, padding)
        skips.append(skip)
        filter_size *= 2

    x = layers.Conv2D(
        filter_size,
        3,
        1,
        activation='relu'
    )(x)
    x = layers.Conv2D(
        filter_size,
        3,
        1,
    )(x)

    skips.reverse()

    for i in range(4):
        filter_size = filter_size // 2
        x = up_conv(x, skips[i], filter_size, similar_output_size)

    x = layers.Conv2D(
        num_classes,
        1,
        1,
        activation='softmax',
        padding='same'
    )(x)

    model = Model(inputs=input, outputs=x)
    model.summary()

    return model