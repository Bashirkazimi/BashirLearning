"""
Tensorflow keras implementation of inception v2 v3
https://github.com/Bashirkazimi/BashirLearning
Author: Bashir Kazimi
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


def conv_batch_relu(x, filters, kernel_size, strides, padding, bn=False):
    """
    convolution batch normalization and relu trio
    Args:
        x (keras tensor): input tensor
        filters (int): filter size
        kernel_size (int): kernel size
        strides (int): stride size
        padding (str): padding
        bn (bool): bn applied or not?

    Returns: keras tensor

    """
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding
    )(x)

    if bn:
        x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    return x


def auxiliary_classifier(input_tensor, v=3, num_classes=1000):
    """
    Auxiliary classification branch, if one wishes!
    Args:
        input_tensor (keras tensor): input tensor
        num_classes (int): number of categories

    Returns: softmax

    """
    x = layers.AveragePooling2D(
        5,
        3,
    )(input_tensor)
    x = layers.Conv2D(
        128,
        1,
        1,
        padding='same'
    )(x)
    x = layers.Flatten()(x)
    x = layers.Dense(
        1024,
        activation='relu'
    )(x)

    x = layers.Dropout(
        rate=0.5
    )(x)
    x = layers.Dense(num_classes)(x)

    return x


def inception_v2_v3(input_shape=(299,299,3), num_classes=1000, v=2):
    """
    Inception v2 and v3 implementation based on
    https://arxiv.org/pdf/1512.00567v3.pdf
    Args:
        input_shape (tuple): input shape
        num_classes (int): number of categories
        v (int): v2 or v3?

    Returns: inception v2 or v3 model

    """
    bn = False if v == 2 else True
    inp = layers.Input(shape=input_shape)

    x = conv_batch_relu(inp, 32, 3, 2, 'valid', bn)
    x = conv_batch_relu(x, 64, 3, 1, 'valid', bn)

    x = layers.MaxPooling2D(
        3,
        2
    )(x)

    x = conv_batch_relu(x, 80, 3, 1, 'valid', bn)
    x = conv_batch_relu(x, 192, 3, 2, 'valid', bn)
    x = conv_batch_relu(x, 288, 3, 1, 'same', bn)

    # 3 x inception as in figure 5
    # first time!
    b1 = conv_batch_relu(x, 48, 1, 1, 'same', bn)
    b1 = conv_batch_relu(b1, 64, 3, 1, 'same', bn)
    b1 = conv_batch_relu(b1, 64, 3, 1, 'same', bn)

    b2 = conv_batch_relu(x, 64, 1, 1, 'same', bn)
    b2 = conv_batch_relu(b2, 96, 3, 1, 'same', bn)

    b3 = layers.AveragePooling2D(
        3,
        1,
        padding='same'
    )(x)
    b3 = conv_batch_relu(b3, 64, 1, 1, 'same', bn)

    b4 = conv_batch_relu(x, 64, 1, 1, 'same', bn)

    x = layers.Concatenate()([b1, b2, b3, b4])

    # 2nd time!
    b1 = conv_batch_relu(x, 72, 1, 1, 'same', bn)
    b1 = conv_batch_relu(b1, 96, 3, 1, 'same', bn)
    b1 = conv_batch_relu(b1, 96, 3, 1, 'same', bn)

    b2 = conv_batch_relu(x, 96, 1, 1, 'same', bn)
    b2 = conv_batch_relu(b2, 144, 3, 1, 'same', bn)

    b3 = layers.AveragePooling2D(
        3,
        1,
        padding='same'
    )(x)
    b3 = conv_batch_relu(b3, 96, 1, 1, 'same', bn)

    b4 = conv_batch_relu(x, 144, 1, 1, 'same', bn)

    x = layers.Concatenate()([b1, b2, b3, b4])

    # 3rd time!
    b1 = conv_batch_relu(x, 144, 1, 1, 'same', bn)
    b1 = conv_batch_relu(b1, 192, 3, 1, 'same', bn)
    b1 = conv_batch_relu(b1, 192, 3, 2, 'valid', bn)

    b2 = conv_batch_relu(x, 192, 1, 1, 'same', bn)
    b2 = conv_batch_relu(b2, 288, 3, 2, 'valid', bn)

    b3 = layers.MaxPooling2D(
        3,
        2,
        padding='valid'
    )(x)
    b3 = conv_batch_relu(b3, 144, 1, 1, 'same', bn)

    b4 = conv_batch_relu(x, 144, 3, 2, 'valid', bn)

    x = layers.Concatenate()([b1, b2, b3, b4])

    # 5 x inception as in figure 6
    # first two times
    for i in range(2):
        b1 = conv_batch_relu(x, 128, 1, 1, 'same', bn)
        b1 = conv_batch_relu(b1, 128, (1, 7), 1, 'same', bn)
        b1 = conv_batch_relu(b1, 128, (7, 1), 1, 'same', bn)
        b1 = conv_batch_relu(b1, 128, (1, 7), 1, 'same', bn)
        b1 = conv_batch_relu(b1, 128, (7, 1), 1, 'same', bn)

        b2 = conv_batch_relu(x, 128, 1, 1, 'same', bn)
        b2 = conv_batch_relu(b2, 128, (1, 7), 1, 'same', bn)
        b2 = conv_batch_relu(b2, 192, (7, 1), 1, 'same', bn)

        b3 = layers.AveragePooling2D(
            3,
            1,
            padding='same'
        )(x)
        b3 = conv_batch_relu(b3, 192, 1, 1, 'same', bn)

        b4 = conv_batch_relu(x, 256, 1, 1, 'same', bn)

        x = layers.Concatenate()([b1, b2, b3, b4])
    # second two times
    for i in range(2):
        b1 = conv_batch_relu(x, 192, 1, 1, 'same', bn)
        b1 = conv_batch_relu(b1, 192, (1, 7), 1, 'same', bn)
        b1 = conv_batch_relu(b1, 192, (7, 1), 1, 'same', bn)
        b1 = conv_batch_relu(b1, 192, (1, 7), 1, 'same', bn)
        b1 = conv_batch_relu(b1, 192, (7, 1), 1, 'same', bn)

        b2 = conv_batch_relu(x, 192, 1, 1, 'same', bn)
        b2 = conv_batch_relu(b2, 192, (1, 7), 1, 'same', bn)
        b2 = conv_batch_relu(b2, 256, (7, 1), 1, 'same', bn)

        b3 = layers.AveragePooling2D(
            3,
            1,
            padding='same'
        )(x)
        b3 = conv_batch_relu(b3, 256, 1, 1, 'same', bn)

        b4 = conv_batch_relu(x, 320, 1, 1, 'same', bn)

        x = layers.Concatenate()([b1, b2, b3, b4])
    # 5th time
    b1 = conv_batch_relu(x, 256, 3, 2, 'valid', bn)
    b1 = conv_batch_relu(b1, 256, (1, 7), 1, 'same', bn)
    b1 = conv_batch_relu(b1, 256, (7, 1), 1, 'same', bn)
    b1 = conv_batch_relu(b1, 256, (1, 7), 1, 'same', bn)
    b1 = conv_batch_relu(b1, 256, (7, 1), 1, 'same', bn)

    b2 = conv_batch_relu(x, 256, 3, 2, 'valid', bn)
    b2 = conv_batch_relu(b2, 256, (1, 7), 1, 'same', bn)
    b2 = conv_batch_relu(b2, 320, (7, 1), 1, 'same', bn)

    b3 = layers.MaxPooling2D(
        3,
        2,
    )(x)
    b3 = conv_batch_relu(b3, 320, 1, 1, 'same', bn)

    b4 = conv_batch_relu(x, 384, 3, 2, 'valid', bn)

    x = layers.Concatenate()([b1, b2, b3, b4])

    # 2 x inception as in figure 7
    for i in range(2):
        b1 = conv_batch_relu(x, 448, 1, 1, 'same', bn)
        b1 = conv_batch_relu(b1, 384, 3, 1, 'same', bn)
        b1_1 = conv_batch_relu(b1, 384, (3, 1), 1, 'same', bn)
        b1_2 = conv_batch_relu(b1, 384, (1, 3), 1, 'same', bn)

        b1 = layers.Concatenate()([b1_1, b1_2])

        b2 = conv_batch_relu(x, 384, 1, 1, 'same', bn)
        b2_1 = conv_batch_relu(b2, 384, (3, 1), 1, 'same', bn)
        b2_2 = conv_batch_relu(b2, 384, (1, 3), 1, 'same', bn)
        b2 = layers.Concatenate()([b2_1, b2_2])

        b3 = layers.AveragePooling2D(
            3,
            1,
            padding='same'
        )(x)
        b3 = conv_batch_relu(b3, 192, 1, 1, 'same', bn)

        b4 = conv_batch_relu(x, 320, 1, 1, 'same', bn)

        x = layers.Concatenate()([b1, b2, b3, b4])

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=x)
    model.summary()

    return model
