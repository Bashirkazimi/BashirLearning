"""
Tensorflow keras implementation of mobile net v2
https://github.com/Bashirkazimi/BashirLearning
Author: Bashir Kazimi
"""


import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


def conv_block(input_tensor, c, s, t, expand=True):
    """
    Convolutional Block for mobile net v2
    Args:
        input_tensor (keras tensor): input tensor
        c (int): output channels
        s (int): stride size of first layer in the series
        t (int): expansion factor
        expand (bool): expand filters or not?

    Returns: keras tensor
    """
    first_conv_channels = input_tensor.get_shape()[-1]
    if expand:
        x = layers.Conv2D(
            first_conv_channels*t,
            1,
            1,
            padding='same',
            use_bias=False
        )(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)
    else:
        x = input_tensor

    x = layers.DepthwiseConv2D(
        3,
        s,
        'same',
        1,
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)

    x = layers.Conv2D(
        c,
        1,
        1,
        padding='same',
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)

    if input_tensor.get_shape() == x.get_shape() and s == 1:
        return x+input_tensor

    return x


def mobile_net_v2(input_shape=(224,224,3), num_classes=1000):
    """
    mobile net v2 based on https://arxiv.org/pdf/1801.04381.pdf
    Args:
        input_shape (tuple): input shape
        num_classes (int): number of categories

    Returns: mobile net v2 model
    """
    input = layers.Input(shape=input_shape)

    x = layers.Conv2D(
        32,
        3,
        2,
        padding='same',
        use_bias=False
    )(input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)

    x = conv_block(x, 16, 1, 1, expand=False)
    x = conv_block(x, 24, 2, 6)
    x = conv_block(x, 24, 1, 6)

    x = conv_block(x, 32, 2, 6)
    x = conv_block(x, 32, 1, 6)
    x = conv_block(x, 32, 1, 6)

    x = conv_block(x, 64, 2, 6)
    x = conv_block(x, 64, 1, 6)
    x = conv_block(x, 64, 1, 6)
    x = conv_block(x, 64, 1, 6)

    x = conv_block(x, 96, 1, 6)
    x = conv_block(x, 96, 1, 6)
    x = conv_block(x, 96, 1, 6)

    x = conv_block(x, 160, 2, 6)
    x = conv_block(x, 160, 1, 6)
    x = conv_block(x, 160, 1, 6)

    x = conv_block(x, 320, 1, 6)

    x = layers.Conv2D(
        1280,
        1,
        1,
        padding='same',
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(num_classes)(x)
    x = layers.Activation('softmax')(x)

    model = Model(inputs=input, outputs=x)
    model.summary()

    return model

