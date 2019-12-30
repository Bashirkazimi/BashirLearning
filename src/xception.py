"""
Tensorflow keras implementation of xception
https://github.com/Bashirkazimi/BashirLearning
Author: Bashir Kazimi
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


def conv_batch_relu(input, filters, kernel_size, strides, padding='same', relu=True):
    """
    Convolution, BatchNormalization, ReLU trio
    Args:
        input (keras tensor): input tensor
        filters (int): filter size
        kernel_size (int): kernel size
        strides (int): stride size
        padding (str): padding
        relu (bool): True or False (apply ReLU or not)

    Returns: output of relu (keras tensor)

    """
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False
    )(input)
    x = layers.BatchNormalization()(x)
    if relu:
        x = layers.ReLU()(x)

    return x


def sep_conv(x, filters, kernel_size, strides, padding, relu=True):
    """
    relu (if True) + depthwise separable convolution + batch norm
    Args:
        x (keras tensor): input tensor
        filters (int): filter size
        kernel_size (int): kernel size
        strides (int): stride size
        padding (str): padding
        relu (bool): apply ReLU or not

    Returns: output of depthwise separable conv

    """
    if relu:
        x = layers.ReLU()(x)

    # separable conv can be done in two steps: DepthwiseConv2 followed by Point Wise Conv (1x1 Conv2D)
    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False
    )(x)

    x = layers.Conv2D(
        filters=filters,
        kernel_size=1,
        strides=1,
        padding=padding,
        use_bias=False
    )(x)

    # # Separable Convolution can be done in one SeparableConv2D call as well!!
    # x = layers.SeparableConv2D(
    #     filters,
    #     kernel_size,
    #     strides,
    #     padding='same',
    #     use_bias=False
    # )(x)

    x = layers.BatchNormalization()(x)

    return x


def middle_sep_conv_block(x, filters, kernel_size, strides, padding):
    """
    depthwise separable convolution block for xception middle flow
    Args:
        x (keras tensor): input tensor
        filters (list of int): filter sizes
        kernel_size (list of int): kernel sizes
        strides (list of int): stride sizes
        padding (list of str): padding

    Returns: keras tensor

    """
    for i in range(3):
        x = sep_conv(
            x,
            filters[i],
            kernel_size[i],
            strides[i],
            padding[i]
        )
    return x


def entry_flow(input):
    """
    entry flow block in xception
    Args:
        input (): keras input tensor

    Returns: keras tensor

    """
    x = conv_batch_relu(
        input,
        filters=32,
        kernel_size=3,
        strides=2,
        padding='valid'
    )
    x = conv_batch_relu(
        x,
        filters=64,
        kernel_size=3,
        strides=1,
        padding='valid'
    )

    res = conv_batch_relu(
        x,
        filters=128,
        kernel_size=1,
        strides=2,
        padding='valid',
        relu=False
    )

    x = sep_conv(x, 128, 3, 1, 'same', relu=False)
    x = sep_conv(x, 128, 3, 1, 'same')
    x = layers.MaxPooling2D(
        3,
        2,
        padding='same'
    )(x)

    x = x + res
    res = conv_batch_relu(
        x,
        filters=256,
        kernel_size=1,
        strides=2,
        padding='valid',
        relu=False
    )

    for i in range(2):
        x = sep_conv(x, 256, 3, 1, 'same')
    x = layers.MaxPooling2D(
        3,
        2,
        padding='same'
    )(x)

    x = x + res
    res = conv_batch_relu(
        x,
        filters=728,
        kernel_size=1,
        strides=2,
        padding='valid',
        relu=False
    )

    for i in range(2):
        x = sep_conv(x, 728, 3, 1, 'same')
    x = layers.MaxPooling2D(
        3,
        2,
        padding='same'
    )(x)

    x = x + res

    return x


def middle_flow(x):
    """
    middle flow for xception
    Args:
        x (): input tensor

    Returns: keras tensor

    """

    for i in range(8):
        x = x + middle_sep_conv_block(x, [728]*3, [3]*3, [1]*3, ['same']*3)
    return x


def exit_flow(x):
    """
    exit flow for xception
    Args:
        x (): input tensor

    Returns: keras tensor

    """
    res = conv_batch_relu(x, 1024, 1, 2, 'valid', False)
    x = sep_conv(x, 728, 3, 1, 'same')
    x = sep_conv(x, 1024, 3, 1, 'same')
    x = layers.MaxPooling2D(
        3,
        2,
        padding='same'
    )(x)

    x = x+res

    x = sep_conv(x, 1536, 3, 1, 'same', relu=False)
    x = sep_conv(x, 2048, 3, 1, 'same')
    x = layers.ReLU()(x)

    return x


def xception(input_shape=(299,299,3), num_classes=1000):
    """
    xception model based on https://arxiv.org/pdf/1610.02357.pdf
    Args:
        input_shape (tuple): input shape
        num_classes (int): number of categories

    Returns: xception model
    """

    input = layers.Input(shape=input_shape)

    x = entry_flow(input)

    x = middle_flow(x)

    x = exit_flow(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input, outputs=x)
    model.summary()

    return model
