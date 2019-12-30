"""
Tensorflow keras implementation of inception v4
https://github.com/Bashirkazimi/BashirLearning
Author: Bashir Kazimi
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


def v2_stem(input_tensor):
    """
    stem for inception v4
    Args:
        input_tensor (keras tensor): input tensor
    Returns: keras tensor
    """
    x = layers.Conv2D(
        32,
        3,
        2,
        padding='valid'
    )(input_tensor)
    x = layers.Conv2D(
        32,
        3,
        1,
        padding='valid'
    )(x)
    x = layers.Conv2D(
        64,
        3,
        1,
        padding='same'
    )(x)
    m = layers.MaxPooling2D(
        3,
        2,
        padding='valid'
    )(x)
    conv = layers.Conv2D(
        96,
        3,
        2,
        padding='valid'
    )(x)
    x = layers.Concatenate()([m, conv])
    x1 = layers.Conv2D(
        64,
        1,
        1,
        padding='same'
    )(x)
    x1 = layers.Conv2D(
        64,
        (7,1),
        1,
        padding='same'
    )(x1)
    x1 = layers.Conv2D(
        64,
        (1,7),
        1,
        padding='same'
    )(x1)
    x1 = layers.Conv2D(
        96,
        3,
        1,
        padding='valid'
    )(x1)
    x2 = layers.Conv2D(
        64,
        1,
        1,
        padding='same'
    )(x)
    x2 = layers.Conv2D(
        96,
        3,
        1,
        padding='valid'
    )(x2)
    x = layers.Concatenate()([x2, x1])
    conv = layers.Conv2D(
        192,
        3,
        2,
        padding='valid'
    )(x)
    m = layers.MaxPooling2D(
        strides=2,
        padding='valid'
    )(x)
    x = layers.Concatenate()([m, conv])
    return x


def inception_v4_a(input_tensor):
    """
    inception v4 block a
    Args:
        input_tensor (keras tensor): input tensor
    Returns: keras tensor
    """
    avgpool = layers.AveragePooling2D(
        padding='same',
        strides=1
    )(input_tensor)
    conv_pool = layers.Conv2D(
        96,
        1,
        1,
        padding='same'
    )(avgpool)
    conv1 = layers.Conv2D(
        96,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv2 = layers.Conv2D(
        64,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv2 = layers.Conv2D(
        96,
        3,
        1,
        padding='same'
    )(conv2)
    conv3 = layers.Conv2D(
        64,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv3 = layers.Conv2D(
        96,
        3,
        1,
        padding='same'
    )(conv3)
    conv3 = layers.Conv2D(
        96,
        3,
        1,
        padding='same'
    )(conv3)
    concat = layers.Concatenate()([conv_pool, conv1, conv2, conv3])

    return concat


def inception_resnet_reduction_a(input_tensor, version=1):
    """
    reduction a block for inception resnet
    Args:
        input_tensor (keras tensor): input tensor
        version (int): version, one of [1,2,4] for inception resnet v1, v2 and inception v4
    Returns: keras tensor
    """
    if version == 1:
        k, l, m, n = [192, 192, 256, 384]
    elif version == 2:
        k, l, m, n = [256, 256, 384, 384]
    else: # if inception v4
        k, l, m, n = [192, 224, 256, 384]
    maxpool = layers.MaxPooling2D(
        3,
        2,
        padding='valid'
    )(input_tensor)
    conv1 = layers.Conv2D(
        n,
        3,
        2,
        padding='valid'
    )(input_tensor)
    conv2 = layers.Conv2D(
        k,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv2 = layers.Conv2D(
        l,
        3,
        1,
        padding='same'
    )(conv2)
    conv2 = layers.Conv2D(
        m,
        3,
        2,
        padding='valid'
    )(conv2)
    x = layers.Concatenate()([maxpool, conv1, conv2])
    return x


def inception_v4_b(input_tensor):
    """
    b block for inception v4
    Args:
        input_tensor (keras tensor): input tensor
    Returns: keras tensor
    """
    avgpool = layers.AveragePooling2D(
        strides=1,
        padding='same'
    )(input_tensor)
    conv_pool = layers.Conv2D(
        128,
        1,
        1,
        padding='same'
    )(avgpool)
    conv1 = layers.Conv2D(
        384,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv2 = layers.Conv2D(
        192,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv2 = layers.Conv2D(
        224,
        (1,7),
        1,
        padding='same'
    )(conv2)
    conv2 = layers.Conv2D(
        256,
        (1,7),
        1,
        padding='same'
    )(conv2)
    conv3 = layers.Conv2D(
        192,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv3 = layers.Conv2D(
        192,
        (1,7),
        1,
        padding='same'
    )(conv3)
    conv3 = layers.Conv2D(
        224,
        (7,1),
        1,
        padding='same'
    )(conv3)
    conv3 = layers.Conv2D(
        224,
        (1,7),
        1,
        padding='same'
    )(conv3)
    conv3 = layers.Conv2D(
        256,
        (7,1),
        1,
        padding='same'
    )(conv3)
    concat = layers.Concatenate()([conv_pool, conv1, conv2, conv3])

    return concat


def inception_v4_reduction_b(input_tensor):
    """
    reduction b for inception v4
    Args:
        input_tensor (keras tensor): input tensor
    Returns: keras tensor
    """
    maxpoool = layers.MaxPooling2D(
        3,
        2,
        padding='valid'
    )(input_tensor)
    conv1 = layers.Conv2D(
        192,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv1 = layers.Conv2D(
        192,
        3,
        2,
        padding='valid'
    )(conv1)
    conv2 = layers.Conv2D(
        256,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv2 = layers.Conv2D(
        256,
        (1,7),
        1,
        padding='same'
    )(conv2)
    conv2 = layers.Conv2D(
        320,
        (7,1),
        1,
        padding='same'
    )(conv2)
    conv2 = layers.Conv2D(
        320,
        3,
        2,
        padding='valid'
    )(conv2)
    x = layers.Concatenate()([maxpoool, conv1, conv2])

    return x


def inception_v4_c(input_tensor):
    """
    block c of inception v4
    Args:
        input_tensor (keras tensor): input tensor
    Returns: keras tensor
    """
    avgpool = layers.AveragePooling2D(
        padding='same',
        strides=1
    )(input_tensor)
    conv_pool = layers.Conv2D(
        256,
        1,
        1,
        padding='same'
    )(avgpool)
    conv1 = layers.Conv2D(
        256,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv2 = layers.Conv2D(
        384,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv21 = layers.Conv2D(
        256,
        (1,3),
        1,
        padding='same'
    )(conv2)
    conv22 = layers.Conv2D(
        256,
        (3,1),
        1,
        padding='same'
    )(conv2)
    conv3 = layers.Conv2D(
        384,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv3 = layers.Conv2D(
        448,
        (1,3),
        1,
        padding='same'
    )(conv3)
    conv3 = layers.Conv2D(
        512,
        (3,1),
        1,
        padding='same'
    )(conv3)
    conv31 = layers.Conv2D(
        256,
        (3,1),
        1,
        padding='same'
    )(conv3)
    conv32 = layers.Conv2D(
        256,
        (1,3),
        1,
        padding='same'
    )(conv3)
    x = layers.Concatenate()([conv_pool, conv1, conv21, conv22, conv31, conv32])

    return x


def inception_v4(input_shape=(299,299,3), num_classes=1000):
    """
    inception v4 based on https://arxiv.org/pdf/1602.07261.pdf
    Args:
        input_shape (tuple): input shape
        num_classes (int): number of categories

    Returns: inception v4 model
    """
    version = 4
    input = layers.Input(shape=input_shape)
    x = v2_stem(input)  # stem for this and inception resnet v2 are similar

    for i in range(4):
        x = inception_v4_a(x)

    x = inception_resnet_reduction_a(x, version)

    for i in range(7):
        x = inception_v4_b(x)

    x = inception_v4_reduction_b(x)

    for i in range(3):
        x = inception_v4_c(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input, outputs=x)
    model.summary()

    return model
