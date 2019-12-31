"""
Tensorflow keras implementation of inception resnet v1 and v2
https://github.com/Bashirkazimi/BashirLearning
Author: Bashir Kazimi
"""


import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


def v2_stem(input_tensor):
    """
    stem for inception resnet v2
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
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        32,
        3,
        1,
        padding='valid'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        64,
        3,
        1,
        padding='same'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
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
    conv = layers.BatchNormalization()(conv)
    conv = layers.ReLU()(conv)
    x = layers.Concatenate()([m, conv])
    x1 = layers.Conv2D(
        64,
        1,
        1,
        padding='same'
    )(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x1 = layers.Conv2D(
        64,
        (7,1),
        1,
        padding='same'
    )(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x1 = layers.Conv2D(
        64,
        (1,7),
        1,
        padding='same'
    )(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x1 = layers.Conv2D(
        96,
        3,
        1,
        padding='valid'
    )(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x2 = layers.Conv2D(
        64,
        1,
        1,
        padding='same'
    )(x)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x2 = layers.Conv2D(
        96,
        3,
        1,
        padding='valid'
    )(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x = layers.Concatenate()([x2, x1])
    conv = layers.Conv2D(
        192,
        3,
        2,
        padding='valid'
    )(x)
    conv = layers.BatchNormalization()(conv)
    conv = layers.ReLU()(conv)
    m = layers.MaxPooling2D(
        strides=2,
        padding='valid'
    )(x)
    x = layers.Concatenate()([m, conv])
    return x


def v1_stem(input_tensor):
    """
    stem for inception resnet v1
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
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        32,
        3,
        1,
        padding='valid'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        64,
        3,
        1,
        padding='same'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(
        3,
        2,
        padding='valid'
    )(x)
    x = layers.Conv2D(
        80,
        1,
        1,
        padding='same'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        192,
        3,
        1,
        padding='valid'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        256,
        3,
        2,
        padding='valid'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def inception_resnet_v1_a(input_tensor):
    """
    inception resnet a block v1
    Args:
        input_tensor (keras tensor): input tensor
    Returns: keras tensor
    """
    x = input_tensor
    x1 = layers.Conv2D(
        32,
        1,
        1,
        padding='same'
    )(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x2 = layers.Conv2D(
        32,
        1,
        1,
        padding='same'
    )(x)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x2 = layers.Conv2D(
        32,
        3,
        1,
        padding='same'
    )(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x3 = layers.Conv2D(
        32,
        1,
        1,
        padding='same'
    )(x)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)
    x3 = layers.Conv2D(
        32,
        3,
        1,
        padding='same'
    )(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)
    x3 = layers.Conv2D(
        32,
        3,
        1,
        padding='same'
    )(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)
    concat = layers.Concatenate()([x1, x2, x3])
    conv_concat = layers.Conv2D(
        256,
        1,
        1,
        padding='same'
    )(concat)
    conv_concat = layers.BatchNormalization()(conv_concat)
    conv_concat = layers.ReLU()(conv_concat)
    x = x+conv_concat

    return x


def inception_resnet_v2_a(input_tensor):
    """
    inception resnet block a for v2
    Args:
        input_tensor (keras tensor): input tensor
    Returns: keras tensor
    """
    x = input_tensor
    x1 = layers.Conv2D(
        32,
        1,
        1,
        padding='same'
    )(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x2 = layers.Conv2D(
        32,
        1,
        1,
        padding='same'
    )(x)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x2 = layers.Conv2D(
        32,
        3,
        1,
        padding='same'
    )(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x3 = layers.Conv2D(
        32,
        1,
        1,
        padding='same'
    )(x)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)
    x3 = layers.Conv2D(
        48,
        3,
        1,
        padding='same'
    )(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)
    x3 = layers.Conv2D(
        64,
        3,
        1,
        padding='same'
    )(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)
    concat = layers.Concatenate()([x1,x2,x3])
    conv_concat = layers.Conv2D(
        384,
        1,
        1,
        padding='same'
    )(concat)
    conv_concat = layers.BatchNormalization()(conv_concat)
    conv_concat = layers.ReLU()(conv_concat)
    x = x+conv_concat

    return x


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
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.ReLU()(conv1)
    conv2 = layers.Conv2D(
        k,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.ReLU()(conv2)
    conv2 = layers.Conv2D(
        l,
        3,
        1,
        padding='same'
    )(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.ReLU()(conv2)
    conv2 = layers.Conv2D(
        m,
        3,
        2,
        padding='valid'
    )(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.ReLU()(conv2)
    x = layers.Concatenate()([maxpool, conv1, conv2])
    return x


def inception_resnet_v1_b(input_tensor):
    """
    b block for inception resnet v1
    Args:
        input_tensor (keras tensor): input tensor
    Returns: keras tensor
    """
    x = input_tensor
    x1 = layers.Conv2D(
        128,
        1,
        1,
        padding='same'
    )(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x2 = layers.Conv2D(
        128,
        1,
        1,
        padding='same'
    )(x)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x2 = layers.Conv2D(
        128,
        (1,7),
        1,
        padding='same'
    )(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x2 = layers.Conv2D(
        128,
        (7,1),
        1,
        padding='same'
    )(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    concat = layers.Concatenate()([x1,x2])
    conv_concat = layers.Conv2D(
        896,
        1,
        1,
        padding='same'
    )(concat)
    conv_concat = layers.BatchNormalization()(conv_concat)
    conv_concat = layers.ReLU()(conv_concat)

    x = x+conv_concat

    return x


def inception_resnet_v2_b(input_tensor):
    """
    b block of inception resnet v2
    Args:
        input_tensor (keras tensor): input tensor
    Returns: keras tensor
    """
    x = input_tensor
    x1 = layers.Conv2D(
        192,
        1,
        1,
        padding='same'
    )(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x2 = layers.Conv2D(
        128,
        1,
        1,
        padding='same'
    )(x)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x2 = layers.Conv2D(
        160,
        (1,7),
        1,
        padding='same'
    )(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x2 = layers.Conv2D(
        192,
        (7,1),
        1,
        padding='same'
    )(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    concat = layers.Concatenate()([x1,x2])
    conv_concat = layers.Conv2D(
        1152,
        1,
        1,
        padding='same'
    )(concat)
    conv_concat = layers.BatchNormalization()(conv_concat)
    conv_concat = layers.ReLU()(conv_concat)
    x = x+conv_concat

    return x


def reduction_b_v1(input_tensor):
    """
    reduction b block for v1 of inception resnet
    Args:
        input_tensor (keras tensor): input tensor
    Returns: keras tensor
    """
    maxpool = layers.MaxPooling2D(
        3,
        2,
        padding='valid'
    )(input_tensor)
    conv1 = layers.Conv2D(
        256,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.ReLU()(conv1)
    conv1 = layers.Conv2D(
        384,
        3,
        2,
        padding='valid'
    )(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.ReLU()(conv1)
    conv2 = layers.Conv2D(
        256,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.ReLU()(conv2)
    conv2 = layers.Conv2D(
        256,
        3,
        2,
        padding='valid'
    )(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.ReLU()(conv2)
    conv3 = layers.Conv2D(
        256,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.ReLU()(conv3)
    conv3 = layers.Conv2D(
        256,
        3,
        1,
        padding='same'
    )(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.ReLU()(conv3)
    conv3 = layers.Conv2D(
        256,
        3,
        2,
        padding='valid'
    )(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.ReLU()(conv3)
    concat = layers.Concatenate()([maxpool, conv1, conv2, conv3])

    return concat


def reduction_b_v2(input_tensor):
    """
    reduction b block for v2 of inception resnet
    Args:
        input_tensor (keras tensor): input tensor
    Returns: keras tensor
    """
    maxpool = layers.MaxPooling2D(
        3,
        2,
        padding='valid'
    )(input_tensor)
    conv1 = layers.Conv2D(
        256,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.ReLU()(conv1)
    conv1 = layers.Conv2D(
        384,
        3,
        2,
        padding='valid'
    )(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.ReLU()(conv1)
    conv2 = layers.Conv2D(
        256,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.ReLU()(conv2)
    conv2 = layers.Conv2D(
        288,
        3,
        2,
        padding='valid'
    )(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.ReLU()(conv2)
    conv3 = layers.Conv2D(
        256,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.ReLU()(conv3)
    conv3 = layers.Conv2D(
        288,
        3,
        1,
        padding='same'
    )(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.ReLU()(conv3)
    conv3 = layers.Conv2D(
        320,
        3,
        2,
        padding='valid'
    )(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.ReLU()(conv3)
    concat = layers.Concatenate()([maxpool, conv1, conv2, conv3])

    return concat


def inception_resnet_v1_c(input_tensor):
    """
    c block for inception resnet v1
    Args:
        input_tensor (keras tensor): input tensor
    Returns: keras tensor
    """
    x = input_tensor
    conv1 = layers.Conv2D(
        192,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.ReLU()(conv1)
    conv2 = layers.Conv2D(
        192,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.ReLU()(conv2)
    conv2 = layers.Conv2D(
        192,
        (1,3),
        1,
        padding='same'
    )(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.ReLU()(conv2)
    conv2 = layers.Conv2D(
        192,
        (3,1),
        1,
        padding='same'
    )(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.ReLU()(conv2)
    concat = layers.Concatenate()([conv1, conv2])
    conv_concat = layers.Conv2D(
        1792,
        1,
        1,
        padding='same'
    )(concat)
    conv_concat = layers.BatchNormalization()(conv_concat)
    conv_concat = layers.ReLU()(conv_concat)
    x = x+conv_concat

    return x


def inception_resnet_v2_c(input_tensor):
    """
    c block for inception resnet v2
    Args:
        input_tensor (keras tensor): input tensor
    Returns: keras tensor
    """
    x = input_tensor
    conv1 = layers.Conv2D(
        192,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.ReLU()(conv1)
    conv2 = layers.Conv2D(
        192,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.ReLU()(conv2)
    conv2 = layers.Conv2D(
        224,
        (1,3),
        1,
        padding='same'
    )(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.ReLU()(conv2)
    conv2 = layers.Conv2D(
        256,
        (3,1),
        1,
        padding='same'
    )(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.ReLU()(conv2)
    concat = layers.Concatenate()([conv1, conv2])
    conv_concat = layers.Conv2D(
        2144,
        1,
        1,
        padding='same'
    )(concat)
    conv_concat = layers.BatchNormalization()(conv_concat)
    conv_concat = layers.ReLU()(conv_concat)
    x = x+conv_concat

    return x


def inception_resnet(input_shape=(299,299,3), num_classes=1000, version=1):
    """
    inception resnet models v1 and v2 based on https://arxiv.org/pdf/1602.07261.pdf
    Args:
        input_shape (tuple): input shape
        num_classes (int): number of categories
        version (int): version 1 or 2 for inception resnet v1 or v2

    Returns: inception v1 or v2 keras model
    """
    input = layers.Input(shape=input_shape)
    x = v1_stem(input) if version == 1 else v2_stem(input)

    for i in range(5):
        x = inception_resnet_v1_a(x) if version == 1 else inception_resnet_v2_a(x)

    x = inception_resnet_reduction_a(x, version)

    for i in range(10):
        x = inception_resnet_v1_b(x) if version == 1 else inception_resnet_v2_b(x)

    x = reduction_b_v1(x) if version == 1 else reduction_b_v2(x)

    for i in range(5):
        x = inception_resnet_v1_c(x) if version == 1 else inception_resnet_v2_c(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input, outputs=x)
    model.summary()

    return model
