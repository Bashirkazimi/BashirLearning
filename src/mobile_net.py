"""
Tensorflow keras implementation of mobile net
https://github.com/Bashirkazimi/BashirLearning
Author: Bashir Kazimi
"""


import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


def mobile_dw_conv(input_tensor, filter_size, kernel, stride):
    """
    a depthwise separable convolution with given filter and kernel size and
    stride
    Args:
        input_tensor (keras tensor): input tensor
        filter_size (int): filter size
        kernel (int): kernel size
        stride (int): stride size

    Returns: keras tensor
    """
    x = layers.DepthwiseConv2D(
        kernel,
        stride,
        'same',
        1,
        use_bias=False
    )(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)
    x = layers.Conv2D(
        filter_size,
        1,
        1,
        padding='same',
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)

    return x


def mobile_net(input_shape=(224,224,3), num_classes=1000):
    """
    mobile net v1 based on https://arxiv.org/pdf/1704.04861.pdf
    Args:
        input_shape (tuple): input shape
        num_classes (int): number of categories

    Returns: mobile net model
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

    x = mobile_dw_conv(x, 64, 3, 1)
    x = mobile_dw_conv(x, 128, 3, 2)
    x = mobile_dw_conv(x, 128, 3, 1)
    x = mobile_dw_conv(x, 256, 3, 2)
    x = mobile_dw_conv(x, 256, 3, 1)
    x = mobile_dw_conv(x, 512, 3, 2)

    for i in range(5):
        x = mobile_dw_conv(x, 512, 3, 1)

    x = mobile_dw_conv(x, 1024, 3, 2)
    x = mobile_dw_conv(x, 1024, 3, 1)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(num_classes)(x)
    x = layers.Activation('softmax')(x)

    model = Model(inputs=input, outputs=x)
    model.summary()

    return model

