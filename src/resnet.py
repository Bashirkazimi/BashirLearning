"""
Tensorflow keras implementation of ResNet
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


def resnet_block(inp, filters_list, kernels_list,
                 strides_list, res_type='identity'):
    """
    ResNet Block
    Args:
        inp (): input tensor
        filters_list (ints): list of convolutional filters
        kernels_list (ints): list of convolutional kernels
        strides_list (ints): list of convolutional strides
        res_type (str): one of 'identity' or 'conv'

    Returns: resnet keras tensor

    """
    x = inp
    for f, k, s in zip(filters_list, kernels_list, strides_list):
        x = conv_batch_relu(x, f, k, s, 'same', True)

    if res_type == 'identity':
        return x+inp
    else:
        return x+conv_batch_relu(inp, filters_list[-1], 1, strides_list[-1],
                                 'same', True)






def resnet50(input_shape=(224,224,3), num_classes=1000):
    """
    ResNet50 implementation based on https://arxiv.org/pdf/1512.03385v1.pdf
    Args:
        input_shape (tuple): input tensor
        num_classes (int): number of categories

    Returns: keras model

    """

    inp = layers.Input(shape=input_shape)

    x = conv_batch_relu(inp, 64, 7, 2, 'same', True)

    x = layers.MaxPooling2D(
        3,
        2,
        padding='same'
    )(x)



    model = Model(inputs=inp, outputs=x)
    model.summary()

    return model