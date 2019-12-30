"""
Tensorflow keras implementation of ResNext
https://github.com/Bashirkazimi/BashirLearning
Author: Bashir Kazimi
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
import numpy as np


def conv_batch_relu(x, filters, kernel_size, strides, padding, bn=False, act=True):
    """
    convolution batch normalization and relu trio
    Args:
        x (keras tensor): input tensor
        filters (int): filter size
        kernel_size (int): kernel size
        strides (int): stride size
        padding (str): padding
        bn (bool): bn applied or not?
        act (bool): apply relu or not?

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
    if act:
        x = layers.ReLU()(x)

    return x


def resnext_block_a(inp, filter_size,
                 strides_list, res_type='identity'):
    """
    resnext Block  as in figure 3a in the paper
    Args:
        inp (): input tensor
        filter_size (int): convolutional filters
        strides_list (ints): list of convolutional strides
        res_type (str): one of 'identity' or 'conv'

    Returns: resnet keras tensor

    """
    xs = []
    for i in range(32):
        x = conv_batch_relu(inp, filter_size // 32, 1, strides_list[0], 'same', True)
        x = conv_batch_relu(x, filter_size // 32, 3, strides_list[1], 'same', True)
        x = conv_batch_relu(x, filter_size * 2, 1, strides_list[2], 'same', True, False)

        xs.append(x)

    x = layers.Add()(xs)
    if res_type == 'identity':
        return layers.ReLU()(x+inp)
    else:
        return layers.ReLU()(x+conv_batch_relu(inp, filter_size *2, 1, strides_list[0],
                                 'same', True, False))


def resnext_block_b(inp, filter_size,
                 strides_list, res_type='identity'):
    """
    resnext Block as in figure 3b in the paper
    Args:
        inp (): input tensor
        filter_size (int): convolutional filters
        strides_list (ints): list of convolutional strides
        res_type (str): one of 'identity' or 'conv'

    Returns: resnet keras tensor

    """
    x_list = []
    for i in range(32):
        x = conv_batch_relu(inp, filter_size // 32, 1, strides_list[0], 'same', True)
        x = conv_batch_relu(x, filter_size // 32, 3, strides_list[1], 'same', True)
        x_list.append(x)
    x = layers.Concatenate()(x_list)

    x = conv_batch_relu(x, filter_size * 2, 1, strides_list[1], 'same', True, False)

    if res_type == 'identity':
        return layers.ReLU()(x+inp)
    else:
        return layers.ReLU()(x+conv_batch_relu(inp, filter_size *2, 1, strides_list[0],
                                 'same', True, False))


def resnext_block_c(inp, filter_size,
                 strides_list, res_type='identity'):
    """
    resnext Block as in figure 3c in the paper
    Args:
        inp (): input tensor
        filter_size (int): convolutional filters
        strides_list (ints): list of convolutional strides
        res_type (str): one of 'identity' or 'conv'

    Returns: resnet keras tensor

    """

    x = conv_batch_relu(inp, filter_size, 1, strides_list[0], 'same', True)
    x = layers.DepthwiseConv2D(
        3,
        strides_list[1],
        padding='same',
        depth_multiplier=filter_size//32
    )(x)
    c = filter_size//32
    kernel = np.zeros((1, 1, filter_size * c, filter_size), dtype=np.float32)
    for i in range(filter_size):
        start = (i // c) * c * c + i % c
        end = start + c * c
        kernel[:, :, start:end:c, i] = 1.
    x = layers.Conv2D(
        filter_size,
        1,
        trainable=False,
        kernel_initializer={'class_name': 'Constant',
                                          'config': {'value': kernel}}
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = conv_batch_relu(x, filter_size*2, 1, strides_list[-1], 'same', True, False)

    if res_type == 'identity':
        return layers.ReLU()(x+inp)
    else:
        return layers.ReLU()(x+conv_batch_relu(inp, filter_size *2, 1, strides_list[0],
                                 'same', True, False))


def resnext(input_shape=(224, 224, 3), num_classes=1000, num_layers=50, block_type='c'):
    """
    ResNext implementation based on https://arxiv.org/pdf/1611.05431.pdf
    Args:
        input_shape (tuple): input tensor
        num_classes (int): number of categories
        num_layers (int): one of 18, 34, 50, 101, 152 as in the paper
        block_type (str): type of resnet, one of ['a','b','c'] as in the paper

    Returns: keras model

    """
    blocktypes = {
        "a": resnext_block_a,
        "b": resnext_block_b,
        "c": resnext_block_c
    }
    resnext_block = blocktypes[block_type]

    # how many blocks to add based on num layers
    num_reps_dict = {
        18: [1, 1, 1, 1],
        34: [2, 3, 5, 2],
        50: [2, 3, 5, 2],
        101: [2, 3, 22, 2],
        152: [2, 7, 35, 2]
    }

    filters_list = [128, 256, 512, 1024]
    num_reps = num_reps_dict[num_layers]

    inp = layers.Input(shape=input_shape)
    x = conv_batch_relu(inp, 64, 7, 2, 'same', True)

    x = layers.MaxPooling2D(
        3,
        2,
        padding='same'
    )(x)

    strides = [1, 1, 1]
    filters_list = [128, 256, 512, 1024]
    for i in range(4):
        if i >= 1:
            # after second block, perform down convolution
            strides[0] = 2

        cur_filters = filters_list[i]
        x = resnext_block(x, cur_filters, strides, 'conv')

        cur_num_reps = num_reps[i]
        strides[0] = 1
        for j in range(cur_num_reps):
            x = resnext_block(x, cur_filters, strides, 'identity')

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=x)
    model.summary()

    return model
