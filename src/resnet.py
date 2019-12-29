"""
Tensorflow keras implementation of ResNet v1 and v2
https://github.com/Bashirkazimi/BashirLearning
Author: Bashir Kazimi
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


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


def batch_relu_conv(x, filters, kernel_size, strides, padding, bn=False):
    """
    batch normalization, relu and convolution trio
    Args:
        x (keras tensor): input tensor
        filters (int): filter size
        kernel_size (int): kernel size
        strides (int): stride size
        padding (str): padding
        bn (bool): bn applied or not?

    Returns: keras tensor

    """
    if bn:
        x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding
    )(x)
    return x


def resnet_block_v2(inp, filters_list, kernels_list,
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
        x = batch_relu_conv(x, f, k, s, 'same', True)

    if res_type == 'identity':
        return x+inp
    else:
        return x+batch_relu_conv(inp, filters_list[-1], 1, strides_list[0],
                                 'same', True)


def resnet_block_v1(inp, filters_list, kernels_list,
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
    for f, k, s in zip(filters_list[:-1], kernels_list[:-1], strides_list[:-1]):
        x = conv_batch_relu(x, f, k, s, 'same', True)
    x = conv_batch_relu(x, filters_list[-1], kernels_list[-1], strides_list[-1], 'same', True, False)

    if res_type == 'identity':
        return layers.ReLU()(x+inp)
    else:
        return layers.ReLU()(x+conv_batch_relu(inp, filters_list[-1], 1, strides_list[0],
                                 'same', True, False))


def resnet(input_shape=(224, 224, 3), num_classes=1000, num_layers=50, version=2):
    """
    ResNet implementation based on https://arxiv.org/pdf/1603.05027.pdf and https://arxiv.org/pdf/1512.03385v1.pdf
    Args:
        input_shape (tuple): input tensor
        num_classes (int): number of categories
        version (int): version 1 or 2
        num_layers (int): one of 18, 34, 50, 101, 152 as in the paper

    Returns: keras model

    """
    # cbr or brc?
    blockfunc = resnet_block_v2 if version == 2 else resnet_block_v1
    # how many blocks to add based on num layers
    num_reps_dict = {
        18: [1, 1, 1, 1],
        34: [2, 3, 5, 2],
        50: [2, 3, 5, 2],
        101: [2, 3, 22, 2],
        152: [2, 7, 35, 2]
    }

    # filters list for small number of layers
    filters_list1 = [
        [64, 64],
        [128, 128],
        [256, 256],
        [512, 512]
    ]
    # filters list for large number of layers
    filters_list2 = [
        [64, 64, 256],
        [128, 128, 512],
        [256, 256, 1024],
        [512, 512, 2048],
    ]

    # fix number of filters and repetitions based on num_layers
    filters_lists = filters_list1 if num_layers <= 34 else filters_list2
    num_reps = num_reps_dict[num_layers]

    inp = layers.Input(shape=input_shape)
    if version == 2:
        x = layers.Conv2D(
            64,
            7,
            2,
            'same'
        )(inp)
    else:
        x = conv_batch_relu(inp, 64, 7, 2, 'same', True)

    x = layers.MaxPooling2D(
        3,
        2,
        padding='same'
    )(x)

    # strides and kernels based on number layers
    strides = [1, 1, 1] if num_layers >= 50 else [1, 1]
    kernels = [1, 3, 1] if num_layers >= 50 else [3, 3]

    for i in range(4):
        if i >= 1:
            # after second block, perform down convolution
            strides[0] = 2

        cur_filters = filters_lists[i]
        x = blockfunc(x, cur_filters, kernels, strides, 'conv')

        cur_num_reps = num_reps[i]
        strides[0] = 1
        for j in range(cur_num_reps):
            x = blockfunc(x, cur_filters, kernels, strides, 'identity')

    if version == 2:
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=x)
    model.summary()

    return model


