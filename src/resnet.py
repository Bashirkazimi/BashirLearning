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


def resnet(input_shape=(224, 224, 3), num_classes=1000, num_layers=50):
    """
    ResNet implementation based on https://arxiv.org/pdf/1512.03385v1.pdf
    Args:
        input_shape (tuple): input tensor
        num_classes (int): number of categories
        num_layers (int): one of 18, 34, 50, 101, 152 as in the paper

    Returns: keras model

    """
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
        x = resnet_block(x, cur_filters, kernels, strides, 'conv')

        cur_num_reps = num_reps[i]
        strides[0] = 1
        for j in range(cur_num_reps):
            print(cur_filters)
            x = resnet_block(x, cur_filters, kernels, strides, 'identity')

    # x = resnet_block(x, [64, 64, 256], [1, 3, 1], [1,1,1], 'conv')
    # for i in range(2):
    #   x = resnet_block(x, [64, 64, 256], [1, 3, 1], [1,1,1], 'identity')

    # x = resnet_block(x, [128, 128, 512], [1, 3, 1], [2,1,1], 'conv')
    # for i in range(3):
    #   x = resnet_block(x, [128, 128, 512], [1, 3, 1], [1,1,1], 'identity')

    # x = resnet_block(x, [256, 256, 1024], [1, 3, 1], [2,1,1], 'conv')
    # for i in range(5):
    #   x = resnet_block(x, [256, 256, 1024], [1, 3, 1], [1,1,1], 'identity')

    # x = resnet_block(x, [512, 512, 2048], [1, 3, 1], [2,1,1], 'conv')
    # for i in range(2):
    #   x = resnet_block(x, [512, 512, 2048], [1, 3, 1], [1,1,1], 'identity')

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=x)
    model.summary()

    return model

