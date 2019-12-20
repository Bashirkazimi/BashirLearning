"""
Tensorflow keras implementation of inception v1
https://github.com/Bashirkazimi/BashirLearning
Author: Bashir Kazimi
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


class LRN(layers.Layer):
    """
    Implementation of Local response normalization in Keras
    """
    def __init__(self):
        super(LRN, self).__init__()

    def call(self, inputs):
        return tf.nn.local_response_normalization(inputs)


def inception_block(inp, filters_list):
    """
    inception block for v1
    Args:
        inp (keras tensor): input tensor
        filters_list (int): list of filter sizes

    Returns: output of inception block

    """
    conv_1x1_1 = layers.Conv2D(
        filters_list[0],
        1,
        1,
        padding='same'
    )(inp)

    conv_1x1_2 = layers.Conv2D(
        filters_list[0],
        1,
        1,
        padding='same'
    )(inp)

    conv_1x1_3 = layers.Conv2D(
        filters_list[0],
        1,
        1,
        padding='same'
    )(inp)

    max_pool = layers.MaxPooling2D(
        3,
        1,
        padding='same'
    )(inp)

    conv_3x3 = layers.Conv2D(
        filters_list[1],
        3,
        1,
        padding='same'
    )(conv_1x1_2)

    conv_5x5 = layers.Conv2D(
        filters_list[2],
        5,
        1,
        padding='same'
    )(conv_1x1_3)

    conv_1x1_4 = layers.Conv2D(
        filters_list[3],
        1,
        1,
        padding='same'
    )(max_pool)

    return layers.Concatenate()([conv_1x1_1, conv_3x3, conv_5x5, conv_1x1_4])


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


def inception_v1(input_shape=(224,224,3), num_classes=1000):
    """
    Inception v1 implementation based on https://arxiv.org/pdf/1409.4842.pdf
    Args:
        input_shape (tuple): input shape
        num_classes (int): number of categories

    Returns: inception v1 model

    """
    inp = layers.Input(shape=input_shape)

    x = inp
    filters_list = [64, 192]
    kernels_list = [7, 3]
    strides_list = [2, 1]

    for i in range(2):
        x = layers.Conv2D(
            filters_list[i],
            kernels_list[i],
            strides_list[i],
            padding='same',
            activation='relu'
        )(x)

        x = layers.MaxPooling2D(
            3,
            2,
            padding='same'
        )(x)

        x = LRN()(x)

    # inception 3a
    # filters_list = [#1x1, #3x3, #5x5, pool proj]
    filters_list = [64, 128, 32, 32]
    x = inception_block(x, filters_list)

    # inception 3b
    # filters_list = [#1x1, #3x3, #5x5, pool proj]
    filters_list = [128, 192, 96, 64]
    x = inception_block(x, filters_list)

    x = layers.MaxPooling2D(
        3,
        2,
        padding='same'
    )(x)

    # inception 4a
    # filters_list = [#1x1, #3x3, #5x5, pool proj]
    filters_list = [192, 208, 48, 64]
    x = inception_block(x, filters_list)

    # inception 4b
    # filters_list = [#1x1, #3x3, #5x5, pool proj]
    filters_list = [160, 224, 64, 64]
    x = inception_block(x, filters_list)

    # inception 4c
    # filters_list = [#1x1, #3x3, #5x5, pool proj]
    filters_list = [128, 256, 64, 64]
    x = inception_block(x, filters_list)

    # inception 4d
    # filters_list = [#1x1, #3x3, #5x5, pool proj]
    filters_list = [112, 288, 64, 64]
    x = inception_block(x, filters_list)

    # inception 4e
    # filters_list = [#1x1, #3x3, #5x5, pool proj]
    filters_list = [256, 320, 128, 128]
    x = inception_block(x, filters_list)

    x = layers.MaxPooling2D(
        3,
        2,
        padding='same'
    )(x)

    # inception 5a
    # filters_list = [#1x1, #3x3, #5x5, pool proj]
    filters_list = [256, 320, 128, 128]
    x = inception_block(x, filters_list)

    # inception 5b
    # filters_list = [#1x1, #3x3, #5x5, pool proj]
    filters_list = [384, 384, 128, 128]
    x = inception_block(x, filters_list)

    # Originally implemented as averagepooling2d layer, but I am using
    # GlobalAveragePooling2D so that other input sizes could be used!
    x = layers.GlobalAveragePooling2D()(x)

    # x = layers.AveragePooling2D(
    #     7,
    #     1
    # )(x)

    x = layers.Dropout(rate=0.4)(x)

    # needed if averagepooling2d is used!
    # x = layers.Flatten()(x)

    x = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=x)
    model.summary()

    return model
