"""
Tensorflow keras implementation of alexnet
https://github.com/Bashirkazimi/BashirLearning
Author: Bashir Kazimi
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


def alex_net(input_shape=(224, 224, 3), num_classes=1000):
    """
    Alex Net implementation based on https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    Args:
        input_shape (tuple): input shape
        num_classes (int): number of categories

    Returns: tf.keras.Model for alexnet

    """
    inp = layers.Input(shape=input_shape)

    x = inp
    filters_list = [96, 256, 384, 384, 256]
    kernel_list = [11, 5, 3, 3, 3]
    stride_list = [4, 1, 1, 1, 1]
    max_pooled_layers = [0, 1, 4]

    for i in range(5):
        x = layers.Conv2D(
            filters=filters_list[i],
            kernel_size=kernel_list[i],
            strides=stride_list[i],
            activation='relu'
        )(x)

        if i in max_pooled_layers:
            x = layers.MaxPooling2D(
                2,
                2
            )(x)

    x = layers.Flatten()(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=x)
    model.summary()

    return model