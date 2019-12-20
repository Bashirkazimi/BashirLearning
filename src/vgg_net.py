"""
Tensorflow keras implementation of vgg net
https://github.com/Bashirkazimi/BashirLearning
Author: Bashir Kazimi
"""


import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


def vgg16(input_shape=(224,224,3),num_classes=1000):
    """
    tf.keras implementation of vgg16 net based on
    https://arxiv.org/pdf/1409.1556.pdf
    Args:
        input_shape (tuple): input shape
        num_classes (int): number of categories

    Returns: tf.keras.Model for vgg16 net

    """
    inp = layers.Input(shape=input_shape)
    x = inp
    filters_list = [64, 128, 256, 512, 512]

    for i in range(5):
        num_convs = 3 if i >= 2 else 2
        for j in range(num_convs):
            x = layers.Conv2D(
                filters_list[i],
                3,
                padding='same',
                activation='relu'
            )(x)
        x = layers.MaxPooling2D(
            2,
            2
        )(x)

    x = layers.Flatten()(x)
    x = layers.Dense(4096)(x)
    x = layers.Dense(4096)(x)

    x = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=x)
    model.summary()

    return model


def vgg19(input_shape=(224,224,3),num_classes=1000):
    """
    tf.keras implementation of vgg19 net based on
    https://arxiv.org/pdf/1409.1556.pdf
    Args:
        input_shape (tuple): input shape
        num_classes (int): number of categories

    Returns: tf.keras.Model for vgg16 net

    """
    inp = layers.Input(shape=input_shape)
    x = inp
    filters_list = [64, 128, 256, 512, 512]

    for i in range(5):
        num_convs = 4 if i >= 2 else 2
        for j in range(num_convs):
            x = layers.Conv2D(
                filters_list[i],
                3,
                padding='same',
                activation='relu'
            )(x)
        x = layers.MaxPooling2D(
            2,
            2
        )(x)

    x = layers.Flatten()(x)
    x = layers.Dense(4096)(x)
    x = layers.Dense(4096)(x)

    x = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=x)
    model.summary()

    return model


