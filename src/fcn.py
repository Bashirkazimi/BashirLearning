"""
Tensorflow keras implementation of U-Net
https://github.com/Bashirkazimi/BashirLearning
Author: Bashir Kazimi
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


def fcn(input_shape=(128, 128, 3), num_classes=21):
    """
    Fully Convolutional Networks for semantic segmentation implemented in
    tf.keras based on https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
    Args:
        input_shape (tuple): input shape
        num_classes (int): number of categories

    Returns: tf.keras.Model of fcn

    """

    base_model = tf.keras.applications.vgg16.VGG16(input_shape=input_shape,
                                                   weights=None,
                                                   include_top=False)
    layer_names = [
        'block3_pool',
        'block4_pool',
        'block5_pool'
    ]

    feature_layers = [base_model.get_layer(name).output for name in layer_names]

    feature_extractor = Model(inputs=base_model.input, outputs=feature_layers)

    inputs = layers.Input(shape=input_shape)

    pool3, pool4, pool5 = feature_extractor(inputs)

    x = layers.Conv2D(
        4096,
        1,
        1,
        padding='same'
    )(pool5)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(
        4096,
        1,
        1,
        padding='same'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(
        num_classes,
        1,
        1,
        padding='same'
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(
        num_classes,
        4,
        2,
        padding='same'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # pool4
    pool4 = layers.Conv2D(
        num_classes,
        1,
        1
    )(pool4)
    pool4 = layers.BatchNormalization()(pool4)
    pool4 = layers.Activation('relu')(pool4)

    x = pool4+x

    x = layers.Conv2DTranspose(
        num_classes,
        4,
        2,
        padding='same'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # pool3
    pool3 = layers.Conv2D(
        num_classes,
        1,
        1
    )(pool3)
    pool3 = layers.BatchNormalization()(pool3)
    pool3 = layers.Activation('relu')(pool3)

    x = x + pool3

    x = layers.Conv2DTranspose(
        num_classes,
        16,
        8,
        padding='same'
    )(x)

    x = layers.Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.summary()

    return model
