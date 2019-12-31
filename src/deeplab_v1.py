"""
Tensorflow keras implementation of deeplab v1
https://github.com/Bashirkazimi/BashirLearning
Author: Bashir Kazimi
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


def atrous_conv(base_output, dilation_rate=12, filter_size=1024, kernel_size=3,
                num_classes=21):
    """
    Atrous Convolution and two 1x1 convolution operations
    Args:
        base_output (): input tensor
        dilation_rate (int): dilation rate
        filter_size (int): filter size
        kernel_size (int): kernel size
        num_classes (int): number of categores

    Returns: atrous convolution, keras tensor

    """
    x = layers.Conv2D(
      filter_size,
      kernel_size,
      dilation_rate=dilation_rate,
      padding='SAME'
    )(base_output)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for i in range(2):
        if i == 1:
            fs = num_classes
        else:
            fs = filter_size

        x = layers.Conv2D(
            fs,
            1,
            padding='same'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    return x


def deeplab_v1(input_shape=(513,513,3), num_classes=21, backbone='VGG'):
    """
    deeplab v1 model based on https://arxiv.org/pdf/1412.7062.pdf
    Args:
        input_shape (): input tensor
        num_classes (int): number of categories
        backbone (str): feature extractor, one of ['VGG', 'Xception', 'ResNet']

    Returns:

    """
    if backbone == 'VGG':
        base_model = tf.keras.applications.vgg16.VGG16(weights=None,
                                                       input_shape=input_shape,
                                                       include_top=False)
    elif backbone == 'Xception':
        base_model = tf.keras.applications.xception.Xception(
            weights=None,
            input_shape=input_shape,
            include_top=False
        )
    else:  # resnet
        base_model = tf.keras.applications.resnet_v2.ResNet101V2(
            weights=None,
            input_shape=input_shape,
            include_top=False
        )

    base_output = base_model.output

    x = atrous_conv(
        base_output,
        dilation_rate=12,
        filter_size=1024,
        kernel_size=3,
        num_classes=num_classes)

    # Originally, instead of upsampling this, the ground truths are
    # downsampled during training, but we upsample the logits. Comment the
    # following line out if downsampling gts is what you want
    x = tf.image.resize(
        x,
        size=input_shape[:2]
    )

    softmax = layers.Activation('softmax')(x)

    model = Model(inputs=base_model.input, outputs=softmax)
    model.summary()

    return model

