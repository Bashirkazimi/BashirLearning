"""
This module contains all models and layers

Author: Bashir Kazimi
"""

import tensorflow as tf


class LRN(tf.keras.layers.Layer):
    """
    Implementation of Local response normalization in Keras
    """
    def __init__(self):
        super(LRN, self).__init__()

    def call(self, inputs):
        return tf.nn.local_response_normalization(inputs)


def alex_net(
        input_shape=(224,224,3),
        num_classes=1000,
        kernel_list=[11, 11, 3, 3, 3],
        stride_list=[4, 1, 1, 1, 1],
        pool_list=[2, 2, 2],
        pool_stride_list=[2, 2, 2]
):
    """
    Returns alex net keras model. kernel, stride, and pool lists should be carefully given
    based on the input shape in order to avoid layer output dimensions to go negative due
    to strides/downsampling. Default parameters give alex net. For a similar model to ZFNet,
    give the following list after num_classes: [7, 5, 3, 3, 3], [2, 2, 1, 1, 1], [3, 3, 3], [2, 2, 2]
    :param input_shape: input shape
    :type input_shape: tuple with 3 integer elements (height, width, channels)
    :param num_classes: number of categories to classify
    :type num_classes: integer
    :param kernel_list: list of kernel size for each layer
    :type list of integers
    :param stride_list: list of stride size for each layer
    :type list of integers
    :param pool_list: list of pool size for each max pooling layer
    :type list of integers
    :param pool_stride_list: list of pool stride size for each max pooling layer
    :type list of integers
    :return: keras model of alex net
    :rtype: keras model
    """
    # initialize the model
    model = tf.keras.Sequential()

    # add first block of convolution, max pooling and batch normalization
    i = 0
    model.add(
        tf.keras.layers.Conv2D(
            filters=96,
            kernel_size=kernel_list[i],
            strides=stride_list[i],
            activation='relu',
            input_shape=input_shape,
            padding='valid'
        )
    )
    model.add(
        tf.keras.layers.MaxPooling2D(
            pool_size=pool_list[i],
            strides=pool_stride_list[i]
        )
    )
    model.add(
        tf.keras.layers.BatchNormalization()
    )

    # add second block of convolution, max pooling and batch normalization
    i = 1
    model.add(
        tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=kernel_list[i],
            strides=stride_list[i],
            activation='relu',
            padding='valid'
        )
    )
    model.add(
        tf.keras.layers.MaxPooling2D(
            pool_size=pool_list[i],
            strides=pool_stride_list[i]
        )
    )
    model.add(
        tf.keras.layers.BatchNormalization()
    )

    # add third block of convolution, and batch normalization
    i = 2
    model.add(
        tf.keras.layers.Conv2D(
            filters=384,
            kernel_size=kernel_list[i],
            strides=stride_list[i],
            activation='relu',
            padding='valid'
        )
    )

    model.add(
        tf.keras.layers.BatchNormalization()
    )

    # add fourth block of convolution, and batch normalization
    i = 3
    model.add(
        tf.keras.layers.Conv2D(
            filters=384,
            kernel_size=kernel_list[i],
            strides=stride_list[i],
            activation='relu',
            padding='valid'
        )
    )

    model.add(
        tf.keras.layers.BatchNormalization()
    )

    # add 5th block of convolution, max pooling and batch normalization
    i = 4
    model.add(
        tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=kernel_list[i],
            strides=stride_list[i],
            activation='relu',
            padding='valid'
        )
    )
    model.add(
        tf.keras.layers.MaxPooling2D(
            pool_size=pool_list[-1],
            strides=pool_stride_list[-1]
        )
    )
    model.add(
        tf.keras.layers.BatchNormalization()
    )

    # Flatten the outputs
    model.add(
        tf.keras.layers.Flatten()
    )

    # add FC layer 1
    model.add(
        tf.keras.layers.Dense(
            4096,
            activation='relu'
        )
    )
    model.add(
        tf.keras.layers.Dropout(
            rate=0.5
        )
    )
    model.add(
        tf.keras.layers.BatchNormalization()
    )

    # add FC layer 2
    model.add(
        tf.keras.layers.Dense(
            4096,
            activation='relu'
        )
    )
    model.add(
        tf.keras.layers.Dropout(
            rate=0.5
        )
    )
    model.add(
        tf.keras.layers.BatchNormalization()
    )

    # add FC layer 3 (output)
    model.add(
        tf.keras.layers.Dense(
            num_classes,
            activation='softmax'
        )
    )

    model.summary()

    return model


def vgg_net(
        input_shape=(224,224,3),
        num_classes=1000
):
    """
    Returns vgg net keras model.
    :param input_shape: input shape
    :type input_shape: tuple with 3 integer elements (height, width, channels)
    :param num_classes: number of categories to classify
    :type num_classes: integer
    :return: keras model of alex net
    :rtype: keras model
    """
    filters = [64, 128, 256, 512, 512]
    model = tf.keras.Sequential()

    # First Block
    i = 0
    model.add(
        tf.keras.layers.Conv2D(
            filters[i],
            3,
            input_shape=input_shape,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters[i],
            3,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        tf.keras.layers.MaxPooling2D(
            2,
            2
        )
    )

    # 2nd Block
    i = 1
    model.add(
        tf.keras.layers.Conv2D(
            filters[i],
            3,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters[i],
            3,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        tf.keras.layers.MaxPooling2D(
            2,
            2
        )
    )
    # 3rd Block
    i = 2
    model.add(
        tf.keras.layers.Conv2D(
            filters[i],
            3,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters[i],
            3,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters[i],
            3,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        tf.keras.layers.MaxPooling2D(
            2,
            2
        )
    )
    # 4th Block
    i = 3
    model.add(
        tf.keras.layers.Conv2D(
            filters[i],
            3,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters[i],
            3,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters[i],
            3,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        tf.keras.layers.MaxPooling2D(
            2,
            2
        )
    )
    # 5th Block
    i = 4
    model.add(
        tf.keras.layers.Conv2D(
            filters[i],
            3,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters[i],
            3,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters[i],
            3,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        tf.keras.layers.MaxPooling2D(
            2,
            2
        )
    )

    # flatten
    model.add(
        tf.keras.layers.Flatten()
    )

    # FC layers
    model.add(
        tf.keras.layers.Dense(
            4096,
            activation='relu'
        )
    )
    model.add(
        tf.keras.layers.Dense(
            4096,
            activation='relu'
        )
    )
    model.add(
        tf.keras.layers.Dense(
            num_classes,
            activation='softmax'
        )
    )

    model.summary()
    return model


class InceptionBlock(tf.keras.layers.Layer):
    """
    Implementation of the inception module
    """
    def __init__(self, filters):
        super(InceptionBlock, self).__init__()
        self.filters = filters

    def call(self, inputs):
        conv1 = tf.keras.layers.Conv2D(
            self.filters,
            1,
            1,
            padding='same'
        )(inputs)
        conv2 = tf.keras.layers.Conv2D(
            self.filters,
            1,
            1,
            padding='same'
        )(inputs)
        maxpool1 = tf.keras.layers.MaxPooling2D(
            3,
            1,
            padding='same'
        )(inputs)
        conv3 = tf.keras.layers.Conv2D(
            self.filters,
            1,
            1,
            padding='same'
        )(inputs)
        conv11 = tf.keras.layers.Conv2D(
            self.filters,
            3,
            1,
            padding='same'
        )(conv1)
        conv22 = tf.keras.layers.Conv2D(
            self.filters,
            5,
            1,
            padding='same'
        )(conv2)
        maxconv = tf.keras.layers.Conv2D(
            self.filters,
            1,
            1,
            padding='same'
        )(maxpool1)

        return tf.keras.layers.Concatenate()([conv3, conv11, conv22, maxconv])


def create_3_inceptions(input_tensor):
    """
    Given an input tensor, passes it through three blocks of inception and returns the output.
    Additionally, it passes the final result through fully connected layers and returns the output.
    :param input_tensor: input tensor to inceptify
    :type input_tensor: keras tensor
    :return: output of three inception blocks, and output of FC layers
    :rtype: keras tensor, keras tensor
    """

    inc_block1 = InceptionBlock(64)(input_tensor)
    inc_block2 = InceptionBlock(96)(inc_block1)
    inc_block3 = InceptionBlock(128)(inc_block2)

    avg_pool = tf.keras.layers.AveragePooling2D(
        5,
        3,
    )(inc_block3)

    conv_block_1 = tf.keras.layers.Conv2D(
        128,
        1,
        1,
        padding='same'
    )(avg_pool)

    # FC layers
    flattened = tf.keras.layers.Flatten()(conv_block_1)
    fc1 = tf.keras.layers.Dense(
        1024,
        activation='relu'
    )(flattened)
    fc1 = tf.keras.layers.Dropout(
        rate=0.5
    )(fc1)
    fc2 = tf.keras.layers.Dense(
        1024,
        activation='relu'
    )(fc1)
    fc2 = tf.keras.layers.Dropout(
        rate=0.5
    )(fc2)

    return inc_block3, fc2


def inception_net(input_shape=(224,224,3), num_classes=1000):
    """
    Returns the inception v1 model in keras. It creates three level classifier as shown in the original paper, but it
    could be edited to create a single level classifier. It is referred to as GoogLeNet
    :param input_shape: shape if the input image
    :type input_shape: tuple of 3 integers
    :param num_classes: number of categories
    :type num_classes: integer
    :return: keras model of inception v1
    :rtype: keras model
    """
    input_layer = tf.keras.layers.Input(
        shape=input_shape
    )
    conv1 = tf.keras.layers.Conv2D(
        64,
        7,
        2,
        activation='relu',
        padding='same'
    )(input_layer)
    max_pool1 = tf.keras.layers.MaxPooling2D(
        3,
        2,
        padding='same'
    )(conv1)

    lrn1 = LRN()(max_pool1)

    conv2 = tf.keras.layers.Conv2D(
        192,
        3,
        1,
        activation='relu',
        padding='same'
    )(lrn1)
    max_pool2 = tf.keras.layers.MaxPooling2D(
        3,
        2,
        padding='same'
    )(conv2)

    lrn2 = LRN()(max_pool2)

    max_pool3 = tf.keras.layers.MaxPooling2D(
        3,
        2,
        padding='same'
    )(lrn2)

    inceptioned_1, fc_1 = create_3_inceptions(max_pool3)
    softmax_layer_1 = tf.keras.layers.Dense(
        num_classes,
        activation='softmax'
    )(fc_1)

    inceptioned_2, fc_2 = create_3_inceptions(inceptioned_1)
    softmax_layer_2 = tf.keras.layers.Dense(
        num_classes,
        activation='softmax'
    )(fc_2)

    _, fc_3 = create_3_inceptions(inceptioned_2)
    softmax_layer_3 = tf.keras.layers.Dense(
        num_classes,
        activation='softmax'
    )(fc_3)

    model = tf.keras.Model(
        inputs=input_layer,
        outputs=[softmax_layer_1, softmax_layer_2, softmax_layer_3]
    )
    return model





