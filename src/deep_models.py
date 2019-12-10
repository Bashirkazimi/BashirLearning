"""
This module contains all models and layers

Author: Bashir Kazimi
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


class LRN(layers.Layer):
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
    Returns alex net keras model:
    https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    kernel, stride, and pool lists should be carefully given
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
        layers.Conv2D(
            filters=96,
            kernel_size=kernel_list[i],
            strides=stride_list[i],
            activation='relu',
            input_shape=input_shape,
            padding='valid'
        )
    )
    model.add(
        layers.MaxPooling2D(
            pool_size=pool_list[i],
            strides=pool_stride_list[i]
        )
    )
    model.add(
        layers.BatchNormalization()
    )

    # add second block of convolution, max pooling and batch normalization
    i = 1
    model.add(
        layers.Conv2D(
            filters=256,
            kernel_size=kernel_list[i],
            strides=stride_list[i],
            activation='relu',
            padding='valid'
        )
    )
    model.add(
        layers.MaxPooling2D(
            pool_size=pool_list[i],
            strides=pool_stride_list[i]
        )
    )
    model.add(
        layers.BatchNormalization()
    )

    # add third block of convolution, and batch normalization
    i = 2
    model.add(
        layers.Conv2D(
            filters=384,
            kernel_size=kernel_list[i],
            strides=stride_list[i],
            activation='relu',
            padding='valid'
        )
    )

    model.add(
        layers.BatchNormalization()
    )

    # add fourth block of convolution, and batch normalization
    i = 3
    model.add(
        layers.Conv2D(
            filters=384,
            kernel_size=kernel_list[i],
            strides=stride_list[i],
            activation='relu',
            padding='valid'
        )
    )

    model.add(
        layers.BatchNormalization()
    )

    # add 5th block of convolution, max pooling and batch normalization
    i = 4
    model.add(
        layers.Conv2D(
            filters=256,
            kernel_size=kernel_list[i],
            strides=stride_list[i],
            activation='relu',
            padding='valid'
        )
    )
    model.add(
        layers.MaxPooling2D(
            pool_size=pool_list[-1],
            strides=pool_stride_list[-1]
        )
    )
    model.add(
        layers.BatchNormalization()
    )

    # Flatten the outputs
    model.add(
        layers.Flatten()
    )

    # add FC layer 1
    model.add(
        layers.Dense(
            4096,
            activation='relu'
        )
    )
    model.add(
        layers.Dropout(
            rate=0.5
        )
    )
    model.add(
        layers.BatchNormalization()
    )

    # add FC layer 2
    model.add(
        layers.Dense(
            4096,
            activation='relu'
        )
    )
    model.add(
        layers.Dropout(
            rate=0.5
        )
    )
    model.add(
        layers.BatchNormalization()
    )

    # add FC layer 3 (output)
    model.add(
        layers.Dense(
            num_classes,
            activation='softmax'
        )
    )

    model.summary()

    return model


def vgg_net(input_shape=(224,224,3),num_classes=1000):
    """
    Returns vgg net keras model: https://arxiv.org/pdf/1409.1556.pdf
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
        layers.Conv2D(
            filters[i],
            3,
            input_shape=input_shape,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        layers.Conv2D(
            filters[i],
            3,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        layers.MaxPooling2D(
            2,
            2
        )
    )

    # 2nd Block
    i = 1
    model.add(
        layers.Conv2D(
            filters[i],
            3,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        layers.Conv2D(
            filters[i],
            3,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        layers.MaxPooling2D(
            2,
            2
        )
    )
    # 3rd Block
    i = 2
    model.add(
        layers.Conv2D(
            filters[i],
            3,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        layers.Conv2D(
            filters[i],
            3,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        layers.Conv2D(
            filters[i],
            3,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        layers.MaxPooling2D(
            2,
            2
        )
    )
    # 4th Block
    i = 3
    model.add(
        layers.Conv2D(
            filters[i],
            3,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        layers.Conv2D(
            filters[i],
            3,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        layers.Conv2D(
            filters[i],
            3,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        layers.MaxPooling2D(
            2,
            2
        )
    )
    # 5th Block
    i = 4
    model.add(
        layers.Conv2D(
            filters[i],
            3,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        layers.Conv2D(
            filters[i],
            3,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        layers.Conv2D(
            filters[i],
            3,
            activation='relu',
            padding='same'
        )
    )
    model.add(
        layers.MaxPooling2D(
            2,
            2
        )
    )

    # flatten
    model.add(
        layers.Flatten()
    )

    # FC layers
    model.add(
        layers.Dense(
            4096,
            activation='relu'
        )
    )
    model.add(
        layers.Dense(
            4096,
            activation='relu'
        )
    )
    model.add(
        layers.Dense(
            num_classes,
            activation='softmax'
        )
    )

    model.summary()
    return model


class InceptionBlock(layers.Layer):
    """
    Implementation of the inception module in inception v1
    """
    def __init__(self, filters):
        super(InceptionBlock, self).__init__()
        self.filters = filters

    def call(self, inputs):
        conv1 = layers.Conv2D(
            self.filters,
            1,
            1,
            padding='same'
        )(inputs)
        conv2 = layers.Conv2D(
            self.filters,
            1,
            1,
            padding='same'
        )(inputs)
        maxpool1 = layers.MaxPooling2D(
            3,
            1,
            padding='same'
        )(inputs)
        conv3 = layers.Conv2D(
            self.filters,
            1,
            1,
            padding='same'
        )(inputs)
        conv11 = layers.Conv2D(
            self.filters,
            3,
            1,
            padding='same'
        )(conv1)
        conv22 = layers.Conv2D(
            self.filters,
            5,
            1,
            padding='same'
        )(conv2)
        maxconv = layers.Conv2D(
            self.filters,
            1,
            1,
            padding='same'
        )(maxpool1)

        return layers.Concatenate()([conv3, conv11, conv22, maxconv])


class InceptionBlockA(layers.Layer):
    """
    Implementation of the inception module A in inception v2
    """
    def __init__(self, filters):
        super(InceptionBlockA, self).__init__()
        self.filters = filters

    def call(self, inputs):
        conv1 = layers.Conv2D(
            self.filters,
            1,
            1,
            padding='same'
        )(inputs)
        conv2 = layers.Conv2D(
            self.filters,
            1,
            1,
            padding='same'
        )(inputs)
        maxpool1 = layers.MaxPooling2D(
            3,
            1,
            padding='same'
        )(inputs)
        conv3 = layers.Conv2D(
            self.filters,
            1,
            1,
            padding='same'
        )(inputs)
        conv11 = layers.Conv2D(
            self.filters,
            3,
            1,
            padding='same'
        )(conv1)
        conv22 = layers.Conv2D(
            self.filters,
            3,
            1,
            padding='same'
        )(conv2)
        conv22 = layers.Conv2D(
            self.filters,
            3,
            1,
            padding='same'
        )(conv22)
        maxconv = layers.Conv2D(
            self.filters,
            1,
            1,
            padding='same'
        )(maxpool1)

        return layers.Concatenate()([conv3, conv11, conv22, maxconv])


class InceptionBlockB(layers.Layer):
    """
    Implementation of the module B in inception v2
    """
    def __init__(self, filters):
        super(InceptionBlockB, self).__init__()
        self.filters = filters

    def call(self, inputs):
        conv1 = layers.Conv2D(
            self.filters,
            1,
            1,
            padding='same'
        )(inputs)
        conv2 = layers.Conv2D(
            self.filters,
            1,
            1,
            padding='same'
        )(inputs)
        maxpool1 = layers.MaxPooling2D(
            3,
            1,
            padding='same'
        )(inputs)
        conv3 = layers.Conv2D(
            self.filters,
            1,
            1,
            padding='same'
        )(inputs)
        conv11 = layers.Conv2D(
            self.filters,
            (1,7),
            1,
            padding='same'
        )(conv1)
        conv11 = layers.Conv2D(
            self.filters,
            (7,1),
            1,
            padding='same'
        )(conv11)
        conv11 = layers.Conv2D(
            self.filters,
            (1,7),
            1,
            padding='same'
        )(conv11)
        conv11 = layers.Conv2D(
            self.filters,
            (7,1),
            1,
            padding='same'
        )(conv11)

        conv22 = layers.Conv2D(
            self.filters,
            (1,7),
            1,
            padding='same'
        )(conv2)
        conv22 = layers.Conv2D(
            self.filters,
            (7,1),
            1,
            padding='same'
        )(conv22)
        maxconv = layers.Conv2D(
            self.filters,
            1,
            1,
            padding='same'
        )(maxpool1)

        return layers.Concatenate()([conv3, conv11, conv22, maxconv])


class InceptionBlockC(layers.Layer):
    """
    Implementation of the inception module C in inception v2
    """
    def __init__(self, filters):
        super(InceptionBlockC, self).__init__()
        self.filters = filters

    def call(self, inputs):
        conv1 = layers.Conv2D(
            self.filters,
            1,
            1,
            padding='same'
        )(inputs)
        conv2 = layers.Conv2D(
            self.filters,
            1,
            1,
            padding='same'
        )(inputs)
        maxpool1 = layers.MaxPooling2D(
            3,
            1,
            padding='same'
        )(inputs)
        conv3 = layers.Conv2D(
            self.filters,
            1,
            1,
            padding='same'
        )(inputs)
        conv11 = layers.Conv2D(
            self.filters,
            3,
            1,
            padding='same'
        )(conv1)
        conv111 = layers.Conv2D(
            self.filters//2,
            (1,3),
            1,
            padding='same'
        )(conv11)
        conv112 = layers.Conv2D(
            self.filters//2,
            (3,1),
            1,
            padding='same'
        )(conv11)

        conv21 = layers.Conv2D(
            self.filters//2,
            (1,3),
            1,
            padding='same'
        )(conv2)
        conv22 = layers.Conv2D(
            self.filters//2,
            (3,1),
            1,
            padding='same'
        )(conv2)
        maxconv = layers.Conv2D(
            self.filters,
            1,
            1,
            padding='same'
        )(maxpool1)

        return layers.Concatenate()([conv3, conv112, conv111, conv21, conv22, maxconv])


def auxiliary_classifier(input_tensor, v=3, num_classes=1000):
    """
    Creates the auxiliary classifier branch for inception models
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :param v: version 2 or 3 of inception model
    :type v: integer
    :param num_classes: number of classes to classify
    :type num_classes: integer
    :return: output of auxiliary classifier for inception
    :rtype: keras tensor
    """
    avg_pool = layers.AveragePooling2D(
        5,
        3,
    )(input_tensor)
    conv_block_1 = layers.Conv2D(
        128,
        1,
        1,
        padding='same'
    )(avg_pool)
    flattened = layers.Flatten()(conv_block_1)
    fc1 = layers.Dense(
        1024,
        activation='relu'
    )(flattened)
    if v==3:
        fc1 = layers.BatchNormalization()(fc1)
    fc1 = layers.Dropout(
        rate=0.7
    )(fc1)
    fc2 = layers.Dense(num_classes)(fc1)
    return fc2


def inception_v1(input_shape=(224,224,3), num_classes=1000):
    """
    Returns inception_v1 model (https://arxiv.org/pdf/1409.4842.pdf)
    :param input_shape:
    :type input_shape:
    :param num_classes:
    :type num_classes:
    :return: keras model of inception v1
    :rtype: keras model
    """
    input_layer = layers.Input(
        shape=input_shape
    )
    conv1 = layers.Conv2D(
        64,
        7,
        2,
        activation='relu',
        padding='same'
    )(input_layer)
    max_pool1 = layers.MaxPooling2D(
        3,
        2,
        padding='same'
    )(conv1)
    max_pool1 = LRN()(max_pool1)
    conv2 = layers.Conv2D(
        192,
        3,
        1,
        activation='relu',
        padding='same'
    )(max_pool1)
    max_pool2 = layers.MaxPooling2D(
        3,
        2,
        padding='same'
    )(conv2)
    max_pool2 = LRN()(max_pool2)

    inception3a = InceptionBlock(64)(max_pool2)
    inception3b = InceptionBlock(120)(inception3a)

    maxpool = layers.MaxPooling2D(
        3,
        2,
        padding='same'
    )(inception3b)

    inception4a = InceptionBlock(128)(maxpool)

    aux_1 = auxiliary_classifier(inception4a, v=1, num_classes=num_classes)

    inception4b = InceptionBlock(128)(inception4a)
    inception4c = InceptionBlock(128)(inception4b)
    inception4d = InceptionBlock(132)(inception4c)

    aux_2 = auxiliary_classifier(inception4d, v=1, num_classes=num_classes)

    inception4e = InceptionBlock(208)(inception4d)

    maxpool = layers.MaxPooling2D(
        3,
        2,
        padding='same'
    )(inception4e)

    inception5a = InceptionBlock(208)(maxpool)
    inception5b = InceptionBlock(256)(inception5a)

    avg_pool = layers.AveragePooling2D(
        7,
        1,
    )(inception5b)

    dropout = layers.Dropout(
        rate=0.4
    )(avg_pool)

    flat = layers.Flatten()(dropout)
    dense = layers.Dense(num_classes)(flat)
    softmax = layers.Activation('softmax')(dense)

    softmax_aux1 = layers.Activation('softmax')(aux_1)
    softmax_aux2 = layers.Activation('softmax')(aux_2)

    model = Model(
        inputs=input_layer,
        outputs=[softmax, softmax_aux1, softmax_aux2]
    )
    return model


class GridReducer(layers.Layer):
    """
    Implementation of the grid reduction for inception v2
    """
    def __init__(self, filters):
        super(GridReducer, self).__init__()
        self.filters = filters

    def __call__(self, inputs):
        maxpool = layers.MaxPooling2D(
            2,
            2,
            padding='valid'
        )(inputs)
        conv = layers.Conv2D(
            self.filters,
            2,
            2,
            padding='valid'
        )(inputs)

        return layers.Concatenate()([conv, maxpool])


def inception_v2_or_v3(input_shape=(299,299,3), num_classes=1000, v=3):
    """
    Returns the inception v2 or v3 model (https://arxiv.org/pdf/1512.00567v3.pdf) in keras.
    :param input_shape: shape if the input image
    :type input_shape: tuple of 3 integers
    :param num_classes: number of categories
    :type num_classes: integer
    :param v: indicate if version 2 and 3 is desired
    :type v: integer
    :return: keras model of inception v2
    :rtype: keras model
    """
    input_layer = layers.Input(
        shape=input_shape
    )
    conv1 = layers.Conv2D(
        32,
        3,
        2,
        activation='relu',
        padding='valid'
    )(input_layer)
    conv1 = layers.Conv2D(
        32,
        3,
        1,
        activation='relu',
        padding='same'
    )(conv1)
    conv1 = layers.Conv2D(
        64,
        3,
        1,
        activation='relu',
        padding='valid'
    )(conv1)
    max_pool1 = layers.MaxPooling2D(
        3,
        2,
        padding='valid'
    )(conv1)
    conv1 = layers.Conv2D(
        80,
        3,
        1,
        activation='relu',
        padding='valid'
    )(max_pool1)
    conv1 = layers.Conv2D(
        192,
        3,
        2,
        activation='relu',
        padding='valid'
    )(conv1)
    conv1 = layers.Conv2D(
        288,
        3,
        1,
        activation='relu',
        padding='same'
    )(conv1)

    conv1 = InceptionBlockA(96)(conv1)
    conv1 = InceptionBlockA(96)(conv1)
    conv1 = InceptionBlockA(96)(conv1)

    conv1 = GridReducer(384)(conv1)

    conv1 = InceptionBlockB(160)(conv1)

    aux_1 = auxiliary_classifier(conv1, v, num_classes)

    conv1 = InceptionBlockB(160)(conv1)
    conv1 = InceptionBlockB(160)(conv1)
    conv1 = InceptionBlockB(160)(conv1)
    conv1 = InceptionBlockB(160)(conv1)

    aux_2 = auxiliary_classifier(conv1, v, num_classes)

    conv1 = GridReducer(640)(conv1)

    conv1 = InceptionBlockC(512)(conv1)
    conv1 = InceptionBlockC(512)(conv1)

    max_pool1 = layers.MaxPooling2D(
        8,
        1,
        padding='valid'
    )(conv1)

    flat = layers.Flatten()(max_pool1)
    dense = layers.Dense(num_classes)(flat)
    softmax = layers.Activation('softmax', name='output_final')(dense)

    softmax_aux1 = layers.Activation('softmax',name='auxiliary_1')(aux_1)
    softmax_aux2 = layers.Activation('softmax',name='auxililary_2')(aux_2)

    model = Model(
        inputs=input_layer,
        outputs=[softmax, softmax_aux1, softmax_aux2]
    )
    return model


def resnetblock_v1(input_tensor, filters):
    """
    implements a v1 resnet block
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :param filters: filters to apply
    :type filters: integer
    :return: output of a resnet block
    :rtype: keras tensor
    """
    x = layers.Conv2D(
        filters,
        3,
        padding='same'
    )(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(
        filters,
        3,
        padding='same'
    )(x)
    x = layers.BatchNormalization()(x)
    x = x+input_tensor
    x = layers.Activation('relu')(x)
    return x


def resnetblock_v2(input_tensor, filters):
    """
    implements a v2 resnet block. Instead of zero padding for shortcut connection, I have made sure the filters at
    the end match the filters in the input tensor.
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :param filters: filters to apply
    :type filters: integer
    :return: output of a resnet block
    :rtype: keras tensor
    """
    x = layers.Conv2D(
        filters//4,
        1,
        padding='same'
    )(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(
        filters//4,
        3,
        padding='same'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(
        filters,
        1,
        padding='same'
    )(x)

    x = x+input_tensor
    x = layers.Activation('relu')(x)
    return x


def resnextblock(input_tensor, filters, cardinality=32):
    """
    implements a resnext block. Instead of zero padding for shortcut connection, I have made sure the filters at
    the end match the filters in the input tensor.
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :param filters: filters to apply
    :type filters: integer
    :param cardinality: number of paths in resnext block
    :type cardinality: integer
    :return: output of a resnext block
    :rtype: keras tensor
    """
    def conv_ops(input_tensor):
        x = layers.Conv2D(
            filters // 4,
            1,
            padding='same'
        )(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(
            filters // 4,
            3,
            padding='same'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(
            filters,
            1,
            padding='same'
        )(x)
        return x

    x = conv_ops(input_tensor)
    for i in range(cardinality-1):
        x += conv_ops(input_tensor)

    x = x+input_tensor
    x = layers.Activation('relu')(x)
    return x


def create_resnetBlocks(x, filter, n, resnetBlockFunc, halved=False):
    """
    creates resnet/resnext blocks n number of times
    :param x: input tensor
    :type x: keras tensor
    :param filter: filter size
    :type filter: integer
    :param n: how many blocks
    :type n: integer
    :param resnetBlockFunc: which version to use
    :type python function
    :param halved: reduced spatial dimension or not
    :type halved: boolean
    :return: keras tensor result
    :rtype: keras tensor
    """
    for i in range(n):
        x = resnetBlockFunc(x, filter)
    if halved:
        x = layers.Conv2D(
            filter*2,
            1,
            2,
            padding='same'
            )(x)
    return x


def create_resnextBlocks(x, filter, n, resnetBlockFunc, halved=False, cadinality=32):
    """
    creates resnet/resnext blocks n number of times
    :param x: input tensor
    :type x: keras tensor
    :param filter: filter size
    :type filter: integer
    :param n: how many blocks
    :type n: integer
    :param resnetBlockFunc: which version to use
    :type python function
    :param halved: reduced spatial dimension or not
    :type halved: boolean
    :param cadinality: cardinality of resnext block
    :type cadinality: integer
    :return: keras tensor result
    :rtype: keras tensor
    """
    for i in range(n):
        x = resnetBlockFunc(x, filter, cadinality)
    if halved:
        x = layers.Conv2D(
            filter*2,
            1,
            2,
            padding='same'
            )(x)
    return x


def resnet(input_shape=(224,224,3), num_classes=1000, version=1, num_layers = 34):
    """
    ResNet model based on https://arxiv.org/pdf/1512.03385v1.pdf
    :param input_shape: shape of the input image
    :type input_shape: tuple of 3 integers
    :param num_classes: number of categories
    :type num_classes: integer
    :param version: resnet v1 or v2
    :type version: integer
    :param num_layers: number of layers (one of [18, 34, 50, 101, 152])
    :type num_layers: integer
    :return: a resnet classification model
    :rtype: keras model
    """

    # use v1 or v2 resnet block?
    blockFunc = resnetblock_v1 if version == 1 else resnetblock_v2
    # print(blockFunc)

    # how many resnet blocks based on num_layers argument
    num_blocks_dict = {18: [2,2,2,2], 34: [3,4,6,3], 50: [3,4,6,3], 101: [3,4,23,3], 152: [3,8,36,3]}
    num_blocks = num_blocks_dict[num_layers]

    # num_filters
    filters_list = [64, 128, 256, 512]

    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(
        64,
        7,
        2,
        padding='same'
    )(inp)
    x = layers.MaxPooling2D(
        3,
        2,
        padding='same'
    )(x)

    # resnet blocks!
    halved = True
    for e, nb_filter in enumerate(zip(num_blocks, filters_list)):
        nb, k = nb_filter
        # if e+1 == len(num_blocks):
        #     halved = False
        x = create_resnetBlocks(x, k, nb, blockFunc, halved)

    # average pooling and softmax
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    # create model and return
    model = Model(inputs=inp, outputs=x)
    model.summary()
    return model


def resnext(input_shape=(224,224,3), num_classes=1000, num_layers=4, cardinality=32):
    """
    ResNet model based on https://arxiv.org/pdf/1611.05431.pdf
    :param input_shape: shape of the input image
    :type input_shape: tuple of 3 integers
    :param num_classes: number of categories
    :type num_classes: integer
    :param num_layers: number of layers (one of [18, 34, 50, 101, 152])
    :type num_layers: integer
    :param cardinality: cardinality of resnext block
    :type cardinality: integer
    :return: a resnet classification model
    :rtype: keras model
    """

    # how many resnet blocks based on num_layers argument
    num_blocks_dict = {18: [2,2,2,2], 34: [3,4,6,3], 50: [3,4,6,3], 101: [3,4,23,3], 152: [3,8,36,3]}
    num_blocks = num_blocks_dict[num_layers]

    # num_filters
    filters_list = [64, 128, 256, 512]

    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(
        64,
        7,
        2,
        padding='same'
    )(inp)
    x = layers.MaxPooling2D(
        3,
        2,
        padding='same'
    )(x)

    # resnext blocks!
    halved = True
    for e, nb_filter in enumerate(zip(num_blocks, filters_list)):
        nb, k = nb_filter
        # if e+1 == len(num_blocks):
        #     halved = True
        x = create_resnextBlocks(x, k, nb, resnextblock, halved, cardinality)

    # average pooling and softmax
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    # create model and return
    model = Model(inputs=inp, outputs=x)
    model.summary()
    return model