"""
This module contains all models and layers

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
    model = Sequential()

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
    model = Sequential()

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
        rate=0.5
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


def resnetblock_fewlayers_v2(input_tensor, filters):
    """
    implements a resnet v2 block for resnet model with 18 and 34 layers
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :param filters: filters to apply
    :type filters: integer
    :return: output of a resnet block
    :rtype: keras tensor
    """
    x = layers.BatchNormalization()(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(
        filters,
        3,
        padding='same'
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(
        filters,
        3,
        padding='same'
    )(x)
    x = x+input_tensor
    return x


def resnetblock_fewlayers(input_tensor, filters):
    """
    implements a resnet block for resnet model with 18 and 34 layers
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


def resnetblock_morelayers_v2(input_tensor, filters):
    """
    implements a resnet block v2 for resnet model with above 50 layers. Instead of zero padding for shortcut connection,
    I have made sure the filters at the end match the filters in the input tensor.
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :param filters: filters to apply
    :type filters: integer
    :return: output of a resnet block
    :rtype: keras tensor
    """
    x = layers.BatchNormalization()(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(
        filters//4,
        1,
        padding='same'
    )(x)
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
    return x


def resnetblock_morelayers(input_tensor, filters):
    """
    implements a resnet block for resnet model with above 50 layers. Instead of zero padding for shortcut connection,
    I have made sure the filters at the end match the filters in the input tensor.
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
    if version == 1:
        blockFunc = resnetblock_fewlayers if num_layers <= 34 else resnetblock_morelayers
    else:
        blockFunc = resnetblock_fewlayers_v2 if num_layers <= 34 else resnetblock_morelayers_v2
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
    print(blockFunc)
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


def inception_resnet_v1_a(input_tensor):
    """
    inception resnet a block v1
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :return: output of block a for inception resnet v1
    :rtype: keras tensor
    """
    x = layers.Activation('relu')(input_tensor)
    x1 = layers.Conv2D(
        32,
        1,
        1,
        padding='same'
    )(x)
    x2 = layers.Conv2D(
        32,
        1,
        1,
        padding='same'
    )(x)
    x2 = layers.Conv2D(
        32,
        3,
        1,
        padding='same'
    )(x2)
    x3 = layers.Conv2D(
        32,
        1,
        1,
        padding='same'
    )(x)
    x3 = layers.Conv2D(
        32,
        3,
        1,
        padding='same'
    )(x3)
    x3 = layers.Conv2D(
        32,
        3,
        1,
        padding='same'
    )(x3)
    concat = layers.Concatenate()([x1, x2, x3])
    conv_concat = layers.Conv2D(
        256,
        1,
        1,
        padding='same'
    )(concat)
    x = x+conv_concat
    x = layers.Activation('relu')(x)
    return x


def inception_resnet_v2_a(input_tensor):
    """
    inception resnet block a for v2
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :return: output of block a for v2
    :rtype: keras tensor
    """
    x = layers.Activation('relu')(input_tensor)
    x1 = layers.Conv2D(
        32,
        1,
        1,
        padding='same'
    )(x)
    x2 = layers.Conv2D(
        32,
        1,
        1,
        padding='same'
    )(x)
    x2 = layers.Conv2D(
        32,
        3,
        1,
        padding='same'
    )(x2)
    x3 = layers.Conv2D(
        32,
        1,
        1,
        padding='same'
    )(x)
    x3 = layers.Conv2D(
        48,
        3,
        1,
        padding='same'
    )(x3)
    x3 = layers.Conv2D(
        64,
        3,
        1,
        padding='same'
    )(x3)
    concat = layers.Concatenate()([x1,x2,x3])
    conv_concat = layers.Conv2D(
        384,
        1,
        1,
        padding='same'
    )(concat)
    x = x+conv_concat
    x = layers.Activation('relu')(x)

    return x


def inception_v4_a(input_tensor):
    """
    inception block a for v4
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :return: output of block a for v4
    :rtype: keras tensor
    """
    avgpool = layers.AveragePooling2D(
        padding='same',
        strides=1
    )(input_tensor)
    conv_pool = layers.Conv2D(
        96,
        1,
        1,
        padding='same'
    )(avgpool)
    conv1 = layers.Conv2D(
        96,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv2 = layers.Conv2D(
        64,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv2 = layers.Conv2D(
        96,
        3,
        1,
        padding='same'
    )(conv2)
    conv3 = layers.Conv2D(
        64,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv3 = layers.Conv2D(
        96,
        3,
        1,
        padding='same'
    )(conv3)
    conv3 = layers.Conv2D(
        96,
        3,
        1,
        padding='same'
    )(conv3)
    concat = layers.Concatenate()([conv_pool, conv1, conv2, conv3])

    return concat


def inception_resnet_a(input_tensor, version=1):
    """
    inception resnet a block for v1 and v2 and v4
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :param version: for inception resnet v1 or v2 or v4
    :type version: integer
    :return: keras tensor
    :rtype: keras tensor
    """
    if version == 1:
        return inception_resnet_v1_a(input_tensor)
    else:  # version == 2:
        return inception_resnet_v2_a(input_tensor)
    # else:  # inception v4
    #     return inception_v4_a(input_tensor)


def v1_stem(input_tensor):
    """
    stem for inception resnet v1
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :return: output tensor
    :rtype: keras tensor
    """
    x = layers.Conv2D(
        32,
        3,
        2,
        padding='valid'
    )(input_tensor)
    x = layers.Conv2D(
        32,
        3,
        1,
        padding='valid'
    )(x)
    x = layers.Conv2D(
        64,
        3,
        1,
        padding='same'
    )(x)
    x = layers.MaxPooling2D(
        3,
        2,
        padding='valid'
    )(x)
    x = layers.Conv2D(
        80,
        1,
        1,
        padding='same'
    )(x)
    x = layers.Conv2D(
        192,
        3,
        1,
        padding='valid'
    )(x)
    x = layers.Conv2D(
        256,
        3,
        2,
        padding='valid'
    )(x)
    return x


def v2_stem(input_tensor):
    """
    stem of inception resnet v2, also works for inception v4
    :param input_tensor: input tensor
    :type input_tensor: output tensor
    :return: keras tensor
    :rtype: keras.tensor
    """
    x = layers.Conv2D(
        32,
        3,
        2,
        padding='valid'
    )(input_tensor)
    x = layers.Conv2D(
        32,
        3,
        1,
        padding='valid'
    )(x)
    x = layers.Conv2D(
        64,
        3,
        1,
        padding='same'
    )(x)
    m = layers.MaxPooling2D(
        3,
        2,
        padding='valid'
    )(x)
    conv = layers.Conv2D(
        96,
        3,
        2,
        padding='valid'
    )(x)
    x = layers.Concatenate()([m, conv])
    x1 = layers.Conv2D(
        64,
        1,
        1,
        padding='same'
    )(x)
    x1 = layers.Conv2D(
        64,
        (7,1),
        1,
        padding='same'
    )(x1)
    x1 = layers.Conv2D(
        64,
        (1,7),
        1,
        padding='same'
    )(x1)
    x1 = layers.Conv2D(
        96,
        3,
        1,
        padding='valid'
    )(x1)
    x2 = layers.Conv2D(
        64,
        1,
        1,
        padding='same'
    )(x)
    x2 = layers.Conv2D(
        96,
        3,
        1,
        padding='valid'
    )(x2)
    x = layers.Concatenate()([x2, x1])
    conv = layers.Conv2D(
        192,
        3,
        2,
        padding='valid'
    )(x)
    m = layers.MaxPooling2D(
        strides=2,
        padding='valid'
    )(x)
    x = layers.Concatenate()([m, conv])
    return x


def stem(input_tensor, v=1):
    """
    stem of the inception resnet model
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :param v: version of inception resnet
    :type v: integer 1 or 2
    :return: output tensor
    :rtype: keras tensor
    """
    if v == 1:
        return v1_stem(input_tensor)
    else:
        return v2_stem(input_tensor)


def inception_resnet_reduction_a(input_tensor, version=1):
    """
    reduction a block for inception resnet
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :param version: version
    :type version: integer
    :return: output of reduction block
    :rtype: keras tensor
    """
    if version == 1:
        k, l, m, n = [192, 192, 256, 384]
    elif version == 2:
        k, l, m, n = [256, 256, 384, 384]
    else: # if inception v4
        k, l, m, n = [192, 224, 256, 384]
    maxpool = layers.MaxPooling2D(
        3,
        2,
        padding='valid'
    )(input_tensor)
    conv1 = layers.Conv2D(
        n,
        3,
        2,
        padding='valid'
    )(input_tensor)
    conv2 = layers.Conv2D(
        k,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv2 = layers.Conv2D(
        l,
        3,
        1,
        padding='same'
    )(conv2)
    conv2 = layers.Conv2D(
        m,
        3,
        2,
        padding='valid'
    )(conv2)
    x = layers.Concatenate()([maxpool, conv1, conv2])
    return x


def inception_resnet_v1_b(input_tensor):
    """
    b block for inception resnet v1
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :return: output of b block for inception resnet v1
    :rtype: keras tensor
    """
    x = layers.Activation('relu')(input_tensor)
    x1 = layers.Conv2D(
        128,
        1,
        1,
        padding='same'
    )(x)
    x2 = layers.Conv2D(
        128,
        1,
        1,
        padding='same'
    )(x)
    x2 = layers.Conv2D(
        128,
        (1,7),
        1,
        padding='same'
    )(x2)
    x2 = layers.Conv2D(
        128,
        (7,1),
        1,
        padding='same'
    )(x2)
    concat = layers.Concatenate()([x1,x2])
    conv_concat = layers.Conv2D(
        896,
        1,
        1,
        padding='same'
    )(concat)

    x = x+conv_concat
    x = layers.Activation('relu')(x)

    return x


def inception_resnet_v2_b(input_tensor):
    """
    b block of inception resnet v2
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :return: output of b block for inception resnet v2
    :rtype: keras tensor
    """
    x = layers.Activation('relu')(input_tensor)
    x1 = layers.Conv2D(
        192,
        1,
        1,
        padding='same'
    )(x)
    x2 = layers.Conv2D(
        128,
        1,
        1,
        padding='same'
    )(x)
    x2 = layers.Conv2D(
        160,
        (1,7),
        1,
        padding='same'
    )(x2)
    x2 = layers.Conv2D(
        192,
        (7,1),
        1,
        padding='same'
    )(x2)
    concat = layers.Concatenate()([x1,x2])
    conv_concat = layers.Conv2D(
        1152,
        1,
        1,
        padding='same'
    )(concat)
    x = x+conv_concat
    x = layers.Activation('relu')(x)

    return x


def inception_resnet_b(input_tensor, version=1):
    """
    inception resnet b block for inception resnet v1 and v2
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :param version: version
    :type version: integer
    :return: output of b block for inception resnet
    :rtype: keras tensor
    """
    if version == 1:
        return inception_resnet_v1_b(input_tensor)
    elif version == 2:
        return inception_resnet_v2_b(input_tensor)


def reduction_b_v1(input_tensor):
    """
    reduction b block for v1 of inception resnet
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :return: output of b block for v1 of inception resnet
    :rtype: keras tensor
    """
    maxpool = layers.MaxPooling2D(
        3,
        2,
        padding='valid'
    )(input_tensor)
    conv1 = layers.Conv2D(
        256,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv1 = layers.Conv2D(
        384,
        3,
        2,
        padding='valid'
    )(conv1)
    conv2 = layers.Conv2D(
        256,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv2 = layers.Conv2D(
        256,
        3,
        2,
        padding='valid'
    )(conv2)
    conv3 = layers.Conv2D(
        256,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv3 = layers.Conv2D(
        256,
        3,
        1,
        padding='same'
    )(conv3)
    conv3 = layers.Conv2D(
        256,
        3,
        2,
        padding='valid'
    )(conv3)
    concat = layers.Concatenate()([maxpool, conv1, conv2, conv3])

    return concat


def reduction_b_v2(input_tensor):
    """
    reduction b block for v2 of inception resnet
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :return: output of b block for v2 of inception resnet
    :rtype: keras tensor
    """
    maxpool = layers.MaxPooling2D(
        3,
        2,
        padding='valid'
    )(input_tensor)
    conv1 = layers.Conv2D(
        256,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv1 = layers.Conv2D(
        384,
        3,
        2,
        padding='valid'
    )(conv1)
    conv2 = layers.Conv2D(
        256,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv2 = layers.Conv2D(
        288,
        3,
        2,
        padding='valid'
    )(conv2)
    conv3 = layers.Conv2D(
        256,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv3 = layers.Conv2D(
        288,
        3,
        1,
        padding='same'
    )(conv3)
    conv3 = layers.Conv2D(
        320,
        3,
        2,
        padding='valid'
    )(conv3)
    concat = layers.Concatenate()([maxpool, conv1, conv2, conv3])

    return concat


def inception_resnet_reduction_b(input_tensor, version=1):
    """
    inception resnet reduction b block
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :param version: version
    :type version: integer
    :return: output of reduction b block
    :rtype: keras tensor
    """
    if version == 1:
        return reduction_b_v1(input_tensor)
    elif version == 2:
        return reduction_b_v2(input_tensor)


def inception_resnet_v1_c(input_tensor):
    """
    c block for inception resnet v1
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :return: output of c block for v1
    :rtype: keras tensor
    """
    x = layers.Activation('relu')(input_tensor)
    conv1 = layers.Conv2D(
        192,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv2 = layers.Conv2D(
        192,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv2 = layers.Conv2D(
        192,
        (1,3),
        1,
        padding='same'
    )(conv2)
    conv2 = layers.Conv2D(
        192,
        (3,1),
        1,
        padding='same'
    )(conv2)
    concat = layers.Concatenate()([conv1, conv2])
    conv_concat = layers.Conv2D(
        1792,
        1,
        1,
        padding='same'
    )(concat)
    x = x+conv_concat
    x = layers.Activation('relu')(x)

    return x


def inception_resnet_v2_c(input_tensor):
    """
    c block for inception resnet v2
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :return: output of c block for v2
    :rtype: keras tensor
    """
    x = layers.Activation('relu')(input_tensor)
    conv1 = layers.Conv2D(
        192,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv2 = layers.Conv2D(
        192,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv2 = layers.Conv2D(
        224,
        (1,3),
        1,
        padding='same'
    )(conv2)
    conv2 = layers.Conv2D(
        256,
        (3,1),
        1,
        padding='same'
    )(conv2)
    concat = layers.Concatenate()([conv1, conv2])
    conv_concat = layers.Conv2D(
        2144,
        1,
        1,
        padding='same'
    )(concat)
    x = x+conv_concat
    x = layers.Activation('relu')(x)

    return x


def inception_resnet_c(input_tensor, version=1):
    """
    inception resnet c block for inception resnet
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :param version: version
    :type version: integer
    :return: output of c block for inception resnet
    :rtype: keras tensor
    """
    if version == 1:
        return inception_resnet_v1_c(input_tensor)
    elif version == 2:
        return inception_resnet_v2_c(input_tensor)


def inception_resnet(input_shape=(299,299,3), num_classes=1000, version=1):
    """
    inception resnet models v1 and v2 based on https://arxiv.org/pdf/1602.07261.pdf
    :param input_shape: input shape
    :type input_shape: tuple of 3 integers
    :param num_classes: number of categories
    :type num_classes: integer
    :param version: version 1 or 2
    :type version: integer
    :return: inception resnet model
    :rtype: tf.keras.Model
    """
    input = layers.Input(shape=input_shape)
    x = stem(input, version)

    for i in range(5):
        x = inception_resnet_a(x, version)

    x = inception_resnet_reduction_a(x, version)

    for i in range(10):
        x = inception_resnet_b(x, version)

    x = inception_resnet_reduction_b(x, version)

    for i in range(5):
        x = inception_resnet_c(x, version)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input, outputs=x)
    model.summary()

    return model


def inception_v4_b(input_tensor):
    """
    b block for inception v4
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :return: output of block b
    :rtype: keras tensor
    """
    avgpool = layers.AveragePooling2D(
        strides=1,
        padding='same'
    )(input_tensor)
    conv_pool = layers.Conv2D(
        128,
        1,
        1,
        padding='same'
    )(avgpool)
    conv1 = layers.Conv2D(
        384,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv2 = layers.Conv2D(
        192,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv2 = layers.Conv2D(
        224,
        (1,7),
        1,
        padding='same'
    )(conv2)
    conv2 = layers.Conv2D(
        256,
        (1,7),
        1,
        padding='same'
    )(conv2)
    conv3 = layers.Conv2D(
        192,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv3 = layers.Conv2D(
        192,
        (1,7),
        1,
        padding='same'
    )(conv3)
    conv3 = layers.Conv2D(
        224,
        (7,1),
        1,
        padding='same'
    )(conv3)
    conv3 = layers.Conv2D(
        224,
        (1,7),
        1,
        padding='same'
    )(conv3)
    conv3 = layers.Conv2D(
        256,
        (7,1),
        1,
        padding='same'
    )(conv3)
    concat = layers.Concatenate()([conv_pool, conv1, conv2, conv3])

    return concat


def inception_v4_reduction_b(input_tensor):
    """
    reduction b for inception v4
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :return: output of reduction b for inception v4
    :rtype: keras tensor
    """
    maxpoool = layers.MaxPooling2D(
        3,
        2,
        padding='valid'
    )(input_tensor)
    conv1 = layers.Conv2D(
        192,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv1 = layers.Conv2D(
        192,
        3,
        2,
        padding='valid'
    )(conv1)
    conv2 = layers.Conv2D(
        256,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv2 = layers.Conv2D(
        256,
        (1,7),
        1,
        padding='same'
    )(conv2)
    conv2 = layers.Conv2D(
        320,
        (7,1),
        1,
        padding='same'
    )(conv2)
    conv2 = layers.Conv2D(
        320,
        3,
        2,
        padding='valid'
    )(conv2)
    x = layers.Concatenate()([maxpoool, conv1, conv2])

    return x


def inception_v4_c(input_tensor):
    """
    block c of inception v4
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :return: output of block c in inception v4
    :rtype: keras tensor
    """
    avgpool = layers.AveragePooling2D(
        padding='same',
        strides=1
    )(input_tensor)
    conv_pool = layers.Conv2D(
        256,
        1,
        1,
        padding='same'
    )(avgpool)
    conv1 = layers.Conv2D(
        256,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv2 = layers.Conv2D(
        384,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv21 = layers.Conv2D(
        256,
        (1,3),
        1,
        padding='same'
    )(conv2)
    conv22 = layers.Conv2D(
        256,
        (3,1),
        1,
        padding='same'
    )(conv2)
    conv3 = layers.Conv2D(
        384,
        1,
        1,
        padding='same'
    )(input_tensor)
    conv3 = layers.Conv2D(
        448,
        (1,3),
        1,
        padding='same'
    )(conv3)
    conv3 = layers.Conv2D(
        512,
        (3,1),
        1,
        padding='same'
    )(conv3)
    conv31 = layers.Conv2D(
        256,
        (3,1),
        1,
        padding='same'
    )(conv3)
    conv32 = layers.Conv2D(
        256,
        (1,3),
        1,
        padding='same'
    )(conv3)
    x = layers.Concatenate()([conv_pool, conv1, conv21, conv22, conv31, conv32])

    return x


def inception_v4(input_shape=(299,299,3), num_classes=1000):
    """
    inception v4 based on https://arxiv.org/pdf/1602.07261.pdf
    :param input_shape: input shape
    :type input_shape: tuple of 3 integers
    :param num_classes: number of categories
    :type num_classes: integer
    :return: inception v4 model
    :rtype: tf.keras.Model
    """
    version = 4
    input = layers.Input(shape=input_shape)
    x = v2_stem(input)  # stem for this and inception resnet v2 are similar

    for i in range(4):
        x = inception_v4_a(x)

    x = inception_resnet_reduction_a(x, version)

    for i in range(7):
        x = inception_v4_b(x)

    x = inception_v4_reduction_b(x)

    for i in range(3):
        x = inception_v4_c(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input, outputs=x)
    model.summary()

    return model


def sep_conv(input_tensor, filter):
    """
    separable convolution
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :param filter: number of filters
    :type filter: integer
    :return: output of separable convolution
    :rtype: keras tensor
    """
    x = layers.SeparableConv2D(
        filter,
        3,
        1,
    )(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x


def entry_block_1(input_tensor):
    """
    Block 1 of entry flow in xception net
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :return: output of block 1 entry flow
    :rtype: keras tensor
    """
    x = layers.Conv2D(
        32,
        3,
        2
    )(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(
        64,
        3,
        1
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x


def entry_sep_conv_block(input_tensor, filter):
    """
    Sep conv Block of entry flow in xception net
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :param filter: filters
    :type filter: integer
    :return: output of block sep_conv in entry flow
    :rtype: keras tensor
    """
    x = layers.SeparableConv2D(
        filter,
        3,
        1,
        padding='same'
    )(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(
        filter,
        3,
        1,
        padding='same'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(
        3,
        2,
        padding='same'
    )(x)

    return x


def entry_flow(input_tensor):
    """
    Entry flow of the xception network
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :return: output of entry flow
    :rtype: keras tensor
    """
    x = entry_block_1(input_tensor)
    sep_conv_e_1 = entry_sep_conv_block(x, 128)

    side_conv1 = layers.Conv2D(
        128,
        1,
        2,
    )(x)
    side_conv1 = layers.BatchNormalization()(side_conv1)

    x = side_conv1 + sep_conv_e_1

    x = layers.Activation('relu')(x)
    sep_conv_e_2 = entry_sep_conv_block(x, 256)
    side_conv2 = layers.Conv2D(
        256,
        1,
        2
    )(side_conv1)
    side_conv2 = layers.BatchNormalization()(side_conv2)

    x = side_conv2 + sep_conv_e_2

    x = layers.Activation('relu')(x)
    sep_conv_e_3 = entry_sep_conv_block(x, 728)
    side_conv3 = layers.Conv2D(
        728,
        1,
        2
    )(side_conv2)
    side_conv3 = layers.BatchNormalization()(side_conv3)

    return side_conv3+sep_conv_e_3


def mid_flow(input_tensor):
    """
    middle flow block of xception
    :param input_tensor: input_tensor
    :type input_tensor: keras tensor
    :return: output of middle flow of xception
    :rtype: keras tensor
    """
    x = layers.Activation('relu')(input_tensor)
    x = layers.SeparableConv2D(
        728,
        3,
        1,
        padding='same'
    )(x)
    x = layers.Activation('relu')(input_tensor)
    x = layers.SeparableConv2D(
        728,
        3,
        1,
        padding='same'
    )(x)
    x = layers.Activation('relu')(input_tensor)
    x = layers.SeparableConv2D(
        728,
        3,
        1,
        padding='same'
    )(x)

    return input_tensor + x


def exit_flow(input_tensor):
    """
    exit flow of xception
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :return: output of exit flow
    :rtype: keras tensor
    """
    x = layers.Activation('relu')(input_tensor)
    x = layers.SeparableConv2D(
        728,
        3,
        1,
        padding='same'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(
        1024,
        3,
        1,
        padding='same'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(
        3,
        2,
        padding='same'
    )(x)

    side_conv = layers.Conv2D(
        1024,
        1,
        2
    )(input_tensor)
    side_conv = layers.BatchNormalization()(side_conv)

    x = side_conv + x

    x = layers.SeparableConv2D(
        1536,
        3,
        1,
        padding='same'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.SeparableConv2D(
        2048,
        3,
        1,
        padding='same'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x


def xception_net(input_shape=(299,299,3), num_classes=1000):
    """
    Xception net based on https://arxiv.org/pdf/1610.02357.pdf
    :param input_shape: input shape
    :type input_shape: tuple of 3 integers
    :param num_classes: number of categories
    :type num_classes: integer
    :return:xception model
    :rtype: tf.keras.Model
    """
    input = layers.Input(shape=input_shape)

    # entry flow
    x = entry_flow(input)

    # middle flow, default to 8, some use 16
    for i in range(8):
        x = mid_flow(x)

    # exit flow
    x = exit_flow(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input, outputs=x)
    model.summary()

    return model


def mobile_dw_conv(input_tensor, filter, kernel, stride):
    """
    a depthwise separable convolution with given filter and kernel size and
    stride
    :param input_tensor: input tensor
    :type input_tensor: keras tensor
    :param filter: number of output filters
    :type filter: integer
    :param kernel: number of kernels
    :type kernel: integer
    :param stride: stride
    :type stride: integer
    :return: output of dws convolution
    :rtype: keras tensor
    """
    x = layers.DepthwiseConv2D(
        kernel,
        stride,
        'same',
        1
    )(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(
        filter,
        1,
        1,
        padding='same'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x


def mobile_net(input_shape=(224,224,3), num_classes=1000):
    """
    mobile net v1 based on https://arxiv.org/pdf/1704.04861.pdf
    :param input_shape: input shape
    :type input_shape: tuple of 3 integers
    :param num_classes: number of categories
    :type num_classes: integer
    :return:mobile net model
    :rtype: tf.keras.Model
    """
    input = layers.Input(shape=input_shape)

    x = layers.Conv2D(
        32,
        3,
        2,
        padding='same'
    )(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = mobile_dw_conv(x, 32, 3, 1)
    x = mobile_dw_conv(x, 64, 3, 2)
    x = mobile_dw_conv(x, 128, 3, 1)
    x = mobile_dw_conv(x, 128, 3, 2)
    x = mobile_dw_conv(x, 256, 3, 1)
    x = mobile_dw_conv(x, 256, 3, 2)

    for i in range(5):
        x = mobile_dw_conv(x, 512, 3, 1)

    x = mobile_dw_conv(x, 512, 3, 2)
    x = mobile_dw_conv(x, 1024, 3, 1)

    x = layers.AveragePooling2D(7, 1)(x)

    x = layers.Conv2D(
        num_classes,
        1,
        1,
        padding='same'
    )(x)

    x = layers.Reshape((num_classes,))(x)
    x = layers.Activation('softmax')(x)

    model = Model(inputs=input, outputs=x)
    model.summary()

    return model

