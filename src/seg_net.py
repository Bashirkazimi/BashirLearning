"""
Tensorflow keras implementation of SegNet
https://github.com/Bashirkazimi/BashirLearning
Author: Bashir Kazimi
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


class CustomMaxpool(layers.Layer):
    """
    Implementation of the maxpooling with indices, returns maxpooled tensor,
    indices of max elements used in max pool, and shape of the original
    tensor that maxpooling is applied to.
    """
    def __init__(self):
        super(CustomMaxpool, self).__init__()

    def call(self, inputs):
        x, indices = tf.nn.max_pool_with_argmax(
            inputs,
            2,
            2,
            'SAME'
        )

        return x, indices


class CustomUnPool(layers.Layer):
    """
    Inverse of maxpooling with indices. Unpools the given tensor using the
    indices of the original tensor of shape original_shape from which it was
    maxpooled. implemented with the help of both of the following sources:
    1. https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/utils
    /builder.py
    2. https://github.com/sangeet259/tensorflow_unpooling/blob/master/unpool.py
    """
    def __init__(self):
        super(CustomUnPool, self).__init__()

    def call(self, inps):
        """
        max unpooling
        Args:
            inps (list): list of maxpooled tensor, argmax indices (second
            output of max_pool_with_argmax in tf), and ksize (the same as
            used for maxpooling)

        Returns: unpooled tensor

        """
        # get pooled tensor, indices and ksize from the argument
        pool, ind, k_size = inps

        # get input shape as list
        input_shape = tf.shape(pool)
        input_shape_aslist = pool.get_shape().as_list()

        # Determine the output shape
        output_shapeaslist = [-1, input_shape_aslist[1] * k_size[1],
                              input_shape_aslist[2] * k_size[2],
                              input_shape_aslist[3]]
        # flatten (reshape) the pooled tensor of rank 4 to that of rank 1
        # because the indices in argmax (ind) are flattened. so a maximum
        # value at position [b, y, x, c] becomes flattened index ((
        # b*height+y)) * width+x) * channels+c
        pool_ = tf.reshape(pool, [
            input_shape_aslist[1] * input_shape_aslist[2] *
            input_shape_aslist[3]])

        # Create a single unit extended cuboid of length bath_size populating
        # it with continous natural number from zero to batch_size
        batch_range = tf.reshape(
            tf.range(tf.cast(input_shape[0], tf.int64), dtype=ind.dtype),
            shape=tf.stack([input_shape[0], 1, 1, 1]))
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, tf.stack([input_shape_aslist[1] *
                                    input_shape_aslist[2] *
                                    input_shape_aslist[3], 1]))
        ind_ = tf.reshape(ind, tf.stack([input_shape_aslist[1] *
                                         input_shape_aslist[2] *
                                         input_shape_aslist[3], 1]))
        ind_ = tf.concat([b, ind_], 1)
        # Update the sparse matrix with the pooled values , it is a batch
        # wise operation
        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast([input_shape[0],
                                                        output_shapeaslist[
                                                            1] *
                                                        output_shapeaslist[
                                                            2] *
                                                        output_shapeaslist[
                                                            3]], tf.int64))
        # Reshape the vector to get the final result
        ret = tf.reshape(ret,
                         [-1, output_shapeaslist[1], output_shapeaslist[2],
                          output_shapeaslist[3]])
        return ret


def conv_batch_relu(x, filter_size, kernel_size):
    """
    Convolution, batch normalization, and relu trio
    Args:
        x (keras tensor): input tensor
        filter_size (int): filter size
        kernel_size (int): kernel size

    Returns:

    """
    """
    applies a convolution, batchnormalization and relu activation to x
    :param x: inpute tensor
    :type x: keras tensor
    :param filter_size: filter size
    :type filter_size: integer
    :param kernel_size: kernel size
    :type kernel_size: integer
    :return: output tensor
    :rtype: keras tensor
    """
    x = layers.Conv2D(
        filter_size,
        kernel_size,
        padding='same'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def segnet(input_shape=(224,224,3), num_classes=21):
    """
    SegNet model based on https://arxiv.org/pdf/1511.00561.pdf
    Args:
        input_shape (tuple): input shape
        num_classes (int): number of categories

    Returns:

    """

    inputs = layers.Input(shape=input_shape)

    pools = []

    x = conv_batch_relu(inputs, 64, 3)
    x = conv_batch_relu(x, 64, 3)

    x, mask = CustomMaxpool()(x)
    pools.append(mask)

    for i in range(2):
        x = conv_batch_relu(x, 128, 3)

    x, mask = CustomMaxpool()(x)
    pools.append(mask)

    for i in range(3):
        x = conv_batch_relu(x, 256, 3)

    x, mask = CustomMaxpool()(x)
    pools.append(mask)

    for i in range(3):
        x = conv_batch_relu(x, 512, 3)

    x, mask = CustomMaxpool()(x)
    pools.append(mask)

    for i in range(3):
        x = conv_batch_relu(x, 512, 3)

    x, mask = CustomMaxpool()(x)
    pools.append(mask)

    k_size = [1,2,2,1]
    x = CustomUnPool()([x, pools[-1], k_size])

    for i in range(3):
        x = conv_batch_relu(x, 512, 3)

    x = CustomUnPool()([x, pools[-2], k_size])

    for i in range(2):
        x = conv_batch_relu(x, 512, 3)
    x = conv_batch_relu(x, 256, 3)

    x = CustomUnPool()([x, pools[-3], k_size])

    for i in range(2):
        x = conv_batch_relu(x, 256, 3)
    x = conv_batch_relu(x, 128, 3)

    x = CustomUnPool()([x, pools[-4], k_size])

    x = conv_batch_relu(x, 128, 3)
    x = conv_batch_relu(x, 64, 3)

    x = CustomUnPool()([x, pools[-5], k_size])

    x = conv_batch_relu(x, 64, 3)

    x = layers.Conv2D(
        num_classes,
        1,
        padding='valid'
    )(x)

    x = layers.Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.summary()

    return model
