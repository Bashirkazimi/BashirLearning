"""
Tensorflow keras implementation of deeplab v2
https://github.com/Bashirkazimi/BashirLearning
Author: Bashir Kazimi
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


def conv_batch_relu(input, filters, kernel_size, strides, padding='same'):
    """
    Convolution, BatchNormalization, ReLU trio
    Args:
        input (keras tensor): input tensor
        filters (int): filter size
        kernel_size (int): kernel size
        strides (int): stride size
        padding (str): padding

    Returns: output of relu (keras tensor)

    """
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding
    )(input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x


def atrous_conv(input, filters, kernel_size, strides, rates, padding):
    """
    atrous convolution
    Args:
        input (keras tensor): input tensor
        filters (int): filter size
        kernel_size (int): kernel size
        strides (int): stride size
        rates (int): dilation rate
        padding (str): padding

    Returns: output of depthwise separable conv

    """
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation_rate=rates
    )(input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x


def atrous_conv_block(x, filters, kernel_size, strides, rates, padding):
    """
    atrous convolution block for deeplab v3
    Args:
        x (keras tensor): input tensor
        filters (list of int): filter sizes
        kernel_size (list of int): kernel sizes
        strides (list of int): stride sizes
        rates (list of int): dilation rates
        padding (list of str): padding

    Returns: keras tensor

    """
    for i in range(3):
        x = atrous_conv(
            x,
            filters[i],
            kernel_size[i],
            strides[i],
            rates[i],
            padding[i]
        )
    return x


def entry_flow(input, output_stride=32):
    """
    entry flow block in xception for deeplab v3
    Args:
        input (): keras input tensor
        output_stride (int): ration of resolution of input to output

    Returns: keras tensor

    """
    x = conv_batch_relu(
        input,
        filters=32,
        kernel_size=3,
        strides=2,
        padding='same'
    )
    x = conv_batch_relu(
        x,
        filters=64,
        kernel_size=3,
        strides=1,
        padding='same'
    )
    filters_lists = [
        [128, 128, 128],
        [256, 256, 256],
        [728, 728, 728]
    ]
    strides_list = [
        [1, 1, 2],
        [1, 1, 2],
        [1, 1, 2]
    ]
    rates_list = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    paddings = ['same', 'same', 'same']
    kernels = [3, 3, 3]

    res_strides = [2, 2, 2]

    res = conv_batch_relu(
        x,
        filters=128,
        kernel_size=1,
        strides=res_strides[0],
        padding='same'
    )

    x = atrous_conv_block(
        x,
        filters_lists[0],
        kernels,
        strides_list[0],
        rates_list[0],
        paddings
    )

    x = x + res

    res2 = conv_batch_relu(
        x,
        filters=256,
        kernel_size=1,
        strides=res_strides[1],
        padding='same'
    )

    x = atrous_conv_block(
        x,
        filters_lists[1],
        kernels,
        strides_list[1],
        rates_list[1],
        paddings
    )

    x = x + res2

    if output_stride == 8:
        strides_list = [
            [1, 1, 2],
            [1, 1, 2],
            [1, 1, 1]
        ]
        res_strides = [2, 2, 1]

    res3 = conv_batch_relu(
        x,
        filters=728,
        kernel_size=1,
        strides=res_strides[2],
        padding='same'
    )
    x = atrous_conv_block(
        x,
        filters_lists[2],
        kernels,
        strides_list[2],
        rates_list[2],
        paddings
    )

    x = x + res3

    return x, res


def middle_flow(x, output_stride=32):
    """
    middle flow for xception in deeplab v3
    Args:
        x (): input tensor
        output_stride (int): input resolution/output resolution ratio

    Returns: keras tensor

    """
    filters = [728]*3
    kernel_size = [3]*3
    strides = [1]*3
    rates = [1]*3
    padding = ['same']*3
    if output_stride == 8:
        rates = [2]*3

    for i in range(16):
        x = x + atrous_conv_block(x, filters, kernel_size, strides, rates, padding)
    return x


def exit_flow(x, output_stride=32):
    """
    exit flow for xception in deeplab v3
    Args:
        x (): input tensor
        output_stride (int): input resolution/output resolution ratio

    Returns: keras tensor

    """
    filters = [728, 1024, 1024]
    kernel_size = [3]*3
    strides = [1, 1, 2]
    rates = [1]*3
    padding = ['same']*3
    res_stride = 2

    if output_stride == 16:
        strides = [1]*3
        res_stride = 1

    if output_stride == 8:
        strides = [1]*3
        res_stride = 1
        rates = [2]*3

    res = conv_batch_relu(
        x,
        filters=1024,
        kernel_size=1,
        strides=res_stride,
        padding='same'
    )

    x = atrous_conv_block(x, filters, kernel_size, strides, rates, padding)
    x = x+res

    filters = [1536, 1536, 2048]
    kernel_size = [3]*3
    strides = [1]*3
    rates = [1]*3
    padding = ['same']*3

    if output_stride == 16:
        rates = [2]*3
    if output_stride == 8:
        rates = [4]*3

    x = atrous_conv_block(x, filters, kernel_size, strides, rates, padding)

    return x


def atrous_spatial_pyrmaid_pooling(x, output_stride):
    """
    Atrous spatial pyramid pooling as in xception for deeplab v3
    Args:
        x (keras tensor): input tensor to apply aspp to
        output_stride (int): resolution ratio

    Returns: keras tensor

    """
    if output_stride == 8:
        rates = [12, 24, 36]
    else:
        rates = [6, 12, 18]
    depth = 256
    aspp_1 = atrous_conv(x, depth, 1, 1, 1, 'same')
    aspp_2 = atrous_conv(x, depth, 3, 1, rates[0], 'same')
    aspp_3 = atrous_conv(x, depth, 3, 1, rates[1], 'same')
    aspp_4 = atrous_conv(x, depth, 3, 1, rates[2], 'same')

    xshape = x.get_shape().as_list()
    image_pooling = layers.GlobalAveragePooling2D()(x)
    image_pooling = layers.Reshape((1, 1, xshape[-1]))(image_pooling)
    image_pooling = conv_batch_relu(image_pooling, depth, 1, 1, 'same')
    image_pooling = tf.image.resize(
        image_pooling,
        size=xshape[1:3]
    )

    x = layers.Concatenate()([aspp_1, aspp_2, aspp_3, aspp_4, image_pooling])
    x = conv_batch_relu(x, depth, 1, 1, 'same')

    return x


def resnet_block(inp, filters, dilation_rate, strides, kernel_size):
    """
    ResNet block with atrous convolution
    Args:
        inp (): input tensor
        filters (int): filter size
        dilation_rate (int): dilation rate
        strides (int): stride
        kernel_size (int): kernel size

    Returns: keras tensor

    """
    x = inp

    for i in range(2):
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = atrous_conv(x, filters, kernel_size, strides, dilation_rate, 'same')

    x = x+inp

    return x


def stack_resnet_blocks(inp, filters, num_blocks, dilation_rate, strides,
                        kernel_size):
    """
    stacks num_blocks blocks of residual modules with atrous separable conv
    Args:
        inp (): input tensor
        filters (int): filter size
        num_blocks (int): number of blocks to stack
        dilation_rate (int): dilation rate
        strides (int): stride
        kernel_size (int): kernel size

    Returns: keras tensor

    """
    x = inp
    for i in range(num_blocks):
        x = resnet_block(x, filters, dilation_rate, strides, kernel_size)

    return x


def resnet_for_deeplab(input, output_stride):
    """
    Resnet with atrous separable convolution
    Args:
        input (keras tensor): input tensor
        output_stride (int): input to output image resolution ratio

    Returns: keras tensor

    """
    x = layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding='same'
    )(input)

    filters_list = [64, 128, 256, 512]
    num_blocks_list = [3, 4, 23, 3]
    rates_list = [1, 1, 1, 1]
    strides_list = [1, 1, 1, 1]
    kernels_list = [3, 3, 3, 3]
    sep_conv_stride = 2

    for i in range(4):
        if (output_stride == 8 and i > 1) or (output_stride == 16 and i > 2):
            sep_conv_stride = 1
            rates_list = [r*2 for r in rates_list]
        x = atrous_conv(x, filters_list[i], 3, sep_conv_stride, 1, 'same')
        x = stack_resnet_blocks(
            x,
            filters=filters_list[i],
            num_blocks=num_blocks_list[i],
            dilation_rate=rates_list[i],
            strides=strides_list[i],
            kernel_size=kernels_list[i]
        )
        if i == 0:
            low_level_features = x

    return low_level_features, x


def deeplab_v2(input_shape=(513,513,3), num_classes=21, output_stride=32,
                   backbone='Xception', aspp_type='LFOV'):
    """
    Deeplab v2 implementation based on https://arxiv.org/pdf/1606.00915.pdf
    Args:
        input_shape (tuple): input shape
        num_classes (int): number of categories
        output_stride (int): ratio of input image to output resolution
        backbone (str): backbone type, one of ['Xception', 'ResNet']
        aspp_type (str): type of aspp layer, one of ['LFOV', 'ASPP-S', 'ASPP-L']

    Returns: deeplab_v2 model (tf.keras.Model)

    """
    input = layers.Input(shape=input_shape)

    if backbone == 'Xception':
        entry_output, _ = entry_flow(input, output_stride)
        middle_flow_output = middle_flow(entry_output, output_stride)
        base_output = exit_flow(middle_flow_output, output_stride)

    else:  # 'ResNet'
        _, base_output = resnet_for_deeplab(
            input,
            output_stride
        )

    if aspp_type=="LFOV":
        x = atrous_conv(
            base_output,
            1024,
            3,
            1,
            12,
            'same'
        )
    elif aspp_type=='ASPP-S':
        xs = [atrous_conv(
            base_output,
            1024,
            3,
            1,
            dr,
            'same'
        ) for dr in [2, 4, 8, 12]
        ]
        x = layers.add(xs)
    else:  # aspp_type==ASPP-L
        xs = [atrous_conv(
            base_output,
            1024,
            3,
            1,
            dr,
            'same'
        ) for dr in [6, 12, 18, 24]
        ]
        x = layers.add(xs)

    x = layers.Conv2D(
        num_classes,
        1,
        1,
        padding='same'
    )(x)

    # Originally, instead of up sampling this, the ground truths are
    # down sampled during training, but we up sample the logits. Comment the
    # following line out if down sampling gts is what you want
    x = tf.image.resize(
        x,
        size=input_shape[:2]
    )
    x = layers.Activation('softmax')(x)

    model = Model(inputs=input, outputs=x)
    model.summary()

    return model






