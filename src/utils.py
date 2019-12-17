import pydensecrf.densecrf as dcrf
from pydensecrf import utils
import numpy as np

MAX_ITER = 10
POS_W = 3
POS_XY_STD = 1
Bi_W = 4
Bi_XY_STD = 67
Bi_RGB_STD = 3


def dense_crf(img, output_probs):
    """
    Conditional Random Field calculation. With the help of the following
    sources:
    1. https://github.com/zllrunning/deeplab-pytorch-crf/blob/master/libs/utils/crf.py
    2. https://github.com/lucasb-eyer/pydensecrf
    :param img: original input image
    :type img: input matrix
    :param output_probs: logits
    :type output_probs: keras tensor or np nd_array
    :return: results after crf
    :rtype: numpy array
    """
    w, h, c = output_probs

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    img = np.ascontiguousarray(img)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=img,
                           compat=Bi_W)
    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q

