# @author: xiwu
# @license:
# @contact: fzy19931001@gmail.com
# @software: PyCharm
# @file: nms_cpu.py
# @time: 2019/6/5 19:43
# @desc:

import numpy as np
import torch


def nms_cpu(dets, thresh):
    x1 = dets[:, 2]
    x2 = dets[:, 3]
    scores = dets[:, 1]

    length = (x2 - x1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order.item(0)
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1 + 1)
        ovr = inter / (length[i] + length[order[1:]] - inter)

        inds = np.where(ovr < thresh)[0]
        order = order[inds + 1]
    return keep
