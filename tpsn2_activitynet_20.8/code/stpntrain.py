# @author: xiwu
# @license:
# @contact: fzy19931001@gmail.com
# @software: PyCharm
# @file: stpn-train.py
# @time: 2019/6/3 21:10
# @desc:

import utils
import os
import torch
from train import train
from val import val
from val_rgb import val_rgb
from val_flow import val_flow
from valf import valf
from stpnmodel import TemporalProposal as StpnModel
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import numpy as np
import random
import torch.nn.functional as F

device = torch.device("cuda")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(_):

    # seed
    setup_seed(2020)

    # Parsing Arguments
    args = utils.parse_args()
    mode = args.mode
    train_iter = args.training_num
    test_iter = args.test_iter
    ckpt = utils.ckpt_path(args.ckpt)
    input_list = {
        'dataset': args.dataset,
        'erased_branch_num': args.erased_branch_num,
        'batch_size': args.batch_size,
        'beta': args.beta,
        'learning_rate': args.learning_rate,
        'ckpt': ckpt,
        'class_threshold': args.class_th,
        'scale': args.scale}

    model = StpnModel(args.dataset)
    model = model.to(device)

    if mode == 'train':
        criterion = nn.BCELoss()
        criterion2 = nn.BCELoss()
        criterion3 = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0005)
        if args.train_stream == 'rgb':
            train("", model, input_list, 'rgb', train_iter, criterion, optimizer, criterion2, criterion3)
        elif args.train_stream == 'flow':
            train("", model, input_list, 'flow', train_iter, criterion, optimizer, criterion2, criterion3)

    elif mode == 'test':
        # ======== fusion ====================================================
        """
        flow 6600
        rgb 2400
        flow + rgb 20.78
        """
        if args.test_mode == 'fusion':
            model2 = StpnModel(args.dataset)
            model2 = model2.to(device)
            test_iter2 = args.test_iter2
            val("", model, "", input_list, test_iter, model2, test_iter2)  # Test
        elif args.test_mode == 'rgb':
            val_rgb("", model, "", input_list, test_iter)
        elif args.test_mode == 'flow':
            val_flow("", model, "", input_list, test_iter)


if __name__ == '__main__':
    main("")
