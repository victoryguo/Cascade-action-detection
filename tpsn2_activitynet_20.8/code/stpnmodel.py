# @author: xiwu
# @license:
# @contact: fzy19931001@gmail.com
# @software: PyCharm
# @file: stpnmodel.py
# @time: 2019/6/3 21:15
# @desc:

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

NUM_SEGMENTS = 400
device = torch.device("cuda")


class TemporalProposal(nn.Module):
    def __init__(self, dataset='thumos14'):
        super(TemporalProposal, self).__init__()

        if dataset == 'thumos14':
            self.classes = 20
        elif dataset == 'activitynet1.2':
            self.classes = 100

        self.fc1 = nn.Linear(1024, 256)
        nn.init.normal_(self.fc1.weight, std=0.001)
        nn.init.constant_(self.fc1.bias, 0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        nn.init.normal_(self.fc2.weight, std=0.001)
        nn.init.constant_(self.fc2.bias, 0)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv1d(1024, self.classes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        nn.init.normal_(self.conv1.weight, std=0.001)
        nn.init.constant_(self.conv1.bias, 0)
        self.emb = nn.Conv1d(1024, 512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        nn.init.normal_(self.emb.weight, std=0.001)
        nn.init.constant_(self.emb.bias, 0)

        self.emb1 = nn.Conv1d(512, 512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        nn.init.normal_(self.emb1.weight, std=0.001)
        nn.init.constant_(self.emb1.bias, 0)

        self.emb2 = nn.Conv1d(512, 1024, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        nn.init.normal_(self.emb2.weight, std=0.001)
        nn.init.constant_(self.emb2.bias, 0)

        self.conv2 = nn.Conv1d(1024, self.classes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        nn.init.normal_(self.conv2.weight, std=0.001)
        # nn.init.constant_(self.conv2.bias, 0)
        self.fc3 = nn.Linear(1024, 256, bias=False)
        nn.init.normal_(self.fc3.weight, std=0.001)
        # nn.init.constant_(self.fc3.bias, 0)
        self.fc4 = nn.Linear(256, 1, bias=False)
        nn.init.normal_(self.fc4.weight, std=0.001)
        # nn.init.constant_(self.fc4.bias, 0)

        self.conv3 = nn.Conv1d(1024, self.classes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        nn.init.normal_(self.conv3.weight, std=0.001)
        # nn.init.constant_(self.conv2.bias, 0)
        self.fc5 = nn.Linear(1024, 256, bias=False)
        nn.init.normal_(self.fc5.weight, std=0.001)
        # nn.init.constant_(self.fc3.bias, 0)
        self.fc6 = nn.Linear(256, 1, bias=False)
        nn.init.normal_(self.fc6.weight, std=0.001)
        # nn.init.constant_(self.fc4.bias, 0)

    def forward(self, inp, labels, is_training=False):
        ori_inp = inp
        inp = inp.permute(1, 0)
        inp = inp.unsqueeze(0)
        inp = F.relu(self.emb(inp))
        inp = F.relu(self.emb1(inp))
        inp = F.relu(self.emb2(inp))
        inp = inp.squeeze(0)
        inp = inp.permute(1, 0)
        inp = inp + ori_inp
        x = inp
        ori_feature = inp
        inp = F.relu(self.fc1(inp))
        inp = F.sigmoid(self.fc2(inp))
        x = inp*x
        # x = x.view(-1, NUM_SEGMENTS, 1024)
        x = x.permute(1, 0)
        x = x.unsqueeze(0)
        x = self.conv1(x)
        x = x.squeeze(0)
        tcam = x.permute(1, 0)
        x = torch.sum(tcam, dim=0)
        x = F.sigmoid(x)

        if is_training:
            single = tcam[:, labels]
        else:
            preds = torch.gt(x, 0.1)
            preds = preds.cpu().data.numpy()
            labels = []
            flag = True
            for pred in range(self.classes):
                if preds[pred] == 1:
                    labels.append(pred)
                    flag = False
            if flag:
                npx = x.cpu().data.numpy()
                maxind = np.argmax(npx)
                labels.append(maxind)
            labels = torch.from_numpy(np.array(labels)).to(device)
            single = tcam[:, labels]
        single, _ = torch.max(single, dim=1)
        single = F.sigmoid(single)
        # single = single.cpu().data.numpy()
        pos = torch.ge(single, 0.53)
        mask = torch.ones(ori_feature.shape[0]).to(device)
        mask[pos] = 0.0
        mask = mask.unsqueeze(1)
        erased_feature = ori_feature * mask
        era_features = erased_feature

        erased_att = erased_feature
        erased_att = F.relu(self.fc3(erased_att))
        erased_att = F.sigmoid(self.fc4(erased_att))

        erased_feature = erased_att * erased_feature
        erased_feature = erased_feature.permute(1, 0)
        erased_feature = erased_feature.unsqueeze(0)
        erased_feature = self.conv2(erased_feature)
        erased_feature = erased_feature.squeeze(0)
        erased_tcam = erased_feature.permute(1, 0)
        erased_feature = torch.sum(erased_tcam, dim=0)
        erased_feature = F.sigmoid(erased_feature)

        if is_training:
            single2 = erased_tcam[:, labels]
        else:
            single2 = erased_tcam[:, labels]
        single2, _ = torch.max(single2, dim=1)
        single2 = F.sigmoid(single2)
        # single22 = single2.cpu().data.numpy()
        pos2 = torch.ge(single2, 0.51)
        mask2 = torch.ones(era_features.shape[0]).to(device)
        mask2[pos2] = 0.0
        mask2 = mask2.unsqueeze(1)

        erased2_feature = era_features * mask2
        erased2_att = erased2_feature
        erased2_att = F.relu(self.fc5(erased2_att))
        erased2_att = F.sigmoid(self.fc6(erased2_att))

        erased2_feature = erased2_att * erased2_feature
        erased2_feature = erased2_feature.permute(1, 0)
        erased2_feature = erased2_feature.unsqueeze(0)
        erased2_feature = self.conv3(erased2_feature)
        erased2_feature = erased2_feature.squeeze(0)
        erased2_tcam = erased2_feature.permute(1, 0)
        erased2_feature = torch.sum(erased2_tcam, dim=0)
        erased2_feature = F.sigmoid(erased2_feature)

        return x, inp, tcam, erased_feature, erased_tcam, erased_att, erased2_feature, erased2_tcam, erased2_att
