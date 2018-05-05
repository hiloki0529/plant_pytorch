# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=ksize, stride=1, padding=pad)
        self.bn_conv = nn.BatchNorm2d(out_channels)

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn_conv(h)
        return F.relu(h)

class ProposeCNN(nn.Module):
    """
    既存のモデルはAlexベース. これがVGGベース.
    """
    def __init__(self, fc=512, out=8):
        super(ProposeCNN, self).__init__()
        self.conv1_1 = Block(3, 24)
        self.conv2_1 = Block(24, 48)
        self.conv3_1 = Block(48, 96)
        self.conv3_2 = Block(96, 96)
        self.conv4_1 = Block(96, 96*2)
        self.conv4_2 = Block(96*2, 96*2)
        self.conv5_1 = Block(96*2, 96*2)
        self.conv5_2 = Block(96*2, 96*2)
        self.fc1 = nn.Linear(96*2*7*7, fc)
        self.bn_fc1 = nn.BatchNorm1d(fc)
        self.fc2 = nn.Linear(fc, fc)
        self.bn_fc2 = nn.BatchNorm1d(fc)
        self.fc3 = nn.Linear(fc, out)

    def __call__(self, x):
        h = self.conv1_1(x)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = self.conv2_1(h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = self.conv4_1(h)
        h = self.conv4_2(h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = self.conv5_1(h)
        h = self.conv5_2(h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = h.view(-1, 96*2*7*7)

        h = self.fc1(h)
        h = self.bn_fc1(h)
        h = F.relu(h)

        h = self.fc2(h)
        h = self.bn_fc2(h)
        h = F.relu(h)

        h = self.fc3(h)

        return F.log_softmax(h, dim=1)

    def forward(self, X, **args):
        X = self(X)
        return X
