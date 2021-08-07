# coding: utf-8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import KaimingNormal
import pdb
import random
import json
import math
from functools import reduce
from .utils import Bernoulli

class LambdaLayer(nn.Layer):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        # 由于paddle的F.pad与pytorch不同，在NCDHW五维格式下才能padding到通道，
        # 故先对输入在axis=1处增一维，padding完再减少回去
        x = paddle.unsqueeze(x, axis=1)
        out = self.lambd(x)
        x = paddle.squeeze(x, axis=1)
        return out.squeeze(axis=1)


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', block_dpr=0.):
        super(BasicBlock, self).__init__()

        self.block_dpr = block_dpr
        # 定义伯努利分布采样对象，用于forward时采样，1-丢弃率=保留率
        self.b = Bernoulli(1-self.block_dpr)
        # 对参数kaimingNormal初始化
        weight_attr = ParamAttr(initializer=KaimingNormal())
        self.conv1 = nn.Conv2D(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias_attr=False, weight_attr=weight_attr)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=1, padding=1, bias_attr=False, weight_attr=weight_attr)
        self.bn2 = nn.BatchNorm2D(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, :, ::2, ::2],
                                                  [0, 0, 0, 0, (planes - x.shape[2])//2, (planes - x.shape[2])//2], "constant", 0, data_format='NCDHW'))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2D(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias_attr=False, weight_attr=weight_attr),
                     nn.BatchNorm2D(self.expansion * planes)
                )

    def forward(self, x):
        identity = self.shortcut(x)
        if self.training:
            # 训练时要随机丢弃block的这一路来加速，根据伯努利分布采样值决定丢不丢，1就保留，0就丢弃，只走skip connection.
            if self.b.sample().item()==1.:
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += identity
            else:
                out = identity
        else:
            # 评估时保留full depth，利用集成效果，且block的主支输出要乘以保留率再加上skip connection的输出才对
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out*(1-self.block_dpr)+identity
        out = F.relu(out)
        return out

# 这里的实现统一用的丢弃率，论文是保留率，是1-丢弃率
class ResNet(nn.Layer):
    def __init__(self, block, num_blocks, num_classes=10, use_drop_path=False, drop_path_rate=0.):
        super(ResNet, self).__init__()
        self.in_planes = 16
        assert drop_path_rate>=0. and drop_path_rate<=1., "Error, drop_path_rate must be  greater than or equal to 0 and equal to or less than 1."
        self.drop_path_rate = drop_path_rate
        self.use_drop_path = use_drop_path
        
        weight_attr = ParamAttr(initializer=KaimingNormal())
        self.conv1 = nn.Conv2D(3, 16, kernel_size=3, stride=1, padding=1, bias_attr=False, weight_attr=weight_attr)

        self.bn1 = nn.BatchNorm2D(16)

        # 得到blocks的总数
        self.blocks_num = reduce(lambda x, y: x+y, num_blocks, 0)
        print('totoal blocks num:', self.blocks_num)
        self.count_blocks = 0
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.linear = nn.Linear(64, num_classes, weight_attr=weight_attr)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            # dpr_l = dpr_L*l/L
            block_dpr = (self.count_blocks+1)*self.drop_path_rate/self.blocks_num
            print(f'{self.count_blocks}:{block_dpr}')
            layers.append(block(self.in_planes, planes, stride, block_dpr=block_dpr if self.use_drop_path else 0.))
            self.in_planes = planes * block.expansion

            self.count_blocks += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.shape[3])
        out = paddle.reshape(out, (out.shape[0], -1))
        out = self.linear(out)
        return out


def resnet110(checkpoint=None, drop_path_rate=0.5):
    model = ResNet(BasicBlock, [18, 18, 18], use_drop_path=True, drop_path_rate=drop_path_rate)
    if checkpoint is not None:
        model.set_state_dict(paddle.load(checkpoint))
    return model
