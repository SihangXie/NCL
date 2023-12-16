"""
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from collections import OrderedDict


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 32+2-3+1=32尺寸不变{Tensor:(4,16,32,32)}
        out = self.bn2(self.conv2(out))  # 32+2-3+1=32尺寸不变{Tensor:(4,16,32,32)}
        out += self.shortcut(x)  # ResNet的融合输入x
        out = F.relu(out)
        return out


class StridedConv(nn.Module):
    """
    downsampling convolution layer
    """

    def __init__(self, in_planes, planes, use_relu=False) -> None:
        super(StridedConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.use_relu = use_relu
        if self.use_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)

        if self.use_relu:
            out = self.relu(out)

        return out


class ShallowExpert(nn.Module):
    """
    shallow features alignment
    """

    def __init__(self, input_dim=None, depth=None) -> None:
        super(ShallowExpert, self).__init__()
        self.convs = nn.Sequential(
            OrderedDict([(f'StridedConv{k}', StridedConv(
                in_planes=input_dim * (2 ** k), planes=input_dim * (2 ** (k + 1)),
                use_relu=(k != 1)
            )) for k in range(depth)])
        )

    def forward(self, x):
        out = self.convs(x)
        return out


class ResNet_Cifar(nn.Module):  # CIFAR的backbone
    def __init__(self, block, num_blocks, num_experts=None):
        super(ResNet_Cifar, self).__init__()
        self.in_planes = 16
        self.next_in_planes = 16
        self.num_experts = num_experts
        self.depth = list(reversed([i + 1 for i in range(len(num_blocks) - 1)]))  # [2, 1]
        self.exp_depth = [self.depth[i % len(self.depth)] for i in range(self.num_experts)]  # [2, 1, 2]
        feat_dim = 16
        self.shallow_exps = nn.ModuleList([ShallowExpert(
            input_dim=feat_dim * (2 ** (d % len(self.depth))), depth=d
        ) for d in self.exp_depth])  # 构建浅层特征对齐器

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        if num_experts:
            self.in_planes = 32
            self.layer3s = nn.ModuleList([self._make_layer(  # 每个专家专属的网络深度
                block, 64, num_blocks[2], stride=2) for _ in range(self.num_experts)])
            self.in_planes = self.next_in_planes
        else:
            self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        self.next_in_planes = self.in_planes
        for stride in strides:
            layers.append(block(self.next_in_planes, planes, stride))
            self.next_in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def load_model(self, pretrain):
        print("Loading Backbone pretrain model from {}......".format(pretrain))
        model_dict = self.state_dict()
        pretrain_dict = torch.load(pretrain)
        pretrain_dict = pretrain_dict["state_dict"] if "state_dict" in pretrain_dict else pretrain_dict
        from collections import OrderedDict

        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                k = k[7:]
            if "last_linear" not in k and "classifier" not in k and "linear" not in k and "fd" not in k:
                k = k.replace("backbone.", "")
                k = k.replace("fr", "layer3.4")
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("Backbone model has been loaded......")

    def forward(self, x, **kwargs):
        # TODO 此处加入网络深度提取融合模块SHIKE
        out = F.relu(self.bn1(self.conv1(x)))

        out1 = self.layer1(out)  # 第一个共享特征图
        # if 'layer' in kwargs and kwargs['layer'] == 'layer1':
        #     out1 = kwargs['coef'] * out1 + (1 - kwargs['coef']) * out1[kwargs['index']]

        out2 = self.layer2(out1)  # 第2个共享特征图
        # if 'layer' in kwargs and kwargs['layer'] == 'layer2':
        #     out2 = kwargs['coef'] * out2 + (1 - kwargs['coef']) * out2[kwargs['index']]

        shallow_outs = [out1, out2]
        if self.num_experts:
            out3s = [self.layer3s[_](out2) for _ in range(self.num_experts)]
            shallow_expe_outs = [self.shallow_exps[i](  # 浅层特征对齐
                shallow_outs[i % len(shallow_outs)]
            ) for i in range(self.num_experts)]

            exp_outs = [out3s[i] * shallow_expe_outs[i] for i in range(self.num_experts)]  # 对齐后的浅层特征与专家专属特征进行哈达玛积融合
            return exp_outs
        else:
            out3 = self.layer3(out2)

        # if 'layer' in kwargs and kwargs['layer'] == 'layer3':
        #     out = kwargs['coef'] * out + (1 - kwargs['coef']) * out[kwargs['index']]

        return out3


def res32_cifar(
        cfg,
        pretrain=True,
        pretrained_model="",
        last_layer_stride=2,
):
    resnet = ResNet_Cifar(BasicBlock, [5, 5, 5], num_experts=len(cfg.BACKBONE.MULTI_NETWORK_TYPE))
    if pretrain and pretrained_model != "":
        resnet.load_model(pretrain=pretrained_model)
    else:
        print("Choose to train from scratch")
    return resnet
