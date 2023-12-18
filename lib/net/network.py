import torch
import torch.nn as nn
from backbone import res32_cifar, res50, res152
from modules import GAP, FCNorm, Identity

import numpy as np
import cv2
import os
import copy
import math
from torch.nn.parameter import Parameter
from net.MOCO import MoCo


class Cos_Classifier(nn.Module):
    """ plain cosine classifier """

    def __init__(self, num_classes=10, in_dim=640, scale=16, bias=False):
        super(Cos_Classifier, self).__init__()
        self.scale = scale
        self.weight = Parameter(torch.Tensor(num_classes, in_dim).cuda())
        self.bias = Parameter(torch.Tensor(num_classes).cuda(), requires_grad=bias)
        self.init_weights()

    def init_weights(self):
        self.bias.data.fill_(0.)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, **kwargs):
        ex = x / torch.norm(x.clone(), 2, 1, keepdim=True)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        out = torch.mm(ex, self.scale * ew.t()) + self.bias
        return out


class multi_Network(nn.Module):  # 多网络模型，CIFAR进
    def __init__(self, cfg, mode="train", num_classes=1000, use_dropout=False):
        super(multi_Network, self).__init__()
        pretrain = (  # 是否预训练
            True
            if mode == "train"
               and cfg.BACKBONE.PRETRAINED_MODEL != ""
            else False
        )

        self.num_classes = num_classes  # 类别总数100
        self.cfg = cfg  # 配置文件参数
        self.network_num = len(self.cfg.BACKBONE.MULTI_NETWORK_TYPE)  # 采用3个ResNet32训练，分别对应3个专家
        self.use_dropout = use_dropout  # 是否使用dropout

        if pretrain:  # 预训练进

            self.backbone = nn.ModuleList(
                eval(self.cfg.BACKBONE.MULTI_NETWORK_TYPE[i])(
                    self.cfg,
                    last_layer_stride=2,
                    pretrained_model=cfg.BACKBONE.PRETRAINED_MODEL
                ) for i in range(self.network_num))
        else:  # 不预训练进

            self.backbone = nn.ModuleList(  # 设置BackBone网络
                eval(self.cfg.BACKBONE.MULTI_NETWORK_TYPE[i])(  # {list:3}['res32_cifar','res32_cifar','res32_cifar']
                    self.cfg,  # 入参1：配置信息
                    last_layer_stride=2,  # 入参2：最后一层的步进？设置为2
                ) for i in range(self.network_num))  # 循环3次创建3个backbone网络

        self.module = nn.ModuleList(  # 生成3个全局平均池化层的ModuleList
            self._get_module()
            for i in range(self.network_num))

        if self.use_dropout:  # 不使用dropout不进
            self.dropout = nn.ModuleList(
                nn.Dropout(p=0.5)
                for i in range(self.network_num))

        self.classifier = nn.ModuleList(  # 获取多分类器(全连接层)
            self._get_multi_classifer(cfg.CLASSIFIER.BIAS, cfg.CLASSIFIER.TYPE)  # 获取多分类器，传入是否有bias和分类器类型(全连接层)
            for i in range(self.network_num))  # 获取3次

    def forward(self, input, **kwargs):  # 前向传播：输入是三份相同的图片batch{list:3}

        if "feature_flag" in kwargs:  # 如果可变形参里有feature_flag这个参数
            return self.extract_feature(input, **kwargs)  # 调用下面的extract_feature()方法，返回图片的特征
        elif "classifier_flag" in kwargs:
            return self.get_logits(input, **kwargs)

        logits = []
        for i in self.network_num:
            x = (self.backbone[i])(input[i], **kwargs)
            x = (self.module[i])(x)
            x = x.view(x.shape[0], -1)
            self.feat.append(copy.deepcopy(x))
            if self.use_dropout:
                x = (self.dropout[i])(x)
            x = (self.classifier[i])(x)
            logits.append(x)

        return logits

    def extract_feature(self, input, **kwargs):  # 提取特征

        feature = []  # 构造空列表存放特征{list:0}
        for i in range(self.network_num):  # 有3个专家就循环3次
            x = (self.backbone[i])(input[i])  # 把一个batch的图片输入backbone网络中
            x = (self.module[i])(x)
            x = x.view(x.shape[0], -1)
            feature.append(x)

        return feature

    def get_logits(self, input, **kwargs):

        logits = []  # 创建空logits列表，用来存放logits
        for i in range(self.network_num):
            x = input[i]
            if self.use_dropout:  # 使用Dropout进
                x = (self.dropout[i])(x)
            x = (self.classifier[i])(x)  # 经过分类器，即Linear+softmax之后的向量就叫logits
            logits.append(x)

        return logits

    def extract_feature_maps(self, x):
        x = self.backbone(x)
        return x

    def freeze_multi_backbone(self):
        print("Freezing backbone .......")
        for p in self.backbone.parameters():
            p.requires_grad = False

    def load_backbone_model(self, backbone_path=""):
        self.backbone.load_model(backbone_path)
        print("Backbone model has been loaded...")

    def load_model(self, model_path, **kwargs):
        pretrain_dict = torch.load(
            model_path, map_location="cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if 'backbone_only' in kwargs.keys() and 'classifier' in k:
                continue;
            if k.startswith("module"):
                if k[7:] not in model_dict.keys():
                    print('not load:{}'.format(k))
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("All model has been loaded...")

    def get_feature_length(self):  # 获取特征长度
        if "cifar" in self.cfg.BACKBONE.TYPE:
            num_features = 64  # CIFAR的backbone怎么计算出有64个特征图的？
        elif 'res10' in self.cfg.BACKBONE.TYPE:
            num_features = 512
        else:
            num_features = 2048
        return num_features

    def _get_module(self):  # 获取全局平均池化或恒等模型
        module_type = self.cfg.MODULE.TYPE
        if module_type == "GAP":  # 全局平均池化
            module = GAP()
        elif module_type == "Identity":  # 恒等函数
            module = Identity()
        else:
            raise NotImplementedError

        return module

    def _get_multi_classifer(self, bias_flag, type):  # 获取分类器

        num_features = self.get_feature_length()  # 获取backbone特征总数，ResNet32特征图总数为64
        if type == "FCNorm":
            classifier = FCNorm(num_features, self.num_classes)
        elif type == "FC":  # 全连接层进
            classifier = nn.Linear(num_features, self.num_classes, bias=bias_flag)  # 全连接层入是[64,1]，输出形状是[100,1]
        elif type == 'cos':
            classifier = Cos_Classifier(self.num_classes, num_features, scale=self.cfg.CLASSIFIER.COS_SCALE,
                                        bias=bias_flag)
        else:
            raise NotImplementedError

        return classifier


class multi_Network_MOCO(nn.Module):
    def __init__(self, cfg, mode="train", num_classes=1000, use_dropout=False):
        super(multi_Network_MOCO, self).__init__()
        pretrain = (
            True
            if mode == "train"
               and cfg.BACKBONE.PRETRAINED_MODEL != ""
            else False
        )

        self.mlp_dim = cfg.NETWORK.MOCO_DIM
        self.num_classes = num_classes
        self.cfg = cfg
        self.network_num = len(self.cfg.BACKBONE.MULTI_NETWORK_TYPE)
        self.use_dropout = use_dropout

        if self.cfg.NETWORK.MOCO:  # 创建3个对比学习MOCO模块
            self.MOCO = nn.ModuleList(
                MoCo(dim=cfg.NETWORK.MOCO_DIM, K=cfg.NETWORK.MOCO_K, T=cfg.NETWORK.MOCO_T)
                for i in range(self.network_num))

        if pretrain:

            self.backbone = nn.ModuleList(
                eval(self.cfg.BACKBONE.MULTI_NETWORK_TYPE[i])(
                    self.cfg,
                    last_layer_stride=2,
                    pretrained_model=cfg.BACKBONE.PRETRAINED_MODEL
                ) for i in range(self.network_num))
        else:

            self.backbone = nn.ModuleList(
                eval(self.cfg.BACKBONE.MULTI_NETWORK_TYPE[i])(
                    self.cfg,
                    last_layer_stride=2,
                ) for i in range(self.network_num))

        self.module = nn.ModuleList(  # 创建全局平均池化层
            self._get_module()
            for i in range(self.network_num))

        if self.use_dropout:
            self.dropout = nn.ModuleList(
                nn.Dropout(p=0.5)
                for i in range(self.network_num))

        self.classifier = nn.ModuleList(  # 创建的分类器2层线性层夹一层ReLU
            self._get_multi_classifer(cfg.CLASSIFIER.BIAS, cfg.CLASSIFIER.SEMI_TYPE)
            for i in range(self.network_num))
        self.feat = []

        if pretrain:
            self.backbone_MA = nn.ModuleList(
                eval(self.cfg.BACKBONE.MULTI_NETWORK_TYPE[i])(
                    self.cfg,
                    last_layer_stride=2,
                    pretrained_model=cfg.BACKBONE.PRETRAINED_MODEL,
                ) for i in range(self.network_num))
        else:
            self.backbone_MA = nn.ModuleList(  # 用来自监督训练的滑动平均模型
                eval(self.cfg.BACKBONE.MULTI_NETWORK_TYPE[i])(
                    self.cfg,
                    last_layer_stride=2,
                ) for i in range(self.network_num))

        for i in range(self.network_num):
            for param in self.backbone_MA[i].parameters():  # 把自监督学习滑动平均模型的梯度流截断
                param.detach_()  # 截断梯度流，不会再计算往后的梯度

        self.module_MA = nn.ModuleList(  # 自监督学习滑动平均模型的GAP层
            self._get_module()
            for i in range(self.network_num))
        for i in range(self.network_num):
            for param in self.module_MA[i].parameters():
                param.detach_()  # 把自监督学习滑动平均模型的GAP层的梯度流截断

        if self.use_dropout:
            self.dropout_MA = nn.ModuleList(
                nn.Dropout(p=0.5)
                for i in range(self.network_num))
            for i in range(self.network_num):
                for param in self.dropout_MA[i].parameters():
                    param.detach_()

        self.classifier_MA = nn.ModuleList(  # 创建自监督学习滑动平均模型的分类器
            self._get_multi_classifer(cfg.CLASSIFIER.BIAS, cfg.CLASSIFIER.SEMI_TYPE)
            for i in range(self.network_num))
        for i in range(self.network_num):
            for param in self.classifier_MA[i].parameters():
                param.detach_()  # 把自监督学习滑动平均模型的分类器的梯度流截断
        self.feat_MA = []

        if cfg.CLASSIFIER.TYPE == 'FC':
            self.classifier_ce = nn.ModuleList(  # 交叉熵损失的分类器
                nn.Linear(self.get_feature_length(), self.num_classes, cfg.CLASSIFIER.BIAS)
                for i in range(self.network_num))
        elif cfg.CLASSIFIER.TYPE == 'cos':
            self.classifier_ce = nn.ModuleList(
                Cos_Classifier(self.num_classes, in_dim=self.get_feature_length(), scale=cfg.CLASSIFIER.COS_SCALE,
                               bias=True)
                for i in range(self.network_num))

    def forward(self, input, **kwargs):
        if "feature_flag" in kwargs:  # 特征提取
            return self.extract_feature(input, **kwargs)
        elif "classifier_flag" in kwargs:  # 分类器学习
            return self.get_logits(input, **kwargs)

        logits = []
        logits_ce = []
        for i in self.network_num:
            x = (self.backbone[i])(input[i], **kwargs)
            x = (self.module[i])(x)
            feature = x.view(x.shape[0], -1)
            self.feat.append(copy.deepcopy(feature))
            if self.use_dropout:
                feature = (self.dropout[i])(feature)

            output = (self.classifier[i])(feature)
            logits.append(output)

            output_ce = (self.classifier_ce[i])(feature)
            logits_ce.append(output_ce)

        logits_MA = []
        for i in self.network_num:
            x = (self.backbone_MA[i])(input[i], **kwargs)
            x = (self.module_MA[i])(x)
            x = x.view(x.shape[0], -1)
            self.feat_MA.append(copy.deepcopy(x))
            if self.use_dropout:
                x = (self.dropout_MA[i])(x)
            x = (self.classifier_MA[i])(x)
            logits_MA.append(x)

        return logits_ce, logits, logits_MA

    def extract_feature(self, input_all, **kwargs):
        input, input_MA = input_all  # input是由3个相同的(bs,3,32,32)输入构成的列表

        exp_outs = (self.backbone[0])(input[0], self.epoch, label=kwargs['label'][0], cfg=self.cfg)  # ResNet骨干网络，得到特征图
        feature = [(self.module[0])(output).view(output.size(0), -1) for output in exp_outs]  # 全局平均池化层，得到(bs, 64)

        exp_outs_MA = (self.backbone_MA[0])(input_MA[0], self.epoch, label=kwargs['label'][0], cfg=self.cfg)
        feature_MA = [(self.module_MA[0])(output).view(output.size(0), -1) for output in exp_outs_MA]
        return feature, feature_MA

    def get_logits(self, input_all, **kwargs):

        input, input_MA = input_all
        logits = []
        logits_ce = []
        for i in range(self.network_num):
            feature = input[i]
            if self.use_dropout:
                feature = (self.dropout[i])(feature)

            output = (self.classifier[i])(feature)  # 经过2个(64, 64)的线性层
            logits.append(output)

            output_ce = (self.classifier_ce[i])(feature)  # 经过1个(64, 100)的线性层，得到最终的分类得分
            logits_ce.append(output_ce)

        logits_MA = []
        for i in range(self.network_num):
            x = input_MA[i]
            if self.use_dropout:
                x = (self.dropout_MA[i])(x)
            x = (self.classifier_MA[i])(x)  # 自监督学习的分类器经过2个(64, 64)的线性层
            logits_MA.append(x)

        return logits_ce, logits, logits_MA  # 返回正常模型形状为(64, 100)、(64, 64)的logits和自监督模型形状为(64, 64)的logits

    def reset_epoch(self, epoch):
        self.epoch = epoch

    def extract_feature_maps(self, x):
        x = self.backbone(x)
        return x

    def freeze_multi_backbone(self):
        print("Freezing backbone .......")
        for p in self.backbone.parameters():
            p.requires_grad = False

    def load_backbone_model(self, backbone_path=""):
        self.backbone.load_model(backbone_path)
        print("Backbone model has been loaded...")

    def load_model(self, model_path, **kwargs):
        pretrain_dict = torch.load(
            model_path, map_location="cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if 'backbone_only' in kwargs.keys() and 'classifier' in k:
                continue;
            if k.startswith("module"):
                if k[7:] not in model_dict.keys():
                    print('not load:{}'.format(k))
                    continue
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("All model has been loaded...")

    def get_feature_length(self):
        if "cifar" in self.cfg.BACKBONE.TYPE:
            num_features = 64
        elif 'res10' in self.cfg.BACKBONE.TYPE:
            num_features = 512
        else:
            num_features = 2048
        return num_features

    def _get_module(self):
        module_type = self.cfg.MODULE.TYPE
        if module_type == "GAP":
            module = GAP()
        elif module_type == "Identity":
            module = Identity()
        else:
            raise NotImplementedError

        return module

    def _get_multi_classifer(self, bias_flag, type):

        num_features = self.get_feature_length()
        if type == "FCNorm":
            classifier = FCNorm(num_features, self.mlp_dim)
        elif type == "FC":
            classifier = nn.Linear(num_features, self.mlp_dim, bias=bias_flag)
        elif type == "mlp":
            classifier = nn.Sequential(nn.Linear(num_features, num_features, bias=bias_flag), \
                                       nn.ReLU(), \
                                       nn.Linear(num_features, self.mlp_dim, bias=bias_flag))
        elif type == 'cos':
            classifier = Cos_Classifier(self.mlp_dim, num_features, scale=self.cfg.CLASSIFIER.COS_SCALE, bias=bias_flag)
        else:
            raise NotImplementedError

        return classifier
