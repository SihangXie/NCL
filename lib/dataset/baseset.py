from torch.utils.data import Dataset
import torch
import json, os, random, time
import cv2
import torchvision.transforms as transforms
from data_transform.transform_wrapper import TRANSFORMS
import numpy as np
from utils.utils import get_category_list
import math
from PIL import Image


class BaseSet(Dataset):  # 继承了PyTorch的Dataset类来定制自己的数据集基础类
    def __init__(self, mode="train", cfg=None, transform=None):
        self.mode = mode  # 模式：训练
        self.transform = transform  # 数据预处理：没有
        self.cfg = cfg  # 配置信息
        self.input_size = cfg.INPUT_SIZE  # 获取配置信息中的图片输入尺寸：CIFAR为(32,32)
        self.color_space = cfg.COLOR_SPACE  # 获取配置信息中的图片色彩空间：'RGB'
        self.size = self.input_size  # 当前尺寸？

        print("Use {} Mode to train network".format(self.color_space))  # 打印信息：声明训练网络的色彩空间

        if self.mode == "train":  # 训练进
            print("Loading train data ...", end=" ")  # 打印信息：加载训练集中...，end入参是用空格代替换行
            self.json_path = cfg.DATASET.TRAIN_JSON  # 训练标签地址
        elif "valid" in self.mode:
            print("Loading valid data ...", end=" ")
            self.json_path = cfg.DATASET.VALID_JSON
        else:
            raise NotImplementedError
        self.update_transform()  # 调用方法：更新数据集预处理流程

        with open(self.json_path, "r") as f:    # 读取数据集标签JSON文件
            self.all_info = json.load(f)    # 标签数据{dict:2}存储在self.all_info属性中
        self.num_classes = self.all_info["num_classes"] # 获取标签信息中的类别数量

        self.data = self.all_info['annotations']    # 获取标签信息中的全部样本的标签列表

        print("Contain {} images of {} classes".format(len(self.data), self.num_classes))

        if self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and mode == "train":   # CIFAR不进
            self.class_weight, self.sum_weight = self.get_weight(self.data, self.num_classes)
            print('-' * 20 + ' dataset' + '-' * 20)
            print('class_weight is (the first 10 classes): ')
            print(self.class_weight[:10])

            num_list, cat_list = get_category_list(self.get_annotations(), self.num_classes, self.cfg)

            self.instance_p = np.array([num / sum(num_list) for num in num_list])
            self.class_p = np.array([1 / self.num_classes for _ in num_list])
            num_list = [math.sqrt(num) for num in num_list]

            self.square_p = np.array([pow(num, 0.5) / sum(pow(np.array(num_list), 0.5)) for num in num_list])

            self.class_dict = self._get_class_dict()

    def update(self, epoch):
        self.epoch = epoch
        if self.sample_type == "weighted_progressive":
            self.progress_p = epoch / self.cfg.TRAIN.MAX_EPOCH * self.class_p + (
                    1 - epoch / self.cfg.TRAIN.MAX_EPOCH) * self.instance_p
            # print('self.progress_p', self.progress_p)

    def __getitem__(self, index):
        print('start get item...')
        now_info = self.data[index]
        img = self._get_image(now_info)
        print('complete get img...')
        meta = dict()
        image = self.transform(img)
        image_label = (
            now_info["category_id"] if "test" not in self.mode else 0
        )  # 0-index
        if self.mode not in ["train", "valid"]:
            meta["image_id"] = now_info["image_id"]
            meta["fpath"] = now_info["fpath"]

        return image, image_label, meta

    def update_transform(self, input_size=None):
        normalize = TRANSFORMS["normalize"](cfg=self.cfg, input_size=input_size)    # 以注册表的方式实例化lib/data_transform/transform_wrapper.py中的归一化预处理器
        transform_list = [transforms.ToPILImage()]  # 预处理列表，存储一系列预处理操作，实例化PILImage转换器：把张量或numpy数组转换为PIL格式的图片
        transform_ops = (   # 图片预处理操作组合
            self.cfg.TRANSFORMS.TRAIN_TRANSFORMS    # 获取配置文件中数据集的预处理操作，返回的是预处理操作名称str的元组
            if self.mode == "train"
            else self.cfg.TRANSFORMS.TEST_TRANSFORMS
        )
        for tran in transform_ops:  # 遍历预处理操作元组，把每个预处理操作添加到transform_list列表中
            transform_list.append(TRANSFORMS[tran](cfg=self.cfg, input_size=input_size))    # 去lib/data_transform/transform_wrapper.py注册表中寻找同名预处理方法实例化
        transform_list.extend([transforms.ToTensor(), normalize])
        self.transform = transforms.Compose(transform_list) # 将预处理操作的列表用Compose封装起来，赋给属性self.transform

    def get_num_classes(self):
        return self.num_classes

    def get_annotations(self):
        return self.all_info['annotations']

    def __len__(self):
        return len(self.all_info['annotations'])

    def imread_with_retry(self, fpath):  # 读取图片，失败会重试10次
        retry_time = 10  # 设置重试次数：10次
        for k in range(retry_time):
            try:
                img = cv2.imread(fpath)  # 调用OpenCV的读取图片
                if img is None:  # 如果为空就重试
                    print("img is None, try to re-read img")
                    continue
                return img
            except Exception as e:
                if k == retry_time - 1:  # 如果重试次数用完，就会报错
                    assert False, "pillow open {} failed".format(fpath)
                time.sleep(0.1)

    def _get_image(self, now_info):  # 获取图片
        # fpath = os.path.join(now_info["fpath"]) # 现在的目录join图片的目录
        fpath = os.path.join('../', now_info["fpath"])  # Windows调试专用路径
        img = self.imread_with_retry(fpath)  # 调用上面的imread_with_retry()方法读图片
        if self.color_space == "RGB":  # 色彩空间是RGB进
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换色彩空间？
        return img

    def _get_trans_image(self, img_idx):
        now_info = self.data[img_idx]
        fpath = os.path.join(now_info["fpath"])
        img = self.imread_with_retry(fpath)
        if self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img)[None, :, :, :]

    def _get_class_dict(self):
        class_dict = dict() # 创建空类别字典，用于存储类别样本分布
        for i, anno in enumerate(self.data):    # 遍历每条JSON标签
            cat_id = (
                anno["category_id"] if "category_id" in anno else anno["image_label"]
            )   # 获取类别ID(category_id)
            if not cat_id in class_dict:    # 如果获取到的cat_id不在class_dict字典中
                class_dict[cat_id] = [] # 在字典中创建这个类别ID的列表，key为cat_id
            class_dict[cat_id].append(i)    # 当前图片ID存入到对应列表ID的列表中
        return class_dict

    def _get_class_dict_PMID2019(self):  # PMID2019数据集专用
        class_dict = dict() # 创建空类别字典，用于存储类别样本分布
        for i, anno in enumerate(self.data):    # 遍历每条JSON标签
            cat_id = (
                anno["category_id"] if "category_id" in anno else anno["image_label"]
            )   # 获取类别ID(category_id)
            img_id = (
                anno["image_id"] if "image_id" in anno else anno["image_id"]
            )
            if not cat_id in class_dict:    # 如果获取到的cat_id不在class_dict字典中
                class_dict[cat_id] = [] # 在字典中创建这个类别ID的列表，key为cat_id
            class_dict[cat_id].append(img_id)    # 当前图片ID存入到对应列表ID的列表中
        return class_dict

    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i if i != 0 else 0 for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight
