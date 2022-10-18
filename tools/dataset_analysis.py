# -*- coding:utf-8 -*-
"""
用途：数据集分析工具，统计每类的样本个数
作者：Sihang Xie
日期：2022年10月18日
"""
import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


class Analyst:
    def __init__(self, file_path):
        self.file_path = file_path
        # 存储类别ID的Numpy数组
        self.class_labels = None
        # 存储类别数量的字典
        self.number = None


    def count_Category_number(self):
        # 初始化一个保存类别信息的空列表
        key_class_list = []
        # 打开json文件，返回的是字典对象
        json_file = json.load(open(self.file_path))
        # 获取json文件中的元素"annotations"，返回的是list
        annotations = json_file['annotations']
        # annotations是list，长度为10847，其中每一个元素都是字典dict类型，代表一张图片
        for image in annotations:
            # 获取每张图片的类别id
            category_id = image['category_id']
            # 存入到类别列表中
            key_class_list.append(category_id)
        # 打印训练集大小
        print(f"Size of train set: {len(key_class_list)}")

        # Counter类：遍历列表，将元素出现的次数保存到一个字典中
        counter = Counter(key_class_list)
        print(counter)
        # 获取每个类的类别ID
        self.class_labels = list(counter)
        # 将类别ID转换成Numpy类型
        self.class_labels = np.array(self.class_labels)
        # 获取每个类别的样本数
        self.number = counter.values()

    def imgshow(self):
        # 对类别信息与类别数目进行可视化
        # 图片标题
        plt.title("")
        plt.xlabel("Class")
        plt.ylabel("Number")
        plt.bar(self.class_labels, self.number)
        plt.show()


if __name__ == '__main__':
    # 指定json标注文件路径
    file_path = '/home/og/XieSH/data/CIFAR-LT/converted/cifar100_imbalance100/cifar100_imbalance100_train.json'
    analyst = Analyst(file_path=file_path)
    analyst.count_Category_number()
    analyst.imgshow()
