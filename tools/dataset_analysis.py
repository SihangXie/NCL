# -*- coding:utf-8 -*-
"""
用途：数据集分析工具，统计每类的样本个数
作者：Sihang Xie
日期：2022年10月18日
"""
import os
import json
import glob
import xml.etree.ElementTree as ET

from unicodedata import name
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


class Analyst:
    def __init__(self):
        # 存储类别ID的Numpy数组
        self.class_labels = None
        # 存储类别数量的字典
        self.class_number = None

    def count_Class_num_json(self, json_file_path):  # 入参是json文件的绝对路径
        # 初始化一个保存类别信息的空列表
        key_class_list = []
        # 打开json文件，返回的是字典对象
        json_file = json.load(open(json_file_path))
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
        d = dict(counter)
        # 对字典进行根据样本数量从大到小排序，返回的是列表
        dict_sorted = sorted(d.items(), key=lambda item: item[1], reverse=True)
        # 把列表转换为字典
        d = dict(dict_sorted)
        # 返回字典
        return d
        # # 获取每个类的类别ID
        # self.class_labels = list(counter)
        # # 将类别ID转换成Numpy类型
        # self.class_labels = np.array(self.class_labels)
        # # 获取每个类别的样本数
        # self.class_number = counter.values()

    def count_Class_num_xml(self, xml_dir_path):  # 入参是xml文件夹的路径
        # 提取xml文件列表
        # 把操作系统当前工作目录切换至xml_dir_path
        os.chdir(xml_dir_path)
        # 获取当前工作目录下的所有文件的名称列表list，列表中每个元素都是字符串类型
        annotations = os.listdir('.')
        # 返回所有文件名后缀是'.xml'的文件的名称list，目的是清除非xml的文件
        annotations = glob.glob(str(annotations) + '*.xml')

        # 新建字典，用于存放各类标签名及其对应的样本数量
        d = {}
        # 遍历xml文件名称列表
        for i, file in enumerate(annotations):
            # 逐一取出xml文件名称，打开为xml文件对象
            xml_file = open(file, encoding='utf-8')
            # 解析xml关系树，返回tree
            tree = ET.parse(xml_file)
            # 获取关系树根
            root = tree.getroot()

            # 遍历单个xml文件的所有标签
            for obj in root.iter('object'):
                # 获取<name>标签中的类别名称
                name = obj.find('name').text
                # 如果标签不是第一次出现，则加一
                if name in d.keys():
                    d[name] += 1
                # 如果标签是第一次出现，则将该标签名对应的value值初始化为1
                else:
                    d[name] = 1

        # 对字典进行根据样本数量从大到小排序，返回的是列表
        dict_sorted = sorted(d.items(), key=lambda item: item[1], reverse=True)
        # 把列表转换为字典
        d = dict(dict_sorted)

        # 打印统计结果
        print("The statistic result: \n")
        for key in d.keys():
            print(f"{key}: {str(d[key])}")

        return d

    def imgshow(self):
        # 对类别信息与类别数目进行可视化
        # 图片标题
        plt.title("")
        plt.xlabel("Class")
        plt.ylabel("Number")
        plt.bar(self.class_labels, self.class_number)
        plt.show()

    def imgshow_dict(self, class_num_dict):
        # 创建一个新图像
        plt.figure()
        # 用于正常显示中文标签
        plt.rcParams['font.sans-serif'] = ['SimHei']
        # 用于正常显示负号
        plt.rcParams['axes.unicode_minus'] = False

        # 定义类名为x轴刻度
        print(class_num_dict.keys())    # 返回的是元组<tuple>(['chaetoceros', ..., 'ceratium carriense'])
        # 类别名作为x轴，把元组转换成列表
        x_class = list(class_num_dict.keys())   # 返回的是列表<list>['chaetoceros', ..., 'ceratium carriense']
        # 确保x轴列表中的元素转换为str类型
        x_class = list(map(str, x_class))   # 返回的是<list>，其中元素都是str类型

        # 定义数量为y轴
        # 把字典value元组转换为列表类型
        y_num = list(class_num_dict.values())   # 返回的是列表<list>[2046, ..., 3]

        # 定义图
        # bar()绘制直方图
        plt.bar(x_class, y_num)
        # 设置x轴上刻度的旋转角度为50°
        # plt.xticks(rotation=60, fontsize=9)
        # 关闭x轴刻度
        plt.xticks([])
        # 设置x轴标签
        plt.xlabel("Category")
        # 设置y轴标签
        plt.ylabel("Sample Number")
        # 设置图的标题
        plt.title("Places_LT Dataset Distribution")

        # 设置显示紧凑型图片
        plt.tight_layout()
        # 显示设置好的图像
        plt.show()


if __name__ == '__main__':
    # 指定json标注文件路径
    # json_file_path = '/home/og/XieSH/data/CIFAR-LT/converted/cifar100_imbalance100/cifar100_imbalance100_train.json'
    json_file_path = '/home/og/XieSH/NCL/dataset_json/Places_LT_train.json'
    # xml_dir_path = '/home/og/XieSH/data/PMID2019/Annotations/'

    analyst = Analyst()
    class_num_dict = analyst.count_Class_num_json(json_file_path)
    analyst.imgshow_dict(class_num_dict)
