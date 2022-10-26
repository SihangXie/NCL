# -*- coding:utf-8 -*-
"""
用途：裁剪出检测框中的部分
作者：Sihang Xie
日期：2022年10月25日
"""
import os
import cv2
import glob
import numpy as np
import argparse
import xml.etree.ElementTree as ET


def parse_args():  # 构建可选参数

    parser = argparse.ArgumentParser(description="Convert xml labels to json label")  # 实例化参数解析器

    parser.add_argument(  # 添加输入路径参数
        "--input_path",
        help="the path of input original images",
        default="/home/og/XieSH/data/PMID2019/JPEGImages/",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--xml_labels_path",
        help="the path of input xml labels files",
        default="/home/og/XieSH/data/PMID2019/Annotations/",
        required=False,
        type=str,
    )

    parser.add_argument(  # 添加输出路径参数
        "--output_path",
        help="the path of output json labels file",
        default="/home/og/XieSH/data/PMID2019/LongTailed/images/",
        required=False,
        type=str,
    )

    args = parser.parse_args()
    return args


def read_and_decode(origin_img_root, xml_labels_root):
    # 创建图片列表，用于存放原始图片
    img_list = []
    # 创建xml标签列表，用于存放检测标签
    xml_list = []

    # 把当前工作目录切换到原始图片目录
    os.chdir(origin_img_root)
    # 获取当前目录下所有的原始图片列表
    origin_images = os.listdir('.')
    # 返回所有文件名后缀是'.jpg'的文件的名称list，目的是清除非jpg的文件
    origin_images = glob.glob(str(origin_images) + '*.jpg')

    # 把当前工作目录切换到xml标签目录
    os.chdir(xml_labels_root)
    # 获取当前目录下所有的xml标签列表
    xml_labels = os.listdir('.')
    # 返回所有文件名后缀是'.xml'的文件的名称list，目的是清除非xml的文件
    xml_labels = glob.glob(str(xml_labels) + '*.xml')

    # 循环读取原始图片
    for image in origin_images:
        # 把图片文件名切分为名称和格式后缀.jpg
        name, suffix = os.path.splitext(image)
        # 只把图片名称存入图片列表
        img_list.append(name)

    # 循环读取xml标签
    for label in xml_labels:
        # 把xml标签文件名切分为名称和格式后缀.xml
        name, suffix = os.path.splitext(label)
        # 只把xml标签名称存入xml列表
        xml_list.append(name)

    return img_list, xml_list


def crop_img(origin_img_root, xml_labels_root, output_img_root):
    """根据xml检测标签裁剪出目标框中的图片"""
    # 调用上面的函数获取图片名称列表和标签名列表
    img_list, xml_list = read_and_decode(origin_img_root, xml_labels_root)
    # 裁剪遗漏图片专用
    # img_list = get_missing_img_list(origin_img_root, xml_labels_root, output_img_root)

    # 开始循环裁剪
    for img_n in img_list:
        if img_n in xml_list:  # 如果图片和标签名能够匹配才进
            img_file = img_n + '.jpg'  # 补充成完整的图片文件名
            img_path = os.path.join(origin_img_root, img_file)  # 补充成完整的原始图片绝对路径
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)  # 解码成NumPy图片
            height = img.shape[0]  # 获取图片高
            width = img.shape[1]  # 获取图片宽

            xml_file = img_n + '.xml'  # 补充成完整的xml文件名
            xml_path = os.path.join(xml_labels_root, xml_file)  # 补充成完整的xml标签文件的绝对路径
            tree = ET.parse(xml_path)  # 解析xml关系树，返回tree
            root = tree.getroot()  # 获取关系树根

            for obj in root.iter('object'):  # 遍历单个xml文件的所有<object>标签
                bbox = obj.find('bndbox')  # 获取每个obj的<bndbox>标签，即目标框标签
                xmin = int(bbox.find('xmin').text)  # 目标框左下角横坐标
                ymin = int(bbox.find('ymin').text)  # 目标框左下角纵坐标
                xmax = int(bbox.find('xmax').text)  # 目标框右上角横坐标
                ymax = int(bbox.find('ymax').text)  # 目标框右上角纵坐标

                if 0 <= xmin <= width and 0 <= xmax <= width and 0 <= ymin <= height and 0 <= ymax <= height:
                    label = obj.find('name').text  # 获取目标类别名称
                    filename_sub_img = label + '_' + img_n + '.jpg'  # 裁剪出来的小图的文件名称
                    save_path = os.path.join(output_img_root, label)  # 裁剪好的每张图片按类别名文件夹分开放
                    path_sub_img = os.path.join(save_path, filename_sub_img)  # 裁剪好的每张图片的绝对路径
                    cropped = img[ymin:ymax, xmin:xmax]  # 裁剪图片

                    if not os.path.exists(save_path):
                        os.makedirs(save_path)  # 如果保存目录不存在，则先创建
                    else:
                        try:
                            cv2.imencode('.jpg', cropped)[1].tofile(path_sub_img)  # 编码成图片
                        except:
                            print('---error---', img_n)
                else:
                    continue


def get_cropped_img_list(img_root):
    """获取所有已裁剪的图片的编号构成的列表"""
    os.chdir(img_root)  # 切换当前工作目录到图片根目录
    category_list = os.listdir('.')  # 获取当前目录下所有文件和文件目录，返回列表

    # img_num = 0 # 图片计数器
    cropped_img_list = []  # 存放裁剪好的图片

    for category in category_list:
        category_dir = os.path.join(img_root, category)  # 获取每个类别图片的根目录
        img_list = os.listdir(category_dir)  # 获取图片名称构成的列表

        for img_filename in img_list:
            name, suffix = os.path.splitext(img_filename)  # 裁剪出名称和扩展名
            name_list = name.split('_', 1)  # 根据'_'切割字符串1次成前后两部分，返回切割好的字符串列表
            img_id = name_list[1]  # 获取切割字符串列表的第2个，即图片编号
            cropped_img_list.append(img_id)  # 存入裁剪好的图片编号列表中

    print(f"The Number of cropped images: {len(cropped_img_list)}")

    return cropped_img_list


def get_missing_img_list(origin_img_root, xml_labels_root, cropped_img_root):
    """获取没有被裁剪的原始图片编号列表"""
    cropped_img_list = get_cropped_img_list(cropped_img_root)   # 已经裁剪好的图片编号列表
    origin_img_list, _ = read_and_decode(origin_img_root, xml_labels_root)   # 原始图片的图片编号列表

    missing_img_list = []   # 存放没被裁剪的图片编号

    for origin_img_id in origin_img_list:   # 遍历整个原始图片编号列表
        if origin_img_id in cropped_img_list:
            continue    # 如果原始图片编号在cropped_img_list中，则无事，继续循环
        else:
            missing_img_list.append(origin_img_id)  # 如果不在，就存入missing_img_list列表中

    print(f"The number of missing images: {len(missing_img_list)}")

    return missing_img_list


if __name__ == '__main__':
    args = parse_args()  # 实例化参数解析器
    # read_and_decode(args.input_path, args.xml_labels_path)
    crop_img(args.input_path, args.xml_labels_path, args.output_path)
    # cropped_img_list = get_cropped_img_list(args.output_path)
    # missing_img_list = get_missing_img_list(args.input_path, args.xml_labels_path, args.output_path)
