# -*- coding:utf-8 -*-
"""
用途：裁剪出检测框中的部分
作者：Sihang Xie
日期：2022年10月25日
"""
import os
import cv2
import json
import glob
import shutil
import numpy as np
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm


def parse_args():  # 构建可选参数

    parser = argparse.ArgumentParser(description="Convert xml labels to json label")  # 实例化参数解析器

    parser.add_argument(  # 添加输入路径参数
        "--input_path",
        help="the path of input original images",
        default="/home/og/XieSH/dataset/long-tailed/fuyou/JPEGImages/",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--xml_labels_path",
        help="the path of input xml labels files",
        default="/home/og/XieSH/dataset/long-tailed/fuyou/Annotations/",
        required=False,
        type=str,
    )

    parser.add_argument(  # 添加输出路径参数
        "--output_path",
        help="the path of output json labels file",
        default="/home/og/XieSH/dataset/long-tailed/PMID2019/images/",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--divided_ratio",
        help="the ratio to divide dataset into train set and valid set",
        default=0.8,
        required=False,
        type=int,
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
    # 获取当前目录下所有的原始图片名称列表
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


def crop_img(origin_img_root, xml_labels_root, output_img_root, scale):
    """根据xml检测标签裁剪出目标框中的图片"""
    # 调用上面的函数获取图片名称列表和标签名列表
    img_list, xml_list = read_and_decode(origin_img_root, xml_labels_root)
    # 裁剪遗漏图片专用
    # img_list = get_missing_img_list(origin_img_root, xml_labels_root, output_img_root)
    # img_list = ['4270', '1814', '10668']

    category_dict = {0: 'detonula pumila', 1: 'ceratium trichoceros', 2: 'ceratium carriense', 3: 'ditylum',
                     4: 'coscinodiscus flank', 5: 'corethron', 6: 'rhizosolenia', 7: 'protoperidinium',
                     8: 'ceratium furca', 9: 'ceratium tripos', 10: 'biddulphia', 11: 'eucampia',
                     12: 'guinardia flaccida', 13: 'helicotheca', 14: 'navicula', 15: 'pleurosigma pelagicum',
                     16: 'bacteriastrum', 17: 'thalassionema nitzschioides', 18: 'thalassionema frauenfeldii',
                     19: 'skeletonema', 20: 'ceratium fusus', 21: 'chaetoceros', 22: 'dinophysis caudata',
                     23: 'coscinodiscus'}

    category_list = list(category_dict.values())  # 类别字典值改成列表

    # 声明裁剪后图片文件名自增编号
    index: int = 0

    # 开始循环裁剪
    for img_n in tqdm(img_list):
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
                # 注意y轴是上0下height
                bbox_coordinate_dict = {'xmin': int(bbox.find('xmin').text), 'ymin': int(bbox.find('ymin').text),
                                        'xmax': int(bbox.find('xmax').text),
                                        'ymax': int(bbox.find('ymax').text)}  # 创建字典存放目标框坐标

                # 此处调用按比例放大后的区域的函数
                cropped_dict = get_cropping_area(img_n, width, height, bbox_coordinate_dict, 1.2)
                # 逐一取出裁剪坐标
                xmin = cropped_dict['xmin']
                xmax = cropped_dict['xmax']
                ymin = cropped_dict['ymin']
                ymax = cropped_dict['ymax']
                # 断言：裁剪边框坐标值必须都非负
                assert (
                        0 <= xmin <= width and 0 <= xmax <= width and 0 <= ymin <= height and 0 <= ymax <= height), f"{img_file} crop coordinate value overflow."

                if 0 <= xmin <= width and 0 <= xmax <= width and 0 <= ymin <= height and 0 <= ymax <= height:
                    label = obj.find('name').text  # 获取目标类别名称
                    # 此处有Bug：同一张图片中多个相同类别目标会因为文件名相同而被覆盖【已修复】
                    filename_sub_img = label + '_' + img_n + '_' + str(index) + '.jpg'  # 裁剪出来的小图的文件名称
                    save_path = os.path.join(output_img_root, str(category_list.index(label)))  # 裁剪好的每张图片按类别ID文件夹分开放
                    path_sub_img = os.path.join(save_path, filename_sub_img)  # 裁剪好的每张图片的绝对路径
                    cropped = img[ymin:ymax, xmin:xmax]  # 裁剪图片

                    if not os.path.exists(save_path):
                        os.makedirs(save_path)  # 如果保存目录不存在，则先创建
                    try:
                        cv2.imencode('.jpg', cropped)[1].tofile(path_sub_img)  # 编码成图片
                        index += 1  # 图片编号自增
                    except:
                        print('---encode error---: {}'.format(img_n))
                else:
                    print("--------error--------: {} cropped area out of the image shape".format(img_n))  # 超出图片范围发出警告
                    continue


def get_cropping_area(img_n, width, height, bbox_coordinate_dict, scale):
    """按照放大倍率裁剪目标框图片"""
    xmin = bbox_coordinate_dict['xmin']  # 先从字典中取出来
    xmax = bbox_coordinate_dict['xmax']
    ymin = bbox_coordinate_dict['ymin']
    ymax = bbox_coordinate_dict['ymax']
    sub_width = xmax - xmin  # 目标框的宽
    sub_height = ymax - ymin  # 目标框的高
    scaled_width = sub_width * scale  # 缩放后目标框的宽
    scaled_height = sub_height * scale  # 缩放后目标框的高
    x_diff = scaled_width - sub_width  # 缩放目标框和原始目标框的x轴差
    y_diff = scaled_height - sub_height  # 缩放目标框和原始目标框的y轴差

    # 如果目标框本身就占原图片的70%以上的话，就不用再扩大了
    if sub_width / width < 0.7 or sub_height / height < 0.7:
        if xmin < width - xmax:  # 说明目标框靠左，要向右裁剪多一点
            bbox_coordinate_dict['xmin'] = int(xmin - x_diff * 0.2)  # 左边框左移横向扩大部分的20%
            bbox_coordinate_dict['xmax'] = int(xmax + x_diff * (1 - 0.2))  # 右边框右移横向扩大部分的80%
            if bbox_coordinate_dict['xmin'] < 0:  # 如果左边框超出图片范围了
                # print("--------error--------: {} xmin out of the width".format(img_n))  # 发出警告
                delta_x = abs(x_diff - xmin)  # xmax要右移的长度
                bbox_coordinate_dict['xmin'] = 0  # xmin向左扩大裁剪范围到左尽头
                bbox_coordinate_dict['xmax'] = int(xmax + delta_x)  # 向右扩大裁剪范围
            if bbox_coordinate_dict['xmax'] > width:  # 如果右边框超出图片范围了
                bbox_coordinate_dict['xmax'] = width  # xmax向右扩大裁剪范围到右尽头
        else:  # 说明目标框靠右，要向左裁剪多一点
            bbox_coordinate_dict['xmax'] = int(xmax + x_diff * 0.2)  # 右边框右移横向扩大部分的20%
            bbox_coordinate_dict['xmin'] = int(xmin - x_diff * (1 - 0.2))  # 左边框左移横向扩大部分的80%
            if bbox_coordinate_dict['xmax'] > width:  # 如果右边框超出图片范围了
                # print("--------error--------: {} xmax out of the width".format(img_n))  # 发出警告
                delta_x = abs(x_diff - (width - xmax))  # xmin要左移的长度
                bbox_coordinate_dict['xmax'] = width  # xmax向右扩大裁剪范围到右尽头
                bbox_coordinate_dict['xmin'] = int(xmin - delta_x)  # 向左扩大裁剪范围
            if bbox_coordinate_dict['xmin'] < 0:  # 如果左边框超出图片范围了
                bbox_coordinate_dict['xmin'] = 0  # xmin向左扩大裁剪范围到左尽头

        if ymin > height - ymax:  # 说明目标框靠下，要向上裁剪
            bbox_coordinate_dict['ymax'] = int(ymax + y_diff * 0.2)  # 下边框下移纵向扩大部分的20%
            bbox_coordinate_dict['ymin'] = int(ymin - y_diff * (1 - 0.2))  # 上边框上移纵向扩大部分的80%
            if bbox_coordinate_dict['ymax'] > height:  # 如果下边框超出图片范围了
                # print("--------error--------: {} ymax out of the height".format(img_n))  # 发出警告
                delta_y = abs(y_diff - (height - ymax))  # ymin要上移的长度
                bbox_coordinate_dict['ymax'] = height  # ymax向下扩大裁剪范围到下尽头
                bbox_coordinate_dict['ymin'] = int(ymin - delta_y)  # ymin向上扩大裁剪范围
            if bbox_coordinate_dict['ymin'] < 0:  # 如果上边框超出图片范围了
                bbox_coordinate_dict['ymin'] = 0  # ymin向上扩大裁剪范围到上尽头
        else:  # 说明目标框靠上，要向下裁剪多一点
            bbox_coordinate_dict['ymin'] = int(ymin - y_diff * 0.2)  # 上边框上移纵向扩大部分的20%
            bbox_coordinate_dict['ymax'] = int(ymax + y_diff * (1 - 0.2))  # 下边框下移纵向扩大部分的80%
            if bbox_coordinate_dict['ymin'] < 0:  # 如果上边框超出图片范围了
                # print("--------error--------: {} ymin out of the height".format(img_n))  # 发出警告
                delta_y = abs(y_diff - ymin)  # ymin要上移的距离
                bbox_coordinate_dict['ymin'] = 0  # ymin向上扩大裁剪范围到上尽头
                bbox_coordinate_dict['ymax'] = int(ymax + delta_y)  # ymax向下扩大裁剪范围
            if bbox_coordinate_dict['ymax'] > height:  # 如果下边框超出图片范围了
                bbox_coordinate_dict['ymax'] = height  # ymax向下扩大裁剪范围到下尽头

    return bbox_coordinate_dict  # 直接返回四个坐标字典


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
    cropped_img_list = get_cropped_img_list(cropped_img_root)  # 已经裁剪好的图片编号列表
    origin_img_list, _ = read_and_decode(origin_img_root, xml_labels_root)  # 原始图片的图片编号列表

    missing_img_list = []  # 存放没被裁剪的图片编号

    for origin_img_id in origin_img_list:  # 遍历整个原始图片编号列表
        if origin_img_id in cropped_img_list:
            continue  # 如果原始图片编号在cropped_img_list中，则无事，继续循环
        else:
            missing_img_list.append(origin_img_id)  # 如果不在，就存入missing_img_list列表中

    print(f"The number of missing images: {len(missing_img_list)}")

    return missing_img_list


def copy2id_dir(origin_img_root, output_img_root):
    """把按类名存放的文件夹通过复制转换成按类别ID文件夹存放"""
    category_list = os.listdir(origin_img_root)  # 获取类别列表

    # category_dict = {}  # 类别字典，key是类别编号，value是类别名称
    # for category in category_list:  # 逐个读入类别字典
    #     category_dict[category_list.index(category)] = category

    category_dict = {0: 'detonula pumila', 1: 'ceratium trichoceros', 2: 'ceratium carriense', 3: 'ditylum',
                     4: 'coscinodiscus flank', 5: 'corethron', 6: 'rhizosolenia', 7: 'protoperidinium',
                     8: 'ceratium furca', 9: 'ceratium tripos', 10: 'biddulphia', 11: 'eucampia',
                     12: 'guinardia flaccida', 13: 'helicotheca', 14: 'navicula', 15: 'pleurosigma pelagicum',
                     16: 'bacteriastrum', 17: 'thalassionema nitzschioides', 18: 'thalassionema frauenfeldii',
                     19: 'skeletonema', 20: 'ceratium fusus', 21: 'chaetoceros', 22: 'dinophysis caudata',
                     23: 'coscinodiscus'}

    for category in category_list:
        category_dir = os.path.join(origin_img_root, category)  # 原始文件的每类的绝对路径
        target_dir = os.path.join(output_img_root, str(category_list.index(category)))  # 复制目标路径
        if os.path.exists(target_dir):
            os.removedirs(target_dir)  # 如果目标路径存在，就先删除它
        shutil.copytree(category_dir, target_dir)  # 复制


def divie_train_valid(input_img_root, output_img_root, divided_ratio):
    """按指定比例划分训练集和验证集"""
    category_list = os.listdir(input_img_root)  # 分类ID文件夹列表
    for category in category_list:  # 遍历类别ID文件夹中的每个文件夹
        category_path = os.path.join(input_img_root, category)  # 每个类别文件夹的绝对路径
        img_list = os.listdir(category_path)  # 获取每个类别文件夹中的所有图片文件名称列表
        num_img = len(img_list)  # 该类别文件夹下的图片总数
        num_train = int(divided_ratio * num_img)  # 要划分为训练集的数量
        num_valid = num_img - num_train  # 要划分成验证集的数量

        count = 0  # 计数器，监督训练集数量够了就换到验证集

        for img_file in img_list:
            img_name, suffix = os.path.splitext(img_file)  # 切割成图片名称和扩展名
            img_id = img_name.split('_', 1)[1]  # 按_切割的名称列表的第2个就是图片的ID

            if 0 <= count <= num_train:
                mode = 'train'
            else:
                mode = 'valid'

            source_path = os.path.join(category_path, img_file)  # 源文件路径
            save_path = os.path.join(output_img_root, category)  # 复制目标路径
            target_path = os.path.join(save_path, '{}_{}{}'.format(mode, img_id, suffix))  # 划分好的文件的绝对路径
            if not os.path.exists(save_path):  # 如果保存目标路径不存在，则先创建
                os.makedirs(save_path)
            # img = cv2.imread(source_path)
            # print(img)
            # cv2.imwrite(target_path, img)
            shutil.copy(source_path, target_path)  # 复制
            count += 1  # 自增


def get_annotations(img_root, json_file_prefix):
    """转换成json数据集标注"""
    train_annotations = []  # 创建空训练标签列表，用于存放每一张图片的标注
    valid_annotations = []  # 创建空验证标签列表，用于存放每一张图片的标注

    label_list = os.listdir(img_root)  # 标签列表
    num_class = len(label_list)  # 类别总数
    for label in label_list:
        category_path = os.path.join(img_root, label)  # 类别绝对路径
        img_list = os.listdir(category_path)  # 每个类的图片名称列表
        for img_file in img_list:
            save_path = os.path.join(category_path, img_file)  # 每个图片的绝对路径
            img_name, _ = os.path.splitext(img_file)  # 切割成图片名和扩展名
            img_id = int(img_name.split('_', 1)[1])  # 图片编号
            mode = img_name.split('_', 1)[0]  # train还是valid
            if mode == 'train':
                train_annotations.append(
                    {'fpath': save_path, 'image_id': img_id, 'category_id': int(label)})  # 把标签字典存入列表中
            else:
                valid_annotations.append(
                    {'fpath': save_path, 'image_id': img_id, 'category_id': int(label)})  # 把标签字典存入列表中

    # 生成训练集标签
    with open(os.path.join(img_root, '..', json_file_prefix + '_train.json'), 'w') as f:
        json.dump({'annotations': train_annotations, 'num_classes': num_class}, f)
    # 生成验证集标签
    with open(os.path.join(img_root, '..', json_file_prefix + '_valid.json'), 'w') as f:
        json.dump({'annotations': valid_annotations, 'num_classes': num_class}, f)


if __name__ == '__main__':
    args = parse_args()  # 实例化参数解析器
    read_and_decode(args.input_path, args.xml_labels_path)
    crop_img(args.input_path, args.xml_labels_path, args.output_path, 1.2)  # 裁剪图片
    cropped_img_list = get_cropped_img_list(args.output_path)  # 统计已裁剪图片
    missing_img_list = get_missing_img_list(args.input_path, args.xml_labels_path, args.output_path)
    copy2id_dir(args.input_path, args.output_path)
    divie_train_valid(args.input_path, args.output_path, args.divided_ratio)  # 划分训练集和测试集
    get_annotations(args.output_path, 'PMID2019')  # 获取标注文件
