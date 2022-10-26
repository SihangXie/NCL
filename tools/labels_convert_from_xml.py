# -*- coding:utf-8 -*-
"""
用途：把多个xml检测标签合并成一个json分类标签
作者：Sihang Xie
日期：2022年10月25日
"""
import argparse

def parse_args():   # 构建可选参数

    parser = argparse.ArgumentParser(description="Convert xml labels to json label") # 实例化参数解析器

    parser.add_argument(    # 添加输入路径参数
        "--input_path",
        help="the path of input xml labels files",
        default="/home/og/XieSH/data/PMID2019/Annotations/",
        required=False,
        type=str,
    )

    parser.add_argument(    # 添加输输出路径参数
        "--output_path",
        help="the path of output json labels file",
        default="/home/og/XieSH/data/PMID2019/LongTailed/",
        required=False,
        type=str,
    )

    args = parser.parse_args()
    return args

def read_xml_labels(filename_queue):
    """从多个xml文件中读取"""
