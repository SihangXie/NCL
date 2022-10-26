import os
import cv2
import xml.etree.ElementTree as ET
import numpy

image_input = 'E:/detect_class/PMID2019/JPEGImages'
xml_input = 'E:/detect_class/PMID2019/Annotations'
path_output = "E:/detect_class/class"
class_names_path = 'E:/detect_class/class/classes.txt'

img_total = []
xml_total = []

file_image = os.listdir(image_input)
for filename in file_image:
    first, last = os.path.splitext(filename)
    img_total.append(first)


file_txt = os.listdir(xml_input)
for filename in file_txt:
    first, last = os.path.splitext(filename)
    xml_total.append(first)


for img_ in img_total:
    if img_ in xml_total:
        filename_img = img_ + ".jpg"
        path1=os.path.join(image_input,filename_img)
        # img = cv2.imread(path1)
        img = cv2.imdecode(numpy.fromfile(path1, dtype=numpy.uint8), -1)
        filename_xml=img_ + ".xml"
        path2 = os.path.join(xml_input,filename_xml)
        tree = ET.parse(path2)
        root = tree.getroot()
        width=img.shape[0]
        height=img.shape[1]

        # size = root.find('size')
        # width = float(size.find('width').text)
        # height = float(size.find('height').text)
        for obj in root.iter('object'):
            xml_box = obj.find('bndbox')
            xmin = (int(xml_box.find('xmin').text))
            ymin = (int(xml_box.find('ymin').text))
            xmax = (int(xml_box.find('xmax').text))
            ymax = (int(xml_box.find('ymax').text))
            if xmin>=0 and xmin<=width and xmax>=0 and xmax<=width and ymax>=0 and ymax<=height and ymin<=height and ymin>=0:
                # print(xmin, ymin, xmax, ymax)
                # captionList = obj.findall('name')  # find只能查找一个，findall可以查找所有的，class名可以换为其他子节点名
                lable = obj.find('name').text
                filename_last = lable + "_" + img_ + ".jpg"  # 裁剪出来的小图文件名
                save_path = os.path.join(path_output,lable)
                save_path1=os.path.join(save_path,filename_last)
                cropped = img[ymin:ymax, xmin:xmax]
                if os.path.exists(save_path):
                    # cv2.imwrite(save_path1, cropped)
                    try:
                        # cv2.imwrite(save_path1,cropped)
                        cv2.imencode('.jpg', cropped)[1].tofile(save_path1)
                    except:
                        print('---error---',img_)
                else:
                    os.makedirs(save_path)
                    try:
                        # cv2.imwrite(save_path1,cropped)
                        cv2.imencode('.jpg', cropped)[1].tofile(save_path1)
                    except:
                        print('---error---',img_)
            else:
                continue















