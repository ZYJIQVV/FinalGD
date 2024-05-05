# -*- encoding: utf-8 -*-
"""
@Time: 2024-05-05 21:57
@Auth: xjjxhxgg
@File: convert_visdrone_to_yolo.py
@IDE: PyCharm
@Motto: xhxgg
"""
import os
import cv2
def convert_visdrone_to_yolo(visdrone_root, yolo_root, cat_reid=False):
    """
    convert the visdrone dataset to yolo format
    :param visdrone_root:
    :param yolo_root:
    :param cat_reid:
    :return:
    """
    ann_rt = os.path.join(visdrone_root, 'annotations')
    img_rt = os.path.join(visdrone_root, 'images')
    anns = [ann for ann in os.listdir(ann_rt) if ann.endswith('.txt')]
    for ann in anns:
        ann_path = os.path.join(ann_rt, ann)
        img_path = os.path.join(img_rt, ann.split('.')[0] + '.jpg')
        H, W = cv2.imread(img_path).shape[:2]
        with open(ann_path, 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            line = line.strip().split(',')
            cat = int(line[5])
            xmin = int(line[0])
            ymin = int(line[1])
            bw = int(line[2])
            bh = int(line[3])
            xmax = xmin + bw
            ymax = ymin + bh
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            x_center /= W
            y_center /= H
            bw /= W
            bh /= H
            new_line = f'{cat} {x_center} {y_center} {bw} {bh}\n'
            new_lines.append(new_line)
        with open(os.path.join(yolo_root, ann), 'w') as f:
            f.writelines(new_lines)

def get_cats():
    yolo_root = r'G:\Graduation Design\Dataset\VisDrone\VisDrone2019-DET-test-dev\labels'
    lbls = [lbl for lbl in os.listdir(yolo_root) if lbl.endswith('.txt')]
    cats = set()
    for lbl in lbls:
        with open(os.path.join(yolo_root, lbl), 'r') as f:
            lines = f.readlines()
        for line in lines:
            cat = int(line.strip().split()[0])
            cats.add(cat)
    print(cats)

if __name__ == '__main__':
    # visdrone_root = r'G:\Graduation Design\Dataset\VisDrone\VisDrone2019-DET-test-dev'
    # yolo_root = fr'{visdrone_root}\labels'
    # convert_visdrone_to_yolo(visdrone_root, yolo_root)
    get_cats()