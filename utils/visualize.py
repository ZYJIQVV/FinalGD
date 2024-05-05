# -*- encoding: utf-8 -*-
"""
@Time: 2024-05-05 23:11
@Auth: xjjxhxgg
@File: visualize.py
@IDE: PyCharm
@Motto: xhxgg
"""
from pyzyj.visualize import visualize
from pyzyj.format import yolo_parser
import os
rt = r'G:\Graduation Design\Dataset\VisDrone\all'
vis_rt = os.path.join(rt, 'vis')
img_rt = os.path.join(rt, 'images')
lbl_rt = os.path.join(rt, 'labels')
imgs = [img for img in os.listdir(img_rt) if img.endswith('.jpg')]
for img in imgs:
    visualize(os.path.join(img_rt, img), os.path.join(lbl_rt, img.replace('.jpg', '.txt')),save_path=os.path.join(vis_rt, img), parser=yolo_parser)
