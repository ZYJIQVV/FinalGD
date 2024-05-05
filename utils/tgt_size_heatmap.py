# -*- encoding: utf-8 -*-
"""
@Time: 2024-05-05 21:53
@Auth: xjjxhxgg
@File: tgt_size_heatmap.py
@IDE: PyCharm
@Motto: xhxgg
"""
from pyzyj.cal_tgt_size_heatmap import gen_heatmap
# gen_heatmap(yolo_root=r'G:\Graduation Design\Dataset\VisDrone',threshold=40,filename='heatmap-size-over-40.jpg')
if __name__ == '__main__':
    import numpy as np
    sizes = np.load(r'G:\Graduation Design\Dataset\VisDrone\VisDrone2019-DET-train\sizes.npy', allow_pickle=True)
    szs = sizes.tolist()
    szs = sorted(szs)
    print()