import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib._png import read_png
from sklearn.manifold import TSNE
import os
import sys
import argparse
import pdb

BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
parser = argparse.ArgumentParser()
parser.add_argument('--clsname', default='Chair')
FLAGS = parser.parse_args()

CLASSNAME = FLAGS.clsname


# get images for visualize the 3D shapes
def get_shapeimg(img_name):
    return read_png(img_name)


def get_transparent_img(img_fn):
    img = Image.open(img_fn)
    img = img.convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    return img


if __name__ == '__main__':
    """ get corresponding images """
    file_list = os.path.join(BASE_DIR,'../train_test_split_matlab',
            CLASSNAME+'_test_file_list.txt')

    with open(file_list) as f:
        for line in f.readlines():
            fn_tmp = line.rstrip('\n').split('/')
            img_dir = os.path.join('E:/LabWork/Data/PartAnnotation',
                    fn_tmp[0], 'expert_verified/seg_img', fn_tmp[1]+'.png')
            # pdb.set_trace()
            img = get_transparent_img(img_dir)
            save_dir = os.path.join('E:/LabWork/Data/PartAnnotation',
                     fn_tmp[0], 'expert_verified/seg_img_trans', fn_tmp[1]+'.png')
            img.save(save_dir, "PNG")

    # fig = plt.figure()
    # plt.scatter(vis_x, vis_y, s=5)
    # pdb.set_trace()
