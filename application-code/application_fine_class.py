import numpy as np
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


# Scale and visualize the embedding vectors
def plot_embedding(X, img_names, title=None, size=(80,80)):
    # x_min, x_max = np.min(X, 0), np.max(X, 0)
    # X = (X - x_min) / (x_max - x_min)
    X = X
    plt.figure(figsize=size)
    ax = plt.subplot(111)
    """ grid coordinates """
    g_size = 50
    grid = 1.0/float(g_size)
    locations_ = (X / grid).astype(int)
    locations_ = locations_ / float(g_size)
    x = locations_[:,0]
    idx_x = np.argsort(x, axis=0)
    locations_ = locations_[idx_x]
    y = locations_[:,1]
    idx_y = np.argsort(-y, axis=0)
    locations = locations_[idx_y]
    idx = idx_x[idx_y]

    # locations = locations_
    # pdb.set_trace()

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(locations.shape[0]):
            dist = np.sum((locations[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-5:  # 4e-3
                # don't show points that are too close
                continue
            shape_img = get_shapeimg(img_names[idx[i]])
            shown_images = np.r_[shown_images, [locations[i]]]  # the coordinate of shown images
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(shape_img, zoom=0.23),  # 0.23
                locations[i], frameon=False)
            ax.add_artist(imagebox)
    # vis_x = X[:,0]
    # vis_y = X[:,1]
    # ax.scatter(vis_x, vis_y, s=5)
    # ax.scatter(locations[:,0], locations[:,1], s=20)
    plt.xticks([]), plt.yticks([])
    # # hide boundary
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if title is not None:
        plt.title(title)


# get images for visualize the 3D shapes
def get_shapeimg(img_name):
    return read_png(img_name)


if __name__ == '__main__':
    feature_fn = os.path.join('application_feature',CLASSNAME+'-avg_fea.npy')
    feature = np.load(feature_fn)  # (704,512)
    # pdb.set_trace()
    tsne = TSNE(n_components=2, perplexity=20, init='pca', early_exaggeration=80.0,
            n_iter=2500, random_state=501)  # perplexity 20, early_exaggeration 80
    fea_tsne = tsne.fit_transform(feature)
    x_min, x_max = np.min(fea_tsne, 0), np.max(fea_tsne, 0)
    X = (fea_tsne - x_min) / (x_max - x_min) * 1
    vis_x = X[:,0]
    vis_y = X[:,1]

    """ get corresponding images """
    file_list = os.path.join(BASE_DIR,'../train_test_split_matlab',
            CLASSNAME+'_test_file_list.txt')
    file_names = []
    with open(file_list) as f:
        for line in f.readlines():
            fn_tmp = line.rstrip('\n').split('/')
            img_dir = os.path.join('E:/LabWork/Data/PartAnnotation',
                    fn_tmp[0], 'expert_verified/seg_img_trans', fn_tmp[1]+'.png')
            # pdb.set_trace()
            file_names.append(img_dir)

    plot_embedding(X, file_names)
    save_dir = os.path.join('application_feature',CLASSNAME+'-avg_fea.png')
    plt.savefig(save_dir, dpi=100)
    # fig = plt.figure()
    # plt.scatter(vis_x, vis_y, s=5)
    pdb.set_trace()
