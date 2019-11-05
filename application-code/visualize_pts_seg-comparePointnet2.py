""" Visualize Points Segmentation: Main Results """
""" Original Author: Haoqiang Fan """
import numpy as np
import show3d_balls
import sys
import os
import pdb
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import color_settings as cg_

parser = argparse.ArgumentParser()
parser.add_argument('--clsname', default='')
parser.add_argument('--atrous_block_num', type=int, default=2)
FLAGS = parser.parse_args()


def read_file(filename):
    ptc_list = []
    with open(filename, 'r') as f:
        for line in f:
            p_list = line.split()[1:]  # point and color
            ptc = [float(i) for i in p_list]
            ptc_list.append(ptc)
    return np.asarray(ptc_list)


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m


def pts_aligned(pts):
    # rotate pts to fit octree data set rotation angles:
    rotation_angle = np.pi / 2
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                    [0, 1, 0],
                    [-sinval, 0, cosval]]).T  # counter-clock wise rotation
    norm_pts, pts_center, max_dis = pc_normalize(pts)
    rot_pts = np.dot(norm_pts.reshape((-1,3)), rotation_matrix)*max_dis + pts_center
    # rot_pts = np.dot(norm_pts.reshape((-1,3)), rotation_matrix)
    return rot_pts


color_tab = cg_.color_table
label_color = cg_.color_origin


def color_mapped(colors, clsname):
    # point label id start from 0;
    source_colors = np.asarray(label_color)
    target_colors = np.asarray(color_tab[clsname])
    label_num = target_colors.shape[0]
    colors_new = colors[:]+0.0
    # pdb.set_trace()
    for l in range(label_num):
        cur_src_clr = source_colors[l,...]
        cur_tar_clr = target_colors[l,...]
        idx = np.sum(np.abs(colors - cur_src_clr), axis=1) == 0
        colors_new[idx,...] = cur_tar_clr
    return colors_new


show_dict = {
             'Airplane': 13, 'Bag': 2, 'Cap': 1,
             'Car': 18, 'Chair': 50, 'Earphone': 0, 'Guitar': 13, 'Knife': 4,
             'Lamp': 40, 'Laptop': 10, 'Motorbike': 11, 'Mug': 1, 'Pistol': 0,
             'Rocket': 1, 'Skateboard': 2, 'Table': 18,
            }


if __name__ == '__main__':
    """ input:  point and color
        output: rendered color points figure
        pred-outs, pred ptnet, gt oc, gt ptnet
    """
    ballsize = 7
    classes = sorted(show_dict.keys())
    for cname in classes:
        # CLSNAME = cname
        CLSNAME = 'Laptop'
        shape_idx = show_dict[CLSNAME]  # Motor 3,4, 14, 17
        Atrous_dir = os.path.join(BASE_DIR, '../../result-data', 'test_results_PA_3DCNN_Atrous',
               CLSNAME+'-withBG-ABlock3-Res')
        PtNet_dir = os.path.join(BASE_DIR, '../../result-data', 'test_results_pointnet2',
               CLSNAME)
        fname_atrous = os.path.join(Atrous_dir, str(shape_idx)+'_pred'+'.obj')
        fname_ptnet = os.path.join(PtNet_dir, str(shape_idx)+'_pred'+'.obj')
        fname_atrous_gt = os.path.join(Atrous_dir, str(shape_idx)+'_gt'+'.obj')
        fname_ptnet_gt = os.path.join(PtNet_dir, str(shape_idx)+'_gt'+'.obj')

        data_atrous = read_file(fname_atrous)
        data_ptnet = read_file(fname_ptnet)
        data_atrous_gt = read_file(fname_atrous_gt)
        data_ptnet_gt = read_file(fname_ptnet_gt)

        pts_a = data_atrous[:,:3]

        pts_p_ = data_ptnet[:,:3]
        pts_p = pts_aligned(pts_p_)

        # if CLSNAME == 'Cap':
        #     color_ptnet = data_ptnet[:,3:6] + 0.0
        #     idx_1 = np.sum(np.abs(color_ptnet - np.array([0.65, 0.05, 0.05])), axis=1) == 0
        #     idx_2 = np.sum(np.abs(color_ptnet - np.array([0.65, 0.35, 0.95])), axis=1) == 0
        #     color_ptnet[idx_1,...] = np.array([0.7, 0.0, 0.0])
        #     color_ptnet[idx_2,...] = np.array([0.0, 1.0, 0.0])
        #     data_ptnet[:,3:6] = color_ptnet

        #     color_ptnet = data_ptnet_gt[:,3:6] + 0.0
        #     idx_1 = np.sum(np.abs(color_ptnet - np.array([0.65, 0.05, 0.05])), axis=1) == 0
        #     idx_2 = np.sum(np.abs(color_ptnet - np.array([0.65, 0.35, 0.95])), axis=1) == 0
        #     color_ptnet[idx_1,...] = np.array([0.7, 0.0, 0.0])
        #     color_ptnet[idx_2,...] = np.array([0.0, 1.0, 0.0])
        #     data_ptnet_gt[:,3:6] = color_ptnet

        clr_a = color_mapped(data_atrous[:,3:6], CLSNAME)
        clr_p = color_mapped(data_ptnet[:,3:6], CLSNAME)
        clr_a_gt = color_mapped(data_atrous_gt[:,3:6], CLSNAME)
        clr_p_gt = color_mapped(data_ptnet_gt[:,3:6], CLSNAME)
        # pdb.set_trace()

        clr_atrous = clr_a * 255
        clr_ptnet = clr_p * 255
        clr_a_gt = clr_a_gt * 255
        clr_p_gt = clr_p_gt * 255

        # savedir = os.path.join(BASE_DIR, '../../result-data', 'Rendered_imgs', CLSNAME+'_shape_'+str(shape_idx)+'.png')
        # show3d_balls.showpoints(xyz=pts_a, c_pred=clr_atrous[:,[1,0,2]], background=(255,255,255),  # background=(b,g,r)
        #     showrot=False,magnifyBlue=0,freezerot=False,
        #     normalizecolor=False, ballradius=ballsize, savedir=savedir)

        savedir = os.path.join(BASE_DIR, '../../result-data', 'Rendered_imgs', CLSNAME+'_shape_'+str(shape_idx)+'_pred_atrous.png')
        show3d_balls.showpoints(xyz=pts_a, c_pred=clr_atrous[:,[1,0,2]], background=(255,255,255),
            showrot=False,magnifyBlue=0,freezerot=False,
            normalizecolor=False, ballradius=ballsize, savedir=savedir)

        savedir = os.path.join(BASE_DIR, '../../result-data', 'Rendered_imgs', CLSNAME+'_shape_'+str(shape_idx)+'_pred_ptnet2.png')
        show3d_balls.showpoints(xyz=pts_p, c_pred=clr_ptnet[:,[1,0,2]], background=(255,255,255),
            showrot=False,magnifyBlue=0,freezerot=False,
            normalizecolor=False, ballradius=ballsize, savedir=savedir)

        savedir = os.path.join(BASE_DIR, '../../result-data', 'Rendered_imgs', CLSNAME+'_shape_'+str(shape_idx)+'_gt_atrous.png')
        show3d_balls.showpoints(xyz=pts_a, c_pred=clr_a_gt[:,[1,0,2]], background=(255,255,255),
            showrot=False,magnifyBlue=0,freezerot=False,
            normalizecolor=False, ballradius=ballsize, savedir=savedir)

        savedir = os.path.join(BASE_DIR, '../../result-data', 'Rendered_imgs', CLSNAME+'_shape_'+str(shape_idx)+'_gt_ptnet2.png')
        show3d_balls.showpoints(xyz=pts_p, c_pred=clr_p_gt[:,[1,0,2]], background=(255,255,255),
            showrot=False,magnifyBlue=0,freezerot=False,
            normalizecolor=False, ballradius=ballsize, savedir=savedir)

    # CLSNAME = 'Airplane'
    # test_dir = os.path.join(BASE_DIR, '../../result-data', 'test_results_PA_3DCNN_Atrous',
    #            CLSNAME+'-withBG-ABlock3-Res')
    # show_list = show_dict[CLSNAME]  # [stage,channel]

    # for i in show_list:
    #     shape_idx = i
    #     fname = os.path.join(test_dir, str(shape_idx)+'_pred'+'.obj')
    #     ptsclr = read_file(fname)

    #     clr = ptsclr[:,3:6] * 255
    #     # pdb.set_trace()
    #     savedir = os.path.join(test_dir, 'dump', 'shape_'+str(shape_idx)+'.png')
    #     show3d_balls.showpoints(xyz=ptsclr[:,:3], c_pred=clr[:,[1,0,2]], background=(255,255,255),
    #         showrot=False,magnifyBlue=0,freezerot=False,
    #         normalizecolor=False, ballradius=7, savedir=savedir)
