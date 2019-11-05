""" Visualize Points Segmentation: Compare Attention Module """
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
             # 'Cap': 8,
             'Car': 9,
             'Chair': 0,
             # 'Pistol': 11,
            }


if __name__ == '__main__':
    """ input:  point and color
        output: rendered color points figure
        pred-outs, pred ptnet, gt oc, gt ptnet
    """
    ballsize = 7
    classes = sorted(show_dict.keys())
    for cname in classes:
        CLSNAME = cname
        shape_idx = show_dict[CLSNAME]
        # CLSNAME = 'Chair'
        # shape_idx = show_dict[CLSNAME]  # Motor 3,4, 14, 17
        Att_dir = os.path.join(BASE_DIR, '../../result-data', 'test_results_PA_3DCNN_Atrous',
                'withAttention', CLSNAME+'-withBG-ABlock2-Res')
        Con_dir = os.path.join(BASE_DIR, '../../result-data', 'test_results_PA_3DCNN_Atrous',
                'directConcate', CLSNAME+'-withBG-ABlock2-Res')
        fname_att = os.path.join(Att_dir, str(shape_idx)+'_pred'+'.obj')
        fname_con = os.path.join(Con_dir, str(shape_idx)+'_pred'+'.obj')
        fname_att_gt = os.path.join(Att_dir, str(shape_idx)+'_gt'+'.obj')
        fname_con_gt = os.path.join(Con_dir, str(shape_idx)+'_gt'+'.obj')

        data_att = read_file(fname_att)
        data_con = read_file(fname_con)
        data_att_gt = read_file(fname_att_gt)
        data_con_gt = read_file(fname_con_gt)

        pts_a = data_att[:,:3]
        pts_c = data_con[:,:3]

        clr_a = color_mapped(data_att[:,3:6], CLSNAME)
        clr_c = color_mapped(data_con[:,3:6], CLSNAME)
        clr_a_gt = color_mapped(data_att_gt[:,3:6], CLSNAME)
        clr_c_gt = color_mapped(data_con_gt[:,3:6], CLSNAME)
        # pdb.set_trace()

        clr_att = clr_a * 255
        clr_con = clr_c * 255
        clr_a_gt = clr_a_gt * 255
        clr_c_gt = clr_c_gt * 255

        # savedir = os.path.join(BASE_DIR, '../../result-data', 'Rendered_imgs_compare_att', CLSNAME+'_shape_'+str(shape_idx)+'.png')
        # show3d_balls.showpoints(xyz=pts_a, c_pred=clr_att[:,[1,0,2]], background=(255,255,255),  # background=(b,g,r)
        #     showrot=False,magnifyBlue=0,freezerot=False,
        #     normalizecolor=False, ballradius=ballsize, savedir=savedir)

        savedir = os.path.join(BASE_DIR, '../../result-data', 'Rendered_imgs_compare_att', CLSNAME+'_shape_'+str(shape_idx)+'_pred_att.png')
        show3d_balls.showpoints(xyz=pts_a, c_pred=clr_att[:,[1,0,2]], background=(255,255,255),
            showrot=False,magnifyBlue=0,freezerot=False,
            normalizecolor=False, ballradius=ballsize, savedir=savedir)

        savedir = os.path.join(BASE_DIR, '../../result-data', 'Rendered_imgs_compare_att', CLSNAME+'_shape_'+str(shape_idx)+'_pred_con.png')
        show3d_balls.showpoints(xyz=pts_a, c_pred=clr_con[:,[1,0,2]], background=(255,255,255),
            showrot=False,magnifyBlue=0,freezerot=False,
            normalizecolor=False, ballradius=ballsize, savedir=savedir)

        savedir = os.path.join(BASE_DIR, '../../result-data', 'Rendered_imgs_compare_att', CLSNAME+'_shape_'+str(shape_idx)+'_gt.png')
        show3d_balls.showpoints(xyz=pts_a, c_pred=clr_a_gt[:,[1,0,2]], background=(255,255,255),
            showrot=False,magnifyBlue=0,freezerot=False,
            normalizecolor=False, ballradius=ballsize, savedir=savedir)
