""" Visualize Filter Responces """
""" Original Author: Haoqiang Fan """
import numpy as np
import show3d_balls
import h5py
import json
import sys
import os
import pdb
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import pc_util
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


def load_h5_volumes_data(h5_filename):
    """ WZJ-20170921 """
    f = h5py.File(h5_filename)
    data = f['data'][:]
    seg = f['seg'][:]
    return (data, seg)


def load_pts_oc_h5(h5_filename):
    f = h5py.File(h5_filename)
    print([key for key in f.keys()])
    data = f['data'][:]
    label = f['label'][:]
    ptsnum = f['ptsnum'][:]
    return (data, label, ptsnum)


def vol_label_projection_to_pts(points, vol_seg, radius=1.0):  # for iou comparison with PointNet result
    """ Input points and voxel seg volume,
        output seg list corresponding to the points """
    normalize_Flag = True
    if normalize_Flag:
        points = pc_util.pc_normalize(points)
    vol_seg = np.squeeze(vol_seg)
    seg_pts = -1*np.ones([points.shape[0], ])  # check shape
    vsize = vol_seg.shape[0]
    assert(vol_seg.shape[1] == vsize and vol_seg.shape[2] == vsize)
    voxel = 2*radius/float(vsize)  # because need shift the pts center to (radius,radius) position.
    locations = (points + radius)/voxel
    locations = locations.astype(int)  # floor(locations)
    locations[locations > (vsize - 1)] = vsize - 1
    locations[locations < 0] = 0
    loca_unique = np.unique(locations.view(np.dtype((np.void, locations.dtype.itemsize*locations.shape[1])))).view(locations.dtype).reshape(-1, locations.shape[1])
    for loca in loca_unique:
        idx = np.sum(np.abs(locations - loca), -1) == 0
        seg_pts[idx] = vol_seg[loca[0], loca[1], loca[2]]

    return seg_pts, loca_unique


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


show_dict = {'Airplane': [8],  # [model idx]}
             'Car': [5],
             'Motorbike': [5],
             'Chair': [299],
             }


if __name__ == '__main__':
    CLSNAME = 'Chair'
    test_dir = os.path.join(BASE_DIR, '../../3DCNN_Voxel', 'hdf5_data_oc', '48',
               CLSNAME+'_vol_test.h5')
    pts_dir = os.path.join(BASE_DIR, '../../3DCNN_Voxel', 'hdf5_data_oc', '4',
               CLSNAME+'_pts_test.h5')
    current_pts, current_plb, current_ptsnum = load_pts_oc_h5(pts_dir)
    end = np.cumsum(current_ptsnum)
    start = np.concatenate(([0], end[:-1]), axis=0)

    vsize_list = [4,8,16,32,48,64]
    show_list = show_dict[CLSNAME]  # [stage,channel]

    color_map_file = os.path.join(BASE_DIR, '../../3DCNN_Voxel/hdf5_data_oc',
                     'part_color_mapping.json')
    color_map = np.asarray(json.load(open(color_map_file, 'r')))

    shape_idx = 6  # 6,10
    st = start[shape_idx]
    ed = end[shape_idx]
    points = current_pts[st:ed, :]
    plb_gt = current_plb[st:ed]

    for v in vsize_list:
        [vox_data, vox_seg] = pc_util.point_cloud_to_volume_wzj(points=points,
                part_label=plb_gt, vsize=v)
        cur_vox = np.squeeze(vox_data)
        cur_seg = np.squeeze(vox_seg)
        plb_pred, _ = vol_label_projection_to_pts(points, cur_seg)
        # pts = np.transpose(np.array(np.where(cur_vox)))
        # plb = cur_seg[pts[:,0], pts[:,1], pts[:,2]] - 1.0
        pts = points
        plb = plb_pred - 1.0

        color = color_map[plb.astype(int)]
        clr_ = color_mapped(color, CLSNAME)
        clr = clr_ * 255
        # pdb.set_trace()
        savedir = os.path.join(BASE_DIR, 'AtrousBlock3', CLSNAME, 'shape_'+str(shape_idx)+'_vsize-'+str(v)+'.png')
        show3d_balls.showpoints(xyz=pts, c_pred=clr[:,[1,0,2]], background=(255,255,255),
            showrot=False,magnifyBlue=0,freezerot=False,
            normalizecolor=False, ballradius=8, savedir=savedir)  # 12


# if __name__ == '__main__':
#     CLSNAME = 'Chair'
#     test_dir = os.path.join(BASE_DIR, '../../3DCNN_Voxel', 'hdf5_data_oc', '48',
#                CLSNAME+'_vol_test.h5')
#     vsize_list = [4,8,16,32,48,64]
#     show_list = show_dict[CLSNAME]  # [stage,channel]

#     color_map_file = os.path.join(BASE_DIR, '../../3DCNN_Voxel/hdf5_data_oc',
#                      'part_color_mapping.json')
#     color_map = np.asarray(json.load(open(color_map_file, 'r')))

#     for i in show_list:
#         shape_idx = 39
#         for v in vsize_list:
#             test_dir = os.path.join(BASE_DIR, '../../3DCNN_Voxel', 'hdf5_data_oc', str(v),
#                 CLSNAME+'_vol_test.h5')
#             fname = test_dir
#             vox_data, vox_seg = load_h5_volumes_data(fname)
#             cur_vox = np.squeeze(vox_data[shape_idx,...])
#             cur_seg = np.squeeze(vox_seg[shape_idx, ...])
#             pts = np.transpose(np.array(np.where(cur_vox)))
#             plb = cur_seg[pts[:,0], pts[:,1], pts[:,2]] - 1.0

#             color = color_map[plb.astype(int)]
#             clr_ = color_mapped(color, CLSNAME)
#             clr = clr_ * 255
#             # pdb.set_trace()
#             savedir = os.path.join(BASE_DIR, 'AtrousBlock3', CLSNAME, 'shape_'+str(shape_idx)+'_vsize-'+str(v)+'.png')
#             show3d_balls.showpoints(xyz=pts, c_pred=clr[:,[1,0,2]], background=(255,255,255),
#                 showrot=False,magnifyBlue=0,freezerot=False,
#                 normalizecolor=False, ballradius=64, savedir=savedir)  # 12
