""" WZJ:20181121-new Ablation study: AtrousCNN+AFA"""
""" WZJ:20180122-Feature Extraction Network-(Pretrain 3DCNN_Unet_extract)"""
""" Conduct feature extraction for furthur RoI network [20171013] """
""" evaluate iou and acc """
""" 3DCNN, Unet and others train """
""" WZJ:20170921-Feature Extraction Network"""
import tensorflow as tf
import numpy as np
import argparse
import socket
import json
import importlib
import h5py
import time
import math
import os
import scipy.misc
import sys
from PIL import Image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import mask_data_prepare as Mask_util
import gen_part_box as Box_util  # for part bbox
import provider
import pc_util
import pdb

Common_SET_DIR = os.path.join(BASE_DIR, '../CommonFile')
sys.path.append(Common_SET_DIR)
import globals as g_

g_.Data_suf = 'OcTree'
g_.Data_suf = '_oc'

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='3DCNN_Atrous', help='Model name: boundary_model,3DCNN_model [default: 3DCNN_model]')
parser.add_argument('--log_dir', default='log-3DCNN_Atrous-ablationAFA', help='Log dir [default: log]')
# parser.add_argument('--log_dir', default='log-3DCNN_Atrous-Res2block-concat', help='Log dir [default: log]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--vox_size', type=int, default=48, help='voxel size: [32/48/128] [default: 32]')
parser.add_argument('--model_load', default='model_epoch_100.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--visu', type=bool, default=True, help='Whether to dump image for error case [default: False]')
parser.add_argument('--clsname', default='')

parser.add_argument('--atrous_block_num', type=int, default=3)
parser.add_argument('--withEmpty', dest='ignore_Empty', action='store_false')
parser.add_argument('--noEmpty', dest='ignore_Empty', action='store_true')
parser.set_defaults(ignore_Empty=False)
FLAGS = parser.parse_args()

# CLASSNAME = g_.CATEGORY_NAME
# part_num = g_.NUM_CLASSES - 1  # part num, Motorbike is 6; Earphone is 3; Rocked is 3;
CLASSNAME = FLAGS.clsname
part_num = g_.part_dict[CLASSNAME]
print('class name is ', CLASSNAME, '\tpart num is ', part_num,
     '\tignore_Empty:', FLAGS.ignore_Empty)


BATCH_SIZE = FLAGS.batch_size
VOX_SIZE = FLAGS.vox_size

# NUM_POINT = FLAGS.num_point
checkpoint_dir = os.path.join(FLAGS.log_dir+g_.Data_suf, str(VOX_SIZE),
    CLASSNAME+'-withBG'+'-ABlock'+str(FLAGS.atrous_block_num), 'trained_models')
MODEL_PATH = os.path.join(FLAGS.log_dir+g_.Data_suf, str(VOX_SIZE), CLASSNAME, 'trained_models', FLAGS.model_load)
if not FLAGS.ignore_Empty:
    MODEL_PATH = os.path.join(
        FLAGS.log_dir+g_.Data_suf, str(VOX_SIZE),
        CLASSNAME+'-withBG'+'-ABlock'+str(FLAGS.atrous_block_num),
        'trained_models', FLAGS.model_load)

GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model)  # import network module
output_dir = os.path.join(
        BASE_DIR, './test_results_' + g_.Data_suf + FLAGS.model, CLASSNAME+'_'+FLAGS.model_load)
if not FLAGS.ignore_Empty:
    output_dir = os.path.join(
            BASE_DIR, './test_results_' + g_.Data_suf + FLAGS.model + '-newAblation',
            CLASSNAME+'-withBG'+'-ABlock'+str(FLAGS.atrous_block_num)+'-evaltime')
output_verbose = False  # FLAGS.visu   # If true, output all color-coded part segmentation Image (Projection) files
# print('output_verbose', output_verbose)
# pdb.set_trace()
output_Box = False
if not os.path.exists(os.path.join(BASE_DIR, './test_results_' + g_.Data_suf + FLAGS.model + '-newAblation')):
    os.mkdir(os.path.join(BASE_DIR, './test_results_' + g_.Data_suf + FLAGS.model + '-newAblation'))
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
DUMP_DIR = os.path.join(output_dir, FLAGS.dump_dir)
if not os.path.exists(DUMP_DIR):
    os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

HOSTNAME = socket.gethostname()

# Part Label Configurations for computing IOU and Visualize color maps
hdf5_data_dir = os.path.join(BASE_DIR, 'hdf5_data'+g_.Data_suf)
oid2cpid = json.load(open(os.path.join(hdf5_data_dir, 'overallid_to_catid_partid.json'), 'r'))

object2setofoid = {}  # object and its corresponding part id set, {'03793390':[36,37], ...}
for idx in range(len(oid2cpid)):
    objid, pid = oid2cpid[idx]
    if objid not in list(object2setofoid.keys()):
        object2setofoid[objid] = []
    object2setofoid[objid].append(idx)

all_obj_cat_file = os.path.join(hdf5_data_dir, 'all_object_categories.txt')
fin = open(all_obj_cat_file, 'r')
lines = [line.rstrip() for line in fin.readlines()]
objcats = [line.split()[1] for line in lines]  # 02691156
objnames = [line.split()[0] for line in lines]  # Airplane
on2oid = {objcats[i]:i for i in range(len(objcats))}  # object name 2 object id, {'02691156': 0, ...}
fin.close()

SHAPE_NAMES = objnames  # for visualizing

color_map_file = os.path.join(hdf5_data_dir, 'part_color_mapping.json')
color_map = json.load(open(color_map_file, 'r'))
# color_map = [[0.95,0.35,0.35],[0.35,0.65,0.95]]  # for boundary and non-boundary
# color_map_red_blue = [[1.0,0.0,0.0],[0.2,0.2,1.0]]  # for correct color[1] blue; and incorrect color[0] red

# ShapeNetSeg official train/test split
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'hdf5_data'+g_.Data_suf+'/'+'0.'+CLASSNAME+'_filelistset/'+'test_hdf5_file_list.txt'))
TEST_FILES_PTS = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'hdf5_data'+g_.Data_suf+'/'+'0.'+CLASSNAME+'_filelistset/'+'test_hdf5_file_list_pts.txt'))


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def volume_to_point_cloud(vol):
    """ vol is occupancy grid (value = 0 or 1) of size vsize*vsize*vsize
        return Nx3 numpy array.
    """
    vsize = vol.shape[0]
    assert(vol.shape[1] == vsize and vol.shape[1] == vsize)
    points = []
    seg_label = []
    for a in range(vsize):
        for b in range(vsize):
            for c in range(vsize):
                if vol[a,b,c] != -1:
                    points.append(np.array([a,b,c]))
                    seg_label.append(vol[a,b,c])
    if len(points) == 0:
        return np.zeros((0,3)), np.zeros((0,1))
    points = np.vstack(points)
    seg_label = np.vstack(seg_label)
    return points, seg_label


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def vol_label_projection_to_pts(points, vol_seg, radius=1.0):  # for iou comparison with PointNet result
    """ Input points and voxel seg volume,
        output seg list corresponding to the points """
    normalize_Flag = True
    if normalize_Flag:
        points = pc_normalize(points)
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
    indice = np.where(seg_pts == -1)
    if np.sum(seg_pts == -1) != 0:
        if (np.sum(seg_pts == -1) / float(len(seg_pts))) > 0.01:
            print('seg_projection_failed, -1 num is : %d/%d' % (np.sum(seg_pts == -1), len(seg_pts)))
            print('vol_seg has -1? Num = ', np.sum(vol_seg == -1))
        # pdb.set_trace()
    return seg_pts


def draw_pts_seg(pts, plb):
    """ for visualization, quick """
    # print('drawing mask...')
    plt.close('all')

    fig = plt.figure()
    # plot the mini_mask
    # ax = fig.gca(projection='3d'

    # plot the full size mask: in pts format
    ax_1 = fig.add_subplot(111, projection='3d')
    ax_1.set_title('tfgraph computed box')
    ax_1.set_aspect('equal')
    ax_1.set_xlabel('--X->')
    ax_1.set_ylabel('--Y->')
    ax_1.set_zlabel('--Z->')
    # ax_1.view_init(elev=79., azim=-65.)
    ax_1.view_init(elev=17., azim=165.)
    # pdb.set_trace()

    X_pts = pts[:,0]
    Y_pts = pts[:,1]
    Z_pts = pts[:,2]

    # axl_m = np.amin(BOXES, axis=0)-1
    # axl_M = np.amax(BOXES, axis=0)+1
    # max_range = axl_M - axl_m
    # axl_m = np.min(BOXES)-1
    # axl_M = np.max(BOXES)+1
    axl_m = np.min(pts)
    axl_M = np.max(pts)
    ax_1.set_xlim(axl_m,axl_M)
    ax_1.set_ylim(axl_m,axl_M)
    ax_1.set_zlim(axl_m,axl_M)
    label_tab = np.unique(plb)
    for l in label_tab:
        idx = plb == l
        ax_1.scatter(X_pts[idx,...],Y_pts[idx,...],Z_pts[idx,...],s=3,alpha=0.8)


def output_color_point_cloud(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            color = color_map[np.int(seg[i])]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))


def output_color_point_cloud_red_blue(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            if seg[i] == 1:
                color = [0, 0, 1]
            elif seg[i] == 0:
                color = [1, 0, 0]
            else:
                color = [0, 0, 0]

            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))


def augment_to_target_num(fea, t_num):
    assert(fea.shape[0] <= t_num)
    cur_len = fea.shape[0]
    res = np.array(fea)
    while cur_len < t_num:
        res = np.concatenate((res, fea))  # axis=0
        cur_len += fea.shape[0]
    return res[:t_num, ...]


def evaluate(num_votes):
    # is_training = False

    with tf.device('/gpu:'+str(GPU_INDEX)):
        vol_ph, seg_ph = MODEL.placeholder_inputs(BATCH_SIZE, VOX_SIZE)
        is_training_ph = tf.placeholder(tf.bool, shape=())

        # simple model
        if FLAGS.ignore_Empty:
            pred, feature = MODEL.get_model(vol_ph, is_training_ph, part_num)
            loss = MODEL.get_loss(pred, seg_ph, part_num)
        else:
            if FLAGS.atrous_block_num == 1:
                # 1 block [1,2,3]
                pred, feature = MODEL.get_model_1block(vol_ph, is_training_ph,
                        part_num+1)
            elif FLAGS.atrous_block_num == 2:
                # 2 block [1,2,3] [1,3,5]
                pred, feature = MODEL.get_model_2block_v1_afa(vol_ph, is_training_ph,
                        part_num+1)  # [2,3,4]+2AFA
            elif FLAGS.atrous_block_num == 3:
                # 3 block [1,2,3] [1,3,5] [2,3,7]
                pred, feature = MODEL.get_model_3block(vol_ph, is_training_ph,
                        part_num+1)

            loss = MODEL.get_loss_withback(pred, seg_ph, part_num+1)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        log_string("Continue training from the model {}".format(
            ckpt.model_checkpoint_path))
    else:
        saver.restore(sess, MODEL_PATH)
        log_string("Model restored from %s" % MODEL_PATH)

    ops = {'vol_ph': vol_ph,
           'seg_ph': seg_ph,
           'is_training_ph': is_training_ph,
           'pred': pred,
           'feature': feature,
           'loss': loss,
           }

    eval_one_epoch(sess, ops, num_votes)


def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    is_training = False
    total_accuracy = 0.0
    total_acc_iou = 0.0  # the sum of average iou for instances
    total_acc_iou_pts = 0.0
    total_accuracy_pts = 0.0
    total_seen = 0.0
    loss_sum = 0.0

    if FLAGS.ignore_Empty:
        # from 0 to part_num
        iou_oids = range(part_num)
    else:
        # from 0 to part_num+1
        iou_oids = range(1, part_num+1, 1)

    # per category values
    # total_seen_class = np.zeros((NUM_CATEGORIES)).astype(np.float32)
    # total_accuracy_class = np.zeros((NUM_CATEGORIES)).astype(np.float32)
    # total_iou_class = np.zeros((NUM_CATEGORIES)).astype(np.float32)  # per category iou

    shape_idx = 0  # the index of the shape
    for fn in range(len(TEST_FILES)):
        cur_test_filename = os.path.join(BASE_DIR, 'hdf5_data'+g_.Data_suf, str(VOX_SIZE), TEST_FILES[fn])
        cur_test_pts_filename = os.path.join(BASE_DIR, 'hdf5_data'+g_.Data_suf, str(VOX_SIZE), TEST_FILES_PTS[fn])
        log_string('----Loading Test file ' + TEST_FILES[fn] + '----')
        # pdb.set_trace()
        current_data, current_seg, _ = provider.load_h5_volumes_data(cur_test_filename)
        current_seg = np.squeeze(current_seg)
        current_seg = current_seg.astype(np.float32)

        if g_.Dataset=='Mine':
            current_pts, current_plb = provider.load_h5(cur_test_pts_filename)
        elif g_.Dataset=='OcTree':
            current_pts, current_plb, current_ptsnum = provider.load_pts_oc_h5(cur_test_pts_filename)
            end = np.cumsum(current_ptsnum)
            start = np.concatenate(([0], end[:-1]), axis=0)

        num_data = current_data.shape[0]
        num_batches = num_data // BATCH_SIZE
        time_elapsed = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE  # BATCH_SIZE==1, start_idx==batch_idx
            end_idx = (batch_idx + 1) * BATCH_SIZE

            if start_idx % 100 == 0:
                print('%d/%d ...' % (start_idx, num_batches))

            feed_dict = {ops['vol_ph']: current_data[start_idx:end_idx, ...],
                         ops['seg_ph']: current_seg[start_idx:end_idx, ...],
                         ops['is_training_ph']: is_training}

            time_st = time.time()
            loss_val, pred_val = sess.run([ops['loss'],
                                         ops['pred']], feed_dict=feed_dict)
            time_ed = time.time()
            log_string('*******Time elapsed******* %.6f' % (time_ed - time_st))
            time_elapsed.append(time_ed-time_st)

            if math.isnan(loss_val):
                print('Detected NaN')
                pdb.set_trace()
            """ ------------------------------------------------------------ """

    log_string('mean time elapsed: %.6f' % (np.mean(time_elapsed)))


if __name__ == '__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=1)
    LOG_FOUT.close()
