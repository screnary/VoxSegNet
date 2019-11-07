""" 3DCNN, Unet and others train """
""" WZJ:20171013-Feature Extraction Network-(Pretrain 3DCNN)"""
""" WZJ:20180122-Feature Extraction Network-(Pretrain 3DCNN_Unet_extract)"""
import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import json
import os
import sys
import pdb
import time
import pynvml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
# sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import provider
import tf_util

Common_SET_DIR = os.path.join(BASE_DIR, '../CommonFile')
sys.path.append(Common_SET_DIR)
import globals as g_

g_.Data_suf = 'OcTree'
g_.Data_suf = '_oc'

# Network Training Parameters Setting
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='3DCNN_Atrous', help='Model name: baseline segmentation network 3DCNN [default: 3DCNN_model]')
parser.add_argument('--log_dir', default='log-3DCNN_Atrous', help='Log dir [default: log]')
# parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--vox_size', type=int, default=48, help='voxel space size [32/48/128] [default: 32]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 32]')  # for extract, defult 8; if RoI, maybe less
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--wd', type=float, default=0.0005, help='Weight Decay [default: 0.0]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=8000, help='Decay step for lr decay [default: 50000, 200000]')
parser.add_argument('--decay_rate', type=float, default=0.75, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--Continue_MODE', type=bool, default=False, help='if train continue [default: False, from scratch]')
parser.add_argument('--Continue_Epoch', type=int, default=100, help='train continue from epoch [default: 50]')
parser.add_argument('--clsname', default='')

parser.add_argument('--atrous_block_num', type=int, default=3)
parser.add_argument('--withEmpty', dest='ignore_Empty', action='store_false')
parser.add_argument('--noEmpty', dest='ignore_Empty', action='store_true')
parser.set_defaults(ignore_Empty=False)
# parser.add_argument('--ignore_Empty', type=bool, default=False)

FLAGS = parser.parse_args()

# CLASSNAME = FLAGS.clsname
# print(CLASSNAME)
# # CLASSNAME = g_.CATEGORY_NAME
# part_num = g_.NUM_CLASSES - 1  # part num, Motorbike is 6; Earphone is 3; Rocked is 3;

CLASSNAME = FLAGS.clsname
part_num = g_.part_dict[CLASSNAME]
print('class name is ', CLASSNAME, '\tpart num is ', part_num,
     '\tignore_Empty:', FLAGS.ignore_Empty,
     '\tContinue_MODE:', FLAGS.Continue_MODE)
if CLASSNAME == 'Rocket' or CLASSNAME == 'Airplane':
    upright = 'z'
else:
    upright = 'y'

BATCH_SIZE = FLAGS.batch_size
VOX_SIZE = FLAGS.vox_size
# NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model + '.py')

LOG_DIR = os.path.join(FLAGS.log_dir+g_.Data_suf, str(VOX_SIZE), CLASSNAME)
if not FLAGS.ignore_Empty:
    LOG_DIR = os.path.join(FLAGS.log_dir+g_.Data_suf, str(VOX_SIZE),
            CLASSNAME+'-withBG'+'-ABlock'+str(FLAGS.atrous_block_num)+'-evaltime')

if not os.path.exists(FLAGS.log_dir+g_.Data_suf):
    os.mkdir(FLAGS.log_dir+g_.Data_suf)
    os.mkdir(os.path.join(FLAGS.log_dir+g_.Data_suf, str(VOX_SIZE)))
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
MODEL_STORAGE_PATH = os.path.join(LOG_DIR, "trained_models")
if not os.path.exists(MODEL_STORAGE_PATH):
    os.mkdir(MODEL_STORAGE_PATH)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp train_3DCNN_Atrous.py %s' % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

Continue_MODE = FLAGS.Continue_MODE  # "train_from_scratch"--False or "train"--True
Continue_Epoch = FLAGS.Continue_Epoch
checkpoint_dir = os.path.join(FLAGS.log_dir+g_.Data_suf, str(VOX_SIZE),
        CLASSNAME+'-withBG'+'-ABlock'+str(FLAGS.atrous_block_num), "trained_models")

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

# ShapeNet official train/val split
# h5_data_dir =
TRAIN_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'hdf5_data'+g_.Data_suf+'/'+'0.'+CLASSNAME+'_filelistset/'+'train_hdf5_file_list.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'hdf5_data'+g_.Data_suf+'/'+'0.'+CLASSNAME+'_filelistset/'+'test_hdf5_file_list.txt'))

# Categories name parsing
color_map_file = os.path.join(BASE_DIR, 'hdf5_data'+g_.Data_suf+'/part_color_mapping.json')
color_map = json.load(open(color_map_file, 'r'))

all_obj_cats_file = os.path.join(BASE_DIR, 'hdf5_data'+g_.Data_suf+'/all_object_categories.txt')
fin = open(all_obj_cats_file, 'r')
lines = [line.rstrip() for line in fin.readlines()]
all_obj_cats = [(line.split()[0], line.split()[1]) for line in lines]
fin.close()

all_cats = json.load(open(os.path.join(BASE_DIR, 'hdf5_data'+g_.Data_suf+'/overallid_to_catid_partid.json'), 'r'))
NUM_CATEGORIES = 16
NUM_PART_CATS = len(all_cats)


# # ---------------the functions definitions----------------
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


# decayed_learning_rate = learning_rate * decay_rate^(global_step / decay_step)
def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,
                        batch * BATCH_SIZE,  # current index into the dataset
                        DECAY_STEP,
                        DECAY_RATE,
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # clip the learning rate
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                        BN_INIT_DECAY,
                        batch * BATCH_SIZE,
                        BN_DECAY_DECAY_STEP,
                        BN_DECAY_DECAY_RATE,
                        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def augment_to_target_num(fea, t_num):
    assert(fea.shape[0] <= t_num)
    cur_len = fea.shape[0]
    res = np.array(fea)
    while cur_len < t_num:
        res = np.concatenate((res, fea))  # axis=0
        cur_len += fea.shape[0]
    return res[:t_num, ...]


def train():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    memo_st = pynvml.nvmlDeviceGetMemoryInfo(handle)
    initMemo = memo_st.used

    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            volumes_ph, seg_ph = MODEL.placeholder_inputs(BATCH_SIZE, VOX_SIZE)
            is_training_ph = tf.placeholder(tf.bool, shape=())
            print(is_training_ph)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)  # in the train_op operation, increased by 1
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            if FLAGS.ignore_Empty:
                pred, feature = MODEL.get_model(volumes_ph, is_training_ph,
                        part_num, bn_decay=bn_decay, weight_decay=FLAGS.wd)
                loss = MODEL.get_loss(pred, seg_ph, part_num)
            else:
                if FLAGS.atrous_block_num == 1:
                    # 1 block [1,2,3]
                    pred, feature = MODEL.get_model_1block(volumes_ph, is_training_ph,
                            part_num+1, bn_decay=bn_decay, weight_decay=FLAGS.wd)
                elif FLAGS.atrous_block_num == 2:
                    # 2 block [1,2,3] [1,3,5] + concat, SDE+concat
                    # pred, feature = MODEL.get_model_2block(volumes_ph, is_training_ph,
                            # part_num+1, bn_decay=bn_decay, weight_decay=FLAGS.wd)
                    # 2 block [1,2,3] [1,3,5] + concat, SDE+AFA
                    pred, feature = MODEL.get_model_2block_att(volumes_ph, is_training_ph,
                            part_num+1, bn_decay=bn_decay, weight_decay=FLAGS.wd)
                    # # 2 block [2,3,4] [1,1,1] + concat, Atrous-Encode
                    # pred, feature = MODEL.get_model_2block_v1(volumes_ph, is_training_ph,
                            # part_num+1, bn_decay=bn_decay, weight_decay=FLAGS.wd)
                    # # [2,3,4] + 2AFA (Ablation-new, Atrous+AFA)
                    # pred, feature = MODEL.get_model_2block_v1_afa(vol_ph, is_training_ph,
                            # part_num+1)
                elif FLAGS.atrous_block_num == 3:
                    # 3 block [1,2,3] [1,3,5] [2,3,7]
                    pred, feature = MODEL.get_model_3block(volumes_ph, is_training_ph,
                            part_num+1, bn_decay=bn_decay, weight_decay=FLAGS.wd)

                loss = MODEL.get_loss_withback(pred, seg_ph, part_num+1)
            tf.summary.scalar('loss', loss)

            if FLAGS.ignore_Empty:
                seg_new = tf.subtract(seg_ph, tf.constant(1, dtype=tf.float32))
                ignore_void = tf.constant(-1, dtype=tf.float32)
                mask_valid = tf.cast(tf.not_equal(seg_new, ignore_void), dtype=tf.float32)
                correct = tf.equal(tf.argmax(pred, -1), tf.to_int64(seg_new))  # argmax value >= 0, so ignore -1
                accuracy_per_instance = tf.reduce_sum(
                    tf.cast(correct, tf.float32), axis=[1,2,3])/\
                    tf.reduce_sum(mask_valid, axis=[1,2,3])
                accuracy = tf.reduce_mean(accuracy_per_instance)
            else:
                seg_new = seg_ph
                ignore_void = tf.constant(0, dtype=tf.float32)
                mask_valid = tf.cast(tf.not_equal(seg_new, ignore_void), dtype=tf.float32)
                correct = tf.cast(
                    tf.equal(tf.argmax(pred, -1), tf.to_int64(seg_new)), tf.float32)\
                    *mask_valid  # argmax value >= 0, so ignore -1
                accuracy_per_instance = tf.reduce_sum(
                    tf.cast(correct, tf.float32), axis=[1,2,3])/\
                    tf.reduce_sum(mask_valid, axis=[1,2,3])
                accuracy = tf.reduce_mean(accuracy_per_instance)
            tf.summary.scalar('accuracy', accuracy)
            # pdb.set_trace()
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables
            saver = tf.train.Saver(max_to_keep=50)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                             sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_ph: True})

        # Unet_vars = tf.trainable_variables()
        # for v in Unet_vars:
        #     print('var_name is : ', v.name)  # len=43
        # pdb.set_trace()
        # all_vars = tf.all_variables()
        # for v in all_vars:
        #     print('var_name is : ', v.name)  # len=149
        # pdb.set_trace()
        # continue training from last training
        
        num_parameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        log_string('Num of parameters: %d' % (num_parameters))

        if Continue_MODE:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("Continue training from the model {}".format(
                    ckpt.model_checkpoint_path))
                saver.restore(sess, ckpt.model_checkpoint_path)

        ops = {'volumes_ph': volumes_ph,
               'seg_ph': seg_ph,
               'is_training_ph': is_training_ph,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        if not os.path.exists(MODEL_STORAGE_PATH):
            os.mkdir(MODEL_STORAGE_PATH)

        epoch_start = 0
        if Continue_MODE:
                epoch_start = Continue_Epoch  # set by human
        # eval_one_epoch(sess, ops, test_writer)
        IoU_acc_max = 0.0
        epoch_use = 0
        for epoch in range(MAX_EPOCH):
            log_string('\n>>>>>>>>>> Training for the epoch %d/%d ...' % (epoch+epoch_start+1, MAX_EPOCH+epoch_start))
            train_one_epoch(sess, ops, train_writer)
            memo_ed = pynvml.nvmlDeviceGetMemoryInfo(handle)
            usedMemo = memo_ed.used
            log_string('GPU memo used: %.6f' % ((usedMemo-initMemo) / (1024.0*1024.0)))
            # if (epoch+epoch_start+1) % 1 == 0:
                # log_string('<<<<<<<<<< Testing on the test dataset ...')
                # sys.stdout.flush()
                # IoU_acc = eval_one_epoch(sess, ops, test_writer)
                # save_Flag = False
                # if IoU_acc >= IoU_acc_max:
                    # IoU_acc_max = IoU_acc
                    # epoch_use = epoch+epoch_start+1
                    # save_Flag = True

            # # Save the variables to disk
            # # if (save_Flag is True) or ((epoch+epoch_start+1) >= 5 and (epoch+epoch_start+1) % 5 == 0):
            # if (save_Flag is True) or epoch+epoch_start+1==MAX_EPOCH:
                # save_path = saver.save(sess, os.path.join(MODEL_STORAGE_PATH, "model_epoch_" + str(epoch+epoch_start+1) + ".ckpt"))
                # log_string("Model saved in file: %s" % save_path)
                # log_string("use model: %s, iou is %.4f" % ('epoch_num_'+str(epoch_use), IoU_acc_max))


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)

    for fn in range(len(TRAIN_FILES)):
        cur_train_filename = os.path.join(BASE_DIR, 'hdf5_data'+g_.Data_suf, str(VOX_SIZE), TRAIN_FILES[train_file_idxs[fn]])
        log_string('--- Loading train file ' + TRAIN_FILES[train_file_idxs[fn]] + '---')
        current_data, current_seg, _ = provider.load_h5_volumes_data(cur_train_filename)
        # pdb.set_trace()
        num_data = current_data.shape[0]
        num_batches = num_data // BATCH_SIZE
        # # shuffle the training data
        idx = np.arange(num_data)
        np.random.shuffle(idx)
        current_data = current_data[idx, ...]
        current_seg = current_seg[idx, ...]  # shape is [b, vsize, vsize, vsize, 1]
        current_seg = np.squeeze(current_seg)  # to the same dim of placeholder [b, vsize, vsize, vsize]
        current_seg = current_seg.astype(np.float32)

        total_accuracy = 0.0
        total_seen = 0.0
        loss_sum = 0.0

        total_acc_iou = 0.0
        if FLAGS.ignore_Empty:
            # from 0 to part_num
            iou_oids = range(part_num)  # for Motorbike part detection
        else:
            # from 0 to part_num+1
            iou_oids = range(1, part_num+1, 1)

        time_elapsed = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE

            if start_idx % 100 == 0:
                print('%d/%d ...' % (start_idx, num_data))

            input_data = current_data[start_idx:end_idx, ...]
            input_seg = current_seg[start_idx:end_idx, ...]
            # augment data by rotation along upright axis

            #input_data, input_seg = provider.rotate_voxel_data(
            #        input_data, input_seg, axis=upright)
            # pdb.set_trace()

            feed_dict = {ops['volumes_ph']: input_data,
                         ops['seg_ph']: input_seg,
                         ops['is_training_ph']: is_training}

            time_st = time.time()
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                         ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            time_ed = time.time()
            log_string('*******Time elapsed******* %.6f' % (time_ed - time_st))
            time_elapsed.append(time_ed-time_st)
            
            # train_writer.add_summary(summary, step)
            # pred_val = np.reshape(np.argmax(pred_val, -1), (BATCH_SIZE, VOX_SIZE, VOX_SIZE, VOX_SIZE))

            # if FLAGS.ignore_Empty:
                # cur_seg_new = input_seg - 1.0
            # else:
                # cur_seg_new = input_seg
            # # pred is from 0 value, but seg gt is from 1 value (0 for the back)
            # cur_voxel = np.reshape(input_data,
                    # (BATCH_SIZE, VOX_SIZE, VOX_SIZE, VOX_SIZE))
            # mask = cur_voxel>0
            # mask = mask.astype(np.float32)
            # correct = np.sum((pred_val == cur_seg_new)*mask, axis=(1,2,3))
            # seen_per_instance = np.sum(mask, axis=(1,2,3))
            # acc_per_instance = np.array(correct) / np.array(seen_per_instance)

            # total_accuracy += np.sum(acc_per_instance)
            # total_seen += BATCH_SIZE
            # loss_sum += loss_val

            # iou_log = ''  # iou details string
            # intersect_mask = np.int32((pred_val == cur_seg_new)*mask)  # [B,V,V,V]
            # # pdb.set_trace()
            # for bid in range(BATCH_SIZE):
                # # bid # batch id
                # total_iou = 0.0  # for this 3D shape.
                # intersect_mask_bid = intersect_mask[bid, ...]
                # mask_bid = mask[bid, ...]
                # pred_val_bid = pred_val[bid, ...]
                # cur_seg_bid = cur_seg_new[bid, ...]
                # for oid in iou_oids:
                    # n_pred = np.sum((pred_val_bid == oid) * mask_bid)  # only the valid grids' pred
                    # # n_pred = np.sum(seg_pred_val == oid)
                    # n_gt = np.sum(cur_seg_bid == oid)
                    # n_intersect = np.sum(np.int32(cur_seg_bid == oid) * intersect_mask_bid)
                    # n_union = n_pred + n_gt - n_intersect
                    # iou_log += '_pred:' + str(n_pred) + '_gt:' + str(n_gt) + '_intersect:' + str(n_intersect) + '_union:' + str(n_union) + '_'
                    # if n_union == 0:
                        # total_iou += 1
                        # iou_log += '_:1\n'
                    # else:
                        # total_iou += n_intersect * 1.0 / n_union  # sum across parts
                        # iou_log += '_:'+str(n_intersect*1.0/n_union)+'\n'

                # avg_iou = total_iou / len(iou_oids)  # average iou across parts, for one object
                # # pdb.set_trace()
                # total_acc_iou += avg_iou

        # log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        # log_string('accuracy: %f' % (total_accuracy / float(total_seen)))
        # log_string('train IoU accuracy: %f\n-----------------------------'
                   # % (total_acc_iou / float(total_seen)))
        log_string('mean time elapsed: %.6f' % (np.mean(time_elapsed)))


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_accuracy = 0.0
    total_seen = 0.0
    loss_sum = 0.0

    total_acc_iou = 0.0

    if FLAGS.ignore_Empty:
        # from 0 to part_num
        iou_oids = range(part_num)
    else:
        # from 0 to part_num+1
        iou_oids = range(1, part_num+1, 1)

    # total_seen_class = np.zeros((NUM_CATEGORIES)).astype(np.float32)
    # total_accuracy_class = np.zeros((NUM_CATEGORIES)).astype(np.float32)

    for fn in range(len(TEST_FILES)):
        cur_test_filename = os.path.join(BASE_DIR, 'hdf5_data'+g_.Data_suf, str(VOX_SIZE), TEST_FILES[fn])
        log_string('----Loading Validation file ' + TEST_FILES[fn] + '----')
        # pdb.set_trace()
        current_data, current_seg, _ = provider.load_h5_volumes_data(cur_test_filename)
        # current_label = np.squeeze(current_label)
        current_seg = np.squeeze(current_seg)
        current_seg = current_seg.astype(np.float32)
        BATCH_SIZE_eval = BATCH_SIZE
        num_data = current_data.shape[0]
        num_batches = np.ceil(num_data / BATCH_SIZE_eval).astype(int)
        # pdb.set_trace()
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE_eval
            end_idx = (batch_idx + 1) * BATCH_SIZE_eval

            if min(num_data-start_idx, end_idx-start_idx) < BATCH_SIZE_eval:
                input_data = augment_to_target_num(current_data[start_idx:min(end_idx, num_data), ...], BATCH_SIZE_eval)
                input_seg = augment_to_target_num(current_seg[start_idx:min(end_idx, num_data), ...], BATCH_SIZE_eval)
            else:
                input_data = current_data[start_idx:end_idx, ...]
                input_seg = current_seg[start_idx:end_idx, ...]

            feed_dict = {ops['volumes_ph']: input_data,
                         ops['seg_ph']: input_seg,
                         ops['is_training_ph']: is_training}

            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                         ops['loss'], ops['pred']], feed_dict=feed_dict)

            if math.isnan(loss_val):
                print('Detected NaN')
                # pdb.set_trace()

            pred_val = np.reshape(np.argmax(pred_val, -1), (BATCH_SIZE_eval, VOX_SIZE, VOX_SIZE, VOX_SIZE))  # B*Vox_size*Vox_size*Vox_size

            if FLAGS.ignore_Empty:
                cur_seg_new = input_seg - 1.0
            else:
                cur_seg_new = input_seg
            mask = np.reshape(input_data > 0,
                    (BATCH_SIZE, VOX_SIZE, VOX_SIZE, VOX_SIZE))
            mask = mask.astype(np.float32)  # [B,V,V,V]
            correct = np.sum((pred_val == cur_seg_new)*mask, axis=(1,2,3))
            seen_per_instance = np.sum(mask, axis=(1,2,3))
            acc_per_instance = np.array(correct) / np.array(seen_per_instance)

            total_accuracy += np.sum(acc_per_instance)
            total_seen += BATCH_SIZE_eval
            loss_sum += (loss_val * BATCH_SIZE_eval)

            iou_log = ''  # iou details string
            intersect_mask = np.int32((pred_val == cur_seg_new)*mask)  # [B,V,V,V]
            # pdb.set_trace()
            for bid in range(BATCH_SIZE_eval):
                # bid # batch id
                total_iou = 0.0  # for this 3D shape.
                intersect_mask_bid = intersect_mask[bid, ...]
                mask_bid = mask[bid, ...]
                pred_val_bid = pred_val[bid, ...]
                cur_seg_bid = cur_seg_new[bid, ...]
                for oid in iou_oids:
                    n_pred = np.sum((pred_val_bid == oid) * mask_bid)  # only the valid grids' pred
                    # n_pred = np.sum(seg_pred_val == oid)
                    n_gt = np.sum(cur_seg_bid == oid)
                    n_intersect = np.sum(np.int32(cur_seg_bid == oid) * intersect_mask_bid)
                    n_union = n_pred + n_gt - n_intersect
                    iou_log += '_pred:' + str(n_pred) + '_gt:' + str(n_gt) + '_intersect:' + str(n_intersect) + '_union:' + str(n_union) + '_'
                    if n_union == 0:
                        total_iou += 1
                        iou_log += '_:1\n'
                    else:
                        total_iou += n_intersect * 1.0 / n_union  # sum across parts
                        iou_log += '_:'+str(n_intersect*1.0/n_union)+'\n'

                avg_iou = total_iou / len(iou_oids)  # average iou across parts, for one object
                # pdb.set_trace()
                total_acc_iou += avg_iou

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f' % (total_accuracy / float(total_seen)))
    log_string('eval IoU accuracy: %f' % (total_acc_iou / float(total_seen)))
    return total_acc_iou / float(total_seen)


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
