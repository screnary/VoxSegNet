""" 3DCNN encode decode U net
    mimick O-CNN segmentation network---20180120
    MASK RCNN: use 3D mask for parts,
    ---20180322--- add module
    #---Ver 0.1--- 3D Mask RCNN: input gt roi and gt mask, to check model efficiency
    ---20180420--- add module
    # 3D Part seek Generative Adversarial Network, input proposals with part label predicted by Unet,
    generate K part masks, in which only the k_th mask contribute to the back propagation.
    Discriminator assess the pair of (input proposal, output mask) is 'real'(gt) or 'false'(generated).
    By doing this, try to use 'whole part shape' information while pred the part mask. [reference: pixel2pixel github]
"""
import tensorflow as tf
import numpy as np
import os
import sys
import collections
import argparse
import pdb
# from multiprocessing import Queue
import multiprocessing
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
import gen_part_box as Box_util
import mask_data_prepare as Mask_util

Common_SET_DIR = os.path.join(BASE_DIR, '../../CommonFile')
sys.path.append(Common_SET_DIR)
import globals as g_

EPS = 1e-12
############################################################
#  3DCNN Unet Graph
############################################################


def placeholder_Unet_inputs(batch_size, vox_size):
    vol_ph = tf.placeholder(tf.float32, shape=(batch_size, vox_size, vox_size, vox_size, 1))
    seg_ph = tf.placeholder(tf.float32, shape=(batch_size, vox_size, vox_size, vox_size))
    return vol_ph, seg_ph


def build_Unet_FCN_v0(volumes, is_training, part_num, bn_decay=None, weight_decay=0.0):
    """ 3D FCN Unet for voxel wise label prediction
        shared feature extraction
        return: voxel wise label pred result: net, and feature maps dict: end_volumes
    """
    batch_size = volumes.get_shape()[0].value  # for TF tensor
    end_volumes = {}
    input_batch = volumes

    # conv3d with batch normalization, Encoding phase
    en1 = tf_util.conv3d(input_batch, 16, kernel_size=[3,3,3],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='conv1_1', bn_decay=bn_decay, weight_decay=weight_decay)  # relu
    en1_ = tf_util.conv3d(en1, 32, kernel_size=[3,3,3],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='conv1_2', bn_decay=bn_decay, weight_decay=weight_decay)  # relu  (48,48,48)
    en1 = tf_util.max_pool3d(en1, kernel_size=[3,3,3], padding='SAME', scope='pool1')  # (48,48,48)-->(24,24,24)

    en2 = tf_util.conv3d(en1, 32, kernel_size=[3,3,3],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='conv2', bn_decay=bn_decay, weight_decay=weight_decay)  # relu
    en2 = tf_util.max_pool3d(en2, kernel_size=[3,3,3], padding='SAME', scope='pool2')  # (24,24,24)-->(12,12,12)

    en3 = tf_util.conv3d(en2, 64, kernel_size=[3,3,3],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='conv3', bn_decay=bn_decay, weight_decay=weight_decay)  # relu
    en3 = tf_util.max_pool3d(en3, kernel_size=[3,3,3], padding='SAME', scope='pool3')  # (12,12,12)-->(6,6,6)

    # pdb.set_trace()
    # end_volumes['conv_features_layer5'] = en4

    # Decoding phase
    de1 = tf_util.conv3d_transpose(en3, 64, kernel_size=[3,3,3],
                                   padding='SAME', stride=[1,1,1], bn=True,
                                   is_training=is_training, scope='deconv_1', bn_decay=bn_decay, weight_decay=weight_decay)  # shape=(6,6,6)
    de1 = tf_util.dropout(de1, keep_prob=0.7, is_training=is_training, scope='drop_1')
    de1 = tf.concat(axis=-1, values=[de1, en3])

    de2 = tf_util.conv3d_transpose(de1, 64, kernel_size=[3,3,3],
                                   padding='SAME', stride=[2,2,2], bn=True,
                                   is_training=is_training, scope='deconv_2', bn_decay=bn_decay, weight_decay=weight_decay)  # shape=(12,12,12)
    de2 = tf_util.dropout(de2, keep_prob=0.7, is_training=is_training, scope='drop_2')
    de2 = tf.concat(axis=-1, values=[de2, en2])

    de3 = tf_util.conv3d_transpose(de2, 32, kernel_size=[3,3,3],
                                   padding='SAME', stride=[2,2,2], bn=True,
                                   is_training=is_training, scope='deconv_3', bn_decay=bn_decay, weight_decay=weight_decay)  # shape=(24,24,24)
    de3 = tf_util.dropout(de3, keep_prob=0.7, is_training=is_training, scope='drop_3')
    de3 = tf.concat(axis=-1, values=[de3, en1])

    de4 = tf_util.conv3d_transpose(de3, 16, kernel_size=[3,3,3],
                                   padding='SAME', stride=[2,2,2], bn=True,
                                   is_training=is_training, scope='deconv_4', bn_decay=bn_decay, weight_decay=weight_decay)  # shape=(48,48,48)
    de4 = tf_util.dropout(de4, keep_prob=0.7, is_training=is_training, scope='drop_4')
    de4 = tf.concat(axis=-1, values=[de4, en1_])
    end_volumes['deconv_features_layer4'] = de4

    # predicting phase
    net = tf_util.conv3d(de4, 64, kernel_size=[1,1,1],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='conv_pred_1', bn_decay=bn_decay, weight_decay=weight_decay)  # conv size 1
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='drop_4')
    net = tf_util.conv3d(net, 64, kernel_size=[1,1,1],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='conv_pred_2', bn_decay=bn_decay, weight_decay=weight_decay)
    end_volumes['conv_pred_2'] = net
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='drop_5')
    net = tf_util.conv3d(net, part_num, kernel_size=[1,1,1],  # predict 50 labels
                         padding='SAME', stride=[1,1,1], bn=False, activation_fn=None,
                         is_training=is_training, scope='conv_pred_4', bn_decay=bn_decay, weight_decay=weight_decay)

    return net, end_volumes


def build_Unet_FCN(volumes, is_training, part_num, bn_decay=None, weight_decay=0.0):
    """ 3D FCN for voxel wise label prediction--Atrous conv net """
    batch_size = volumes.get_shape()[0].value  # for TF tensor
    end_volumes = {}
    end_points = {}
    input_batch = volumes

    # conv3d with batch normalization, Encoding phase
    conv = tf_util.conv3d_atrous(input_batch, 16, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=1, bn=True,
                         is_training=is_training, use_xavier=True, scope='conv1')  # relu
    end_points['atrous_conv1'] = conv
    conv = tf_util.conv3d_atrous(conv, 32, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=2, bn=True,
                         is_training=is_training, use_xavier=True, scope='conv2')  # relu
    end_points['atrous_conv2'] = conv

    conv = tf_util.conv3d_atrous(conv, 32, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=4, bn=True,
                         is_training=is_training, use_xavier=True, scope='conv3')  # relu
    end_points['atrous_conv3'] = conv

    conv = tf_util.conv3d_atrous(conv, 64, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=8, bn=True,
                         is_training=is_training, use_xavier=True, scope='conv4')  # relu
    end_points['atrous_conv4'] = conv

    features = conv
    end_volumes['atrous_features_layer4'] = features

    # Decoding phase
    with tf.variable_scope("3DCNN_decoder"):
        output_c = [16,16,32,32]
        layers = []

        unet_point = []
        unet_point.append(end_points['atrous_conv1'])
        unet_point.append(end_points['atrous_conv2'])
        unet_point.append(end_points['atrous_conv3'])
        unet_point.append(end_points['atrous_conv4'])

        for i in range(3, -1, -1):
            if i==3:
                curr = tf_util.rrb(features, output_c[i], is_training, 'rrb_%d_1' % i)
                end_volumes['decode_features_layer1'] = curr
                curr = tf_util.dropout(curr, keep_prob=0.7, is_training=is_training,
                            scope='drop_%d' % i)
                layers.append(curr)
                continue
            curr = tf_util.rrb(unet_point[i], None, is_training, 'rrb_%d_1' % i)
            curr = tf_util.cab(layers[-1], curr, 'cab_%d' % i)
            curr = tf_util.rrb(curr, output_c[i], is_training, 'rrb_%d_2' % i)
            # curr = tf_util.dropout(curr, keep_prob=0.7, is_training=is_training,
            #            scope='drop_%d' % i)
            layers.append(curr)

    end_volumes['decode_features_layer4'] = layers[-1]
    net = tf_util.conv3d(layers[-1], 64, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=True,
                is_training=is_training, scope='pred_1')
    net = tf_util.conv3d(net, 64, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=True,
                is_training=is_training, scope='pred_2')
    predicted = tf_util.conv3d(net, part_num, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=False, activation_fn=None,
                is_training=is_training, scope='pred')
    return predicted, end_volumes


def build_Unet_FCN_atrous_nodecode(volumes, is_training, part_num, bn_decay=None, weight_decay=0.0):
    """ 3D FCN for voxel wise label prediction--Atrous conv net """
    batch_size = volumes.get_shape()[0].value  # for TF tensor
    end_volumes = {}
    end_points = {}
    input_batch = volumes

    # conv3d with batch normalization, Encoding phase
    conv = tf_util.conv3d_atrous(input_batch, 16, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=1, bn=True,
                         is_training=is_training, use_xavier=True, scope='conv1')  # relu
    end_points['atrous_conv1'] = conv
    conv = tf_util.conv3d_atrous(conv, 32, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=2, bn=True,
                         is_training=is_training, use_xavier=True, scope='conv2')  # relu
    end_points['atrous_conv2'] = conv

    conv = tf_util.conv3d_atrous(conv, 32, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=4, bn=True,
                         is_training=is_training, use_xavier=True, scope='conv3')  # relu
    end_points['atrous_conv3'] = conv

    conv = tf_util.conv3d_atrous(conv, 64, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=8, bn=True,
                         is_training=is_training, use_xavier=True, scope='conv4')  # relu
    end_points['atrous_conv4'] = conv

    features = conv
    end_volumes['atrous_features_layer4'] = features

    # Decoding phase

    net = tf_util.conv3d(features, 64, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=True,
                is_training=is_training, scope='pred_1')
    net = tf_util.conv3d(net, 64, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=True,
                is_training=is_training, scope='pred_2')
    predicted = tf_util.conv3d(net, part_num, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=False, activation_fn=None,
                is_training=is_training, scope='pred')
    return predicted, end_volumes

############################################################
#  ROIAlign Layer
############################################################
def crop_and_resize_by_axis(data, boxes, box_ind, re_size, ax):
    """ 3d ver, crop_and_resize should have input: (batch, h,w,ch) """
    resized_list = []
    unstack_img_depth_list = tf.unstack(data, axis=ax)
    for i in unstack_img_depth_list:
        resized_list.append(tf.image.crop_and_resize(image=i,
                boxes=boxes, box_ind=box_ind, crop_size=re_size, method='bilinear'))
    stack_img = tf.stack(resized_list, axis=ax)

    return stack_img


def crop_and_resize_data(boxes, data, crop_size=16):
    """ data: feature map, [1,h,w,d,n_ch]
        boxes: [num_rois, (x1,y1,z1,x2,y2,z2)], normalized
        crop_size: size, # [size,size,size]
    """
    boxes = tf.reshape(boxes, [-1, 6])
    box_ids = tf.zeros(shape=[boxes.get_shape()[0],], dtype=tf.int32)
    """ boxes need be normalized coordinates! """
    x1,y1,z1,x2,y2,z2 = tf.split(boxes, 6, axis=1)

    boxes_1 = tf.concat([x1,y1,x2,y2], axis=1)
    resized_along_depth = crop_and_resize_by_axis(data, boxes_1, box_ids,
                          [crop_size,crop_size], 3)  # shape [1,15,15,48,1]
    ones = tf.ones(shape=x1.get_shape())
    boxes_2 = tf.concat([ones*0,z1,ones,z2], axis=1)
    resized_along_width = crop_and_resize_by_axis(resized_along_depth, boxes_2,
                          box_ids, [crop_size,crop_size], 2)
    """ shape [1,15,15,15,n_ch] """
    return resized_along_width


def roi_pooling(boxes, inputs, pool_size):
    """ mimic tf.image.crop_and_resize()
    boxes: rois factor, Normalized! shape=(batch, num_rois, (x1,y1,z1,x2,y2,z2))
    inputs: feature map, shape=(batch, 48,48,48, channels)
    pool_size: 15 (output is [batch,15,15,15,15,n_ch])

    Return: pooled regions in the shape: (batch, num_boxes, h,w,d,channels)
        might be zero padded if not enough target ROIs. maybe use tf.map_fn()
    """

    batch_size = inputs.get_shape()[0]
    crop_batch = []
    for b in range(batch_size):
        fea = tf.expand_dims(inputs[b], axis=0)
        try:
            rois = boxes[b]
        except ValueError as e:
            print(e)
            print('batch id is: ', b)
            pdb.set_trace()
        # fn = lambda bboxes: crop_and_resize_data(bboxes, fea, pool_size)
        crop_datas = tf.squeeze(tf.map_fn(
                lambda bboxes: crop_and_resize_data(bboxes, fea, pool_size),
                elems=(rois)))
        crop_batch.append(tf.expand_dims(crop_datas, axis=0))

    return tf.concat(crop_batch, axis=0)


############################################################
#  Network Heads
############################################################

def placeholder_Mask_inputs(batch_size, num_rois, mask_shape):
    rois_ph = tf.placeholder(tf.float32, shape=(batch_size, num_rois, 6))
    gt_masks_ph = tf.placeholder(tf.float32,
            shape=(batch_size, num_rois, mask_shape, mask_shape, mask_shape))
    partid_ph = tf.placeholder(tf.float32, shape=(batch_size, num_rois))
    return rois_ph, gt_masks_ph, partid_ph


def build_mask_graph(rois, feature_maps, pool_size, part_num,
                     is_training, bn_decay=None, weight_decay=0.0):
    """ Builds the computation graph of the mask head.
    rois: [batch, num_rois, (x1,y1,z1,x2,y2,z2)]
    feature_maps:
    pool_size: the width of the cube feature map generated from ROI Pooling.
    num_classes (part_num): number of (part) classes, determines the channel of results

    Returns: Masks [batch,roi_count,height,width,depth,num_classes]
    """
    feature_map = feature_maps['deconv_features_layer4']  # [batch, vox,vox,vox,48]
    batch_size = feature_map.get_shape()[0]
    rois_num = rois.get_shape()[1]
    # ROI Pooling
    pooled_fea = roi_pooling(rois, feature_map, pool_size)  # [batch,num_rois,15,15,15,n_ch]
    # Conv Layers
    input_batch = tf.reshape(pooled_fea,
                [-1,pool_size,pool_size,pool_size,feature_map.get_shape()[-1]])
    en1 = tf_util.conv3d(input_batch, 32, kernel_size=[3,3,3],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='mask_conv1',
                         bn_decay=bn_decay, weight_decay=weight_decay)
    en2 = tf_util.conv3d(en1, 32, kernel_size=[3,3,3],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='mask_conv2',
                         bn_decay=bn_decay, weight_decay=weight_decay)
    en3 = tf_util.conv3d(en2, 32, kernel_size=[3,3,3],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='mask_conv3',
                         bn_decay=bn_decay, weight_decay=weight_decay)
    en4 = tf_util.conv3d(en3, 32, kernel_size=[3,3,3],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='mask_conv4',
                         bn_decay=bn_decay, weight_decay=weight_decay)

    net = tf_util.conv3d(en4, part_num, kernel_size=[1,1,1],
                         padding='SAME', stride=[1,1,1], bn=False,
                         activation_fn=tf.nn.sigmoid, is_training=is_training,
                         scope='mask')
    """ Masks [batch,roi_count,height,width,depth,num_classes] """
    mask = tf.reshape(net,
            [batch_size, rois_num,
            net.get_shape()[1], net.get_shape()[2], net.get_shape()[3],
            net.get_shape()[-1]])
    return mask


############################################################
#  Loss Functions
############################################################

def Unet_FCN_loss(pred, seg_labels, part_num):  # seg_labels: type uint8; pred: type float32
    """ pred: B * vol_size*vol_size*vol_size * num_pid,
        seg_label: B * vol_size*vol_size*vol_size """
    seg_shift_labels = tf.subtract(seg_labels, tf.constant(1, dtype=tf.float32))
    ignore_label = tf.constant(-1, dtype=tf.float32)
    mask = tf.cast(tf.not_equal(seg_shift_labels, ignore_label), dtype=tf.float32)
    gt_shift = tf.one_hot(tf.to_int32(seg_shift_labels), depth=part_num)

    batch_seg_loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=gt_shift)  # return shape [Batch_size, vox_size, vox_size, vox_size]
    # batch_seg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=seg_shift_labels)  # this returns nan, not right
    # # tf.nn.sparse_... returns a Tensor of the same shape as labels and of the same type as logits with the softmax cross entropy loss
    per_instance_seg_loss = tf.reduce_sum(tf.multiply(batch_seg_loss, mask), axis=[1,2,3]) / tf.reduce_sum(mask, axis=[1,2,3])
    seg_loss = tf.reduce_mean(per_instance_seg_loss, axis=None)

    # # >>>>>>add regularization term--wzj 20170921

    return seg_loss, batch_seg_loss, seg_shift_labels  # the 'batch_seg_loss' and 'seg_shift_labels' are for debug


def mask_loss_graph(pred_masks, target_masks, target_class_ids):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width, depth].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, depth, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = tf.reshape(target_class_ids, [-1])
    mask_shape = tf.shape(target_masks)
    target_masks = tf.reshape(target_masks,
               [-1,mask_shape[2],mask_shape[3],mask_shape[4]])
    pred_shape = tf.shape(pred_masks)
    pred_masks = tf.reshape(pred_masks,
               [-1,pred_shape[2],pred_shape[3],pred_shape[4],pred_shape[5]])
    # Permute predicted masks to [N, num_classes, height, width, depth]
    pred_masks = tf.transpose(pred_masks, [0, 4, 1, 2, 3])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids-1], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)  # pred n_ch start from 0, so indices should -1

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = tf.keras.backend.switch(tf.size(y_true) > 0,
                    tf.keras.backend.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    loss = tf.keras.backend.mean(loss)
    loss = tf.squeeze(tf.keras.backend.reshape(loss, [1,1]))  # [1,1]
    # pdb.set_trace()
    return loss


############################################################
#  part mask generation conditional GAN module---20180420
############################################################
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake,\
        discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1,\
        gen_grads_and_vars, train, global_step, proposals")


def placeholder_GAN_inputs_v0(batch_size, num_rois, mask_shape):
    """ proposals: [batch_size, mask_shape, mask_shape, mask_shape, 1];
        part_id: [batch_size, 1]; (the predicted part label of the proposals)
    """
    proposals_ph = tf.placeholder(tf.float32, shape=(batch_size, num_rois,
                    mask_shape, mask_shape, mask_shape,1))
    targets_ph = tf.placeholder(tf.float32,
            shape=(batch_size, num_rois, mask_shape, mask_shape, mask_shape))
    partid_ph = tf.placeholder(tf.float32, shape=(batch_size, num_rois))
    return proposals_ph, targets_ph, partid_ph


def placeholder_GAN_inputs(batch_size, num_rois, mask_shape):
    """ rois: (batch, num_rois, (x1,y1,z1,x2,y2,z2))
        # proposals: [batch_size, mask_shape, mask_shape, mask_shape, 1];
        part_id: [batch_size, 1]; (the predicted part label of the proposals)
    """
    rois_ph = tf.placeholder(tf.float32, shape=(batch_size, num_rois, 6))
    partid_ph = tf.placeholder(tf.float32, shape=(batch_size, num_rois))
    return rois_ph, partid_ph


def get_roi_func(vox_datas, pred_segs, Max_Ins_Num):
    """ np function
        input a batch of voxels and gt_segs, pred_segs
        compute boxes (rois factors)
        output: rois, normalized
    """
    batch_size = vox_datas.shape[0]
    vox_size = vox_datas.shape[2]
    Bbox_list = []
    Pid_list = []
    for b in range(batch_size):
        cur_vox = np.squeeze(vox_datas[b, ...])
        cur_pred_seg = np.squeeze(pred_segs[b, ...])
        pts = np.transpose(np.array(np.where(cur_vox)))
        plb = cur_pred_seg[pts[:,0], pts[:,1], pts[:,2]]
        pts = pts.astype(float)
        # print('compute box...')
        # Mask_util.tic()
        Boxes, BLabels, _, _ = Box_util.computeBox(pts=pts, plb=plb, alpha=1.5)
        # Mask_util.toc()  # 4.04s
        # print('compute rois...')
        # Mask_util.tic()
        boxes_, part_ids_ = Box_util.gen_rois_info(Boxes, BLabels)
        # boxes, part_ids = Box_util.augment_rois(boxes_, part_ids_, vox_size)
        # pdb.set_trace()
        # Mask_util.toc()  # 0.0001s
        Bbox_data = np.zeros((Max_Ins_Num, 6))  # [rois, x1,y1,z1, x2,y2,z2]
        Label_data = np.zeros((Max_Ins_Num,))
        b_count = len(part_ids)  # box count
        if b_count <= Max_Ins_Num:
            Bbox_data[0:b_count, ...] = np.asarray(boxes)
            Label_data[0:b_count, ...] = np.asarray(part_ids)
        elif b_count > Max_Ins_Num:
            print("Warning: box count larger than Max_Ins_Num! ", b_count, '/', Max_Ins_Num)
            b_count = Max_Ins_Num
            Bbox_data[0:b_count, ...] = np.asarray(boxes)[0:b_count, ...]
            Label_data[0:b_count, ...] = np.asarray(part_ids)[0:b_count, ...]
        Bbox_list.append(np.expand_dims(Bbox_data, axis=0))
        Pid_list.append(np.expand_dims(Label_data, axis=0))
    bbox_res = np.concatenate(Bbox_list,0) / (vox_size-1)  # normalize to [0~1]
    partid_res = np.concatenate(Pid_list, 0)
    # pdb.set_trace()
    return bbox_res.astype(np.float32), partid_res.astype(np.float32)


def _compute_roi(vox, seg, is_training):
    cur_vox = np.asarray(vox)
    vox_size = cur_vox.shape[1]
    cur_pred_seg = np.asarray(seg)
    Max_Ins_Num = g_.MAX_INS_NUM
    info = {}
    pts = np.transpose(np.array(np.where(cur_vox)))
    plb = cur_pred_seg[pts[:,0], pts[:,1], pts[:,2]]
    pts = pts.astype(float)
    Boxes, BLabels, _, _ = Box_util.computeBox(pts=pts, plb=plb, alpha=1.5)
    boxes_, part_ids_ = Box_util.gen_rois_info(Boxes, BLabels)
    boxes = boxes_
    part_ids = part_ids_
    # # augment rois

    # if is_training:
    #     boxes_rand, part_ids_rand = Box_util.augment_pred_rois(
    #         np.expand_dims(boxes,0), np.expand_dims(part_ids,0),
    #         vox_size, Max_Ins_Num)
    #     boxes = np.squeeze(boxes_rand)
    #     part_ids = np.squeeze(part_ids_rand)
    # else:
    #     # print('_compute_roi boxes shape', np.asarray(boxes).shape)
    #     boxes_rand, part_ids_rand = Box_util.augment_pred_rois(
    #         np.expand_dims(boxes,0), np.expand_dims(part_ids,0),
    #         vox_size, Max_Ins_Num)
    #     boxes = np.squeeze(boxes_rand)
    #     part_ids = np.squeeze(part_ids_rand)
    #   # pass
    # print('_compute_roi part_ids', part_ids, len(part_ids))
    Bbox_data = np.zeros((Max_Ins_Num, 6))  # [rois, x1,y1,z1, x2,y2,z2]
    Label_data = np.zeros((Max_Ins_Num,))
    b_count = len(part_ids)  # box count, 30
    if b_count <= Max_Ins_Num:
        Bbox_data[0:b_count, ...] = np.asarray(boxes)
        Label_data[0:b_count, ...] = np.asarray(part_ids)
    elif b_count > Max_Ins_Num:
        print("Warning: box count larger than Max_Ins_Num! ", b_count, '/', Max_Ins_Num)
        b_count = Max_Ins_Num
        Bbox_data[0:b_count, ...] = np.asarray(boxes)[0:b_count, ...]
        Label_data[0:b_count, ...] = np.asarray(part_ids)[0:b_count, ...]
    info['roi'] = Bbox_data
    info['part_id'] = Label_data
    # print('_compute_roi aug boxes', boxes)
    return info


def get_roi_func_fast(vox_datas, pred_segs, Max_Ins_Num, is_training):
    """ multi process version; Not success, slower
        np function
        input a batch of voxels and gt_segs, pred_segs
        compute boxes (rois factors)
        output: rois, normalized
    """
    num_shape = vox_datas.shape[0]  # batch size
    vox_size = vox_datas.shape[2]

    assert Max_Ins_Num == g_.MAX_INS_NUM, 'max ins num not right!'
    sub_size = 1

    cores = multiprocessing.cpu_count()  # 8 cores
    pool = multiprocessing.Pool(processes=int(cores))
    pool_list = []
    result_list = []
    # start_time = time.time()
    for i in range(0, num_shape, sub_size):
        sub_vox = np.squeeze(vox_datas[i:i+sub_size,...])
        sub_pred_seg = np.squeeze(pred_segs[i:i+sub_size, ...])
        pool_list.append(pool.apply_async(_compute_roi, args=(sub_vox,sub_pred_seg,is_training)))
    result_list = [xx.get() for xx in pool_list]
    pool.close()  # close pool, do not recept new process any more
    pool.join()  # block main process, waiting for sub processes output

    x = np.zeros((num_shape, Max_Ins_Num, 6))
    y = np.zeros((num_shape, Max_Ins_Num))
    # pdb.set_trace()
    for i in range(0, num_shape, sub_size):
        info = result_list[i]
        x[i:i+sub_size, ...] = info['roi'] / (vox_size-1)
        y[i:i+sub_size, ...] = info['part_id']
    # print('get_roi_func_fast run time: %.2f' % (time.time()-start_time))

    return x.astype(np.float32), y.astype(np.float32)


def build_bbox_graph(volumes, seg_pred, Max_Ins_Num, is_training):
    """
        volumes:  (batch_size, vox_size, vox_size, vox_size, 1)
        seg_pred: (batch_size, vox_size, vox_size, vox_size, part_num)
    """
    vol_shape = volumes.get_shape()
    # shift pred label start from 1
    seg_pred_ = tf.cast(tf.argmax(seg_pred, axis=-1) + 1, tf.float32)  # (batch_size, vox_size, vox_size, vox_size)

    boxes, part_ids = tf.py_func(get_roi_func_fast,
            inp=[volumes, seg_pred_, Max_Ins_Num, is_training],
            Tout=[tf.float32,tf.float32])

    # part_ids = tf.py_func(get_roi_func,
    #         inp=[volumes, seg_pred_, Max_Ins_Num],
    #         Tout=[tf.float32,tf.float32])[1]
    # pdb.set_trace()
    boxes = tf.reshape(boxes, [vol_shape[0],Max_Ins_Num,6])
    part_ids = tf.reshape(part_ids, [vol_shape[0],Max_Ins_Num])
    # pdb.set_trace()
    return boxes, part_ids


def create_generator_v1(proposals, part_num,
                     is_training, bn_decay=None, weight_decay=0.0):
    """ Builds the computation graph of the mask head.
    light weighted generator
    num_classes (part_num): number of (part) classes, determines the channel of results

    Returns: Masks [batch,roi_count,height,width,depth,num_classes]
    """
    # Conv Layers
    batch_size = proposals.get_shape()[0]
    channel_num = proposals.get_shape()[5]
    rois_num = proposals.get_shape()[1]
    mask_shape = proposals.get_shape()[2]
    input_batch = tf.reshape(proposals,
            [-1, mask_shape, mask_shape, mask_shape, channel_num])

    en1 = tf_util.conv3d(input_batch, 32, kernel_size=[3,3,3],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='mask_conv1',
                         bn_decay=bn_decay, weight_decay=weight_decay)
    en2 = tf_util.conv3d(en1, 32, kernel_size=[3,3,3],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='mask_conv2',
                         bn_decay=bn_decay, weight_decay=weight_decay)
    en3 = tf_util.conv3d(en2, 32, kernel_size=[3,3,3],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='mask_conv3',
                         bn_decay=bn_decay, weight_decay=weight_decay)
    en4 = tf_util.conv3d(en3, 32, kernel_size=[3,3,3],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='mask_conv4',
                         bn_decay=bn_decay, weight_decay=weight_decay)

    net = tf_util.conv3d(en4, part_num, kernel_size=[1,1,1],
                         padding='SAME', stride=[1,1,1], bn=False,
                         activation_fn=tf.nn.sigmoid, is_training=is_training,
                         scope='mask')
    """ Masks [batch,roi_count,height,width,depth,num_classes] """
    mask = tf.reshape(net,
            [batch_size, rois_num,
            net.get_shape()[1], net.get_shape()[2], net.get_shape()[3],
            net.get_shape()[-1]])

    return mask


def create_generator_v0(proposals, part_num,
                     is_training, bn_decay=None, weight_decay=0.0):
    """ Unet Structure! using this as default
    input proposals of possible part region, generate its precise mask
        proposals: [batch_size, num_rois, mask_shape, mask_shape, mask_shape, 1]; integer of [0,1]
        part_ids: [batch_size, num_rois, 1]; (the predicted part label of the proposals)

        output: pred masks: [batch_size, num_rois, mask_shape, mask_shape, mask_shape, part_num], real num [0~1]
    """
    batch_size = proposals.get_shape()[0]
    rois_num = proposals.get_shape()[1]
    mask_shape = proposals.get_shape()[2]
    channel_num = proposals.get_shape()[5]
    input_batch = tf.reshape(proposals,
            [-1, mask_shape, mask_shape, mask_shape, channel_num])
    en1 = tf_util.conv3d(input_batch, 16, kernel_size=[3,3,3],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='gen_conv1',
                         bn_decay=bn_decay, weight_decay=weight_decay)
    # (b*num_rois, 16,16,16,1)-->(b*num_rois, 8,8,8,16)
    en1 = tf_util.max_pool3d(en1, kernel_size=[3,3,3], padding='SAME', scope='gen_pool1')

    en2 = tf_util.conv3d(en1, 32, kernel_size=[3,3,3],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='gen_conv2',
                         bn_decay=bn_decay, weight_decay=weight_decay)
    # (b*num_rois, 8,8,8,16)-->(b*num_rois, 4,4,4,32)
    en2 = tf_util.max_pool3d(en2, kernel_size=[3,3,3], padding='SAME', scope='gen_pool2')

    en3 = tf_util.conv3d(en2, 32, kernel_size=[2,2,2],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='gen_conv3',
                         bn_decay=bn_decay, weight_decay=weight_decay)
    # (b*num_rois, 4,4,4,32)-->(b*num_rois, 2,2,2,32)
    en3 = tf_util.max_pool3d(en3, kernel_size=[2,2,2], padding='SAME', scope='gen_pool3')

    en4 = tf_util.conv3d(en3, 64, kernel_size=[2,2,2],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='gen_conv4',
                         bn_decay=bn_decay, weight_decay=weight_decay)
    # (b*num_rois, 2,2,2,32)-->(b*num_rois, 1,1,1,64)
    en4 = tf_util.max_pool3d(en4, kernel_size=[2,2,2], padding='SAME', scope='gen_pool4')

    # Decoding phase
    de1 = en4
    de1 = tf_util.conv3d_transpose(de1, 32, kernel_size=[2,2,2],
                                   padding='SAME', stride=[2,2,2], bn=True,
                                   is_training=is_training, scope='gen_deconv_1',
                                   bn_decay=bn_decay, weight_decay=weight_decay)
    de1 = tf_util.dropout(de1, keep_prob=0.7, is_training=is_training, scope='gen_drop_1')
    de1 = tf.concat(axis=-1, values=[de1, en3])  # (b*n,1,1,1,64)-->(b*n,2,2,2,32*2)

    de2 = tf_util.conv3d_transpose(de1, 32, kernel_size=[2,2,2],
                                   padding='SAME', stride=[2,2,2], bn=True,
                                   is_training=is_training, scope='gen_deconv_2',
                                   bn_decay=bn_decay, weight_decay=weight_decay)
    de2 = tf_util.dropout(de2, keep_prob=0.7, is_training=is_training, scope='gen_drop_2')
    de2 = tf.concat(axis=-1, values=[de2, en2])  # (b*n,2,2,2,32*2)-->(b*n,4,4,4,32*2)

    de3 = tf_util.conv3d_transpose(de2, 32, kernel_size=[3,3,3],
                                   padding='SAME', stride=[2,2,2], bn=True,
                                   is_training=is_training, scope='gen_deconv_3',
                                   bn_decay=bn_decay, weight_decay=weight_decay)
    de3 = tf_util.dropout(de3, keep_prob=0.7, is_training=is_training, scope='gen_drop_3')
    de3 = tf.concat(axis=-1, values=[de3, en1])  # (b*n,4,4,4,32*2)-->(b*n,8,8,8,32+16)

    de4 = tf_util.conv3d_transpose(de3, 32, kernel_size=[3,3,3],
                                   padding='SAME', stride=[2,2,2], bn=True,
                                   is_training=is_training, scope='gen_deconv_4',
                                   bn_decay=bn_decay, weight_decay=weight_decay)
    # (b*n,8,8,8,32+16)-->(b*n,16,16,16,32)
    # ## fully connected pred layer
    net = tf_util.conv3d(de4, part_num, kernel_size=[1,1,1],
                         padding='SAME', stride=[1,1,1], bn=False,
                         activation_fn=tf.nn.sigmoid, is_training=is_training,
                         scope='gen_mask')
    """ Masks [batch,roi_count,height,width,depth,num_classes] """
    mask = tf.reshape(net,
                      [batch_size, rois_num,
                      net.get_shape()[1], net.get_shape()[2], net.get_shape()[3],
                      net.get_shape()[-1]])
    return mask


def create_generator_v2(proposals, part_num,
                     is_training, bn_decay=None, weight_decay=0.0):
    """ Unet Structure! using this as default
    input proposals of possible part region, generate its precise mask
        proposals: [batch_size, num_rois, mask_shape, mask_shape, mask_shape, 1]; integer of [0,1]
        part_ids: [batch_size, num_rois, 1]; (the predicted part label of the proposals)

        output: pred masks: [batch_size, num_rois, mask_shape, mask_shape, mask_shape, part_num], real num [0~1]
    """
    batch_size = proposals.get_shape()[0]
    rois_num = proposals.get_shape()[1]
    mask_shape = proposals.get_shape()[2]
    channel_num = proposals.get_shape()[5]
    input_batch = tf.reshape(proposals,
            [-1, mask_shape, mask_shape, mask_shape, channel_num])
    en1 = tf_util.conv3d(input_batch, 16, kernel_size=[3,3,3],
                         padding='SAME', stride=[2,2,2], bn=True,
                         is_training=is_training, scope='gen_conv1',
                         bn_decay=bn_decay, weight_decay=weight_decay)
    # (b*num_rois, 16,16,16,1)-->(b*num_rois, 8,8,8,16)
    # en1 = tf_util.max_pool3d(en1, kernel_size=[3,3,3], padding='SAME', scope='gen_pool1')

    en2 = tf_util.conv3d(en1, 32, kernel_size=[3,3,3],
                         padding='SAME', stride=[2,2,2], bn=True,
                         is_training=is_training, scope='gen_conv2',
                         bn_decay=bn_decay, weight_decay=weight_decay)
    # (b*num_rois, 8,8,8,16)-->(b*num_rois, 4,4,4,32)
    # en2 = tf_util.max_pool3d(en2, kernel_size=[3,3,3], padding='SAME', scope='gen_pool2')

    en3 = tf_util.conv3d(en2, 32, kernel_size=[2,2,2],
                         padding='SAME', stride=[2,2,2], bn=True,
                         is_training=is_training, scope='gen_conv3',
                         bn_decay=bn_decay, weight_decay=weight_decay)
    # (b*num_rois, 4,4,4,32)-->(b*num_rois, 2,2,2,32)
    # en3 = tf_util.max_pool3d(en3, kernel_size=[2,2,2], padding='SAME', scope='gen_pool3')

    en4 = tf_util.conv3d(en3, 64, kernel_size=[2,2,2],
                         padding='VALID', stride=[2,2,2], bn=True,
                         is_training=is_training, scope='gen_conv4',
                         bn_decay=bn_decay, weight_decay=weight_decay)
    # (b*num_rois, 2,2,2,32)-->(b*num_rois, 1,1,1,64)
    # en4 = tf_util.max_pool3d(en4, kernel_size=[2,2,2], padding='SAME', scope='gen_pool4')

    # Decoding phase
    de1 = en4
    de1 = tf_util.conv3d_transpose(de1, 32, kernel_size=[2,2,2],
                                   padding='SAME', stride=[2,2,2], bn=True,
                                   is_training=is_training, scope='gen_deconv_1',
                                   bn_decay=bn_decay, weight_decay=weight_decay)
    de1 = tf_util.dropout(de1, keep_prob=0.7, is_training=is_training, scope='gen_drop_1')
    de1 = tf.concat(axis=-1, values=[de1, en3])  # (b*n,1,1,1,64)-->(b*n,2,2,2,32*2)

    de2 = tf_util.conv3d_transpose(de1, 32, kernel_size=[2,2,2],
                                   padding='SAME', stride=[2,2,2], bn=True,
                                   is_training=is_training, scope='gen_deconv_2',
                                   bn_decay=bn_decay, weight_decay=weight_decay)
    de2 = tf_util.dropout(de2, keep_prob=0.7, is_training=is_training, scope='gen_drop_2')
    de2 = tf.concat(axis=-1, values=[de2, en2])  # (b*n,2,2,2,32*2)-->(b*n,4,4,4,32*2)

    de3 = tf_util.conv3d_transpose(de2, 32, kernel_size=[3,3,3],
                                   padding='SAME', stride=[2,2,2], bn=True,
                                   is_training=is_training, scope='gen_deconv_3',
                                   bn_decay=bn_decay, weight_decay=weight_decay)
    de3 = tf_util.dropout(de3, keep_prob=0.7, is_training=is_training, scope='gen_drop_3')
    de3 = tf.concat(axis=-1, values=[de3, en1])  # (b*n,4,4,4,32*2)-->(b*n,8,8,8,32+16)

    de4 = tf_util.conv3d_transpose(de3, 32, kernel_size=[3,3,3],
                                   padding='SAME', stride=[2,2,2], bn=True,
                                   is_training=is_training, scope='gen_deconv_4',
                                   bn_decay=bn_decay, weight_decay=weight_decay)
    # (b*n,8,8,8,32+16)-->(b*n,16,16,16,32)
    # ## fully connected pred layer
    net = tf_util.conv3d(de4, part_num, kernel_size=[1,1,1],
                         padding='SAME', stride=[1,1,1], bn=False,
                         activation_fn=tf.nn.sigmoid, is_training=is_training,
                         scope='gen_mask')
    """ Masks [batch,roi_count,height,width,depth,num_classes] """
    mask = tf.reshape(net,
                      [batch_size, rois_num,
                      net.get_shape()[1], net.get_shape()[2], net.get_shape()[3],
                      net.get_shape()[-1]])
    return mask


def create_generator(proposals, part_num,
                     is_training, bn_decay=None, weight_decay=0.0):
    """ ref Hao Zhi Xiang, CAB, RRB, Atrous Conv
    input proposals of possible part region, generate its precise mask
        proposals: [batch_size, num_rois, mask_shape, mask_shape, mask_shape, 1]; integer of [0,1]
        part_ids: [batch_size, num_rois, 1]; (the predicted part label of the proposals)

        output: pred masks: [batch_size, num_rois, mask_shape, mask_shape, mask_shape, part_num], real num [0~1]
    """
    batch_size = proposals.get_shape()[0]
    rois_num = proposals.get_shape()[1]
    mask_shape = proposals.get_shape()[2]
    channel_num = proposals.get_shape()[5]
    input_batch = tf.reshape(proposals,
            [-1, mask_shape, mask_shape, mask_shape, channel_num])
    end_points = {}
    # Encoding phase
    conv = tf_util.conv3d_atrous(
            inputs=input_batch,
            num_output_channels=16,
            kernel_size=[3,3,3],
            stride=[1, 1, 1],
            padding='same',
            dilation_rate=1,
            is_training=is_training,
            bn=True,
            scope='atrous_1')
    end_points['atrous_conv1'] = conv

    conv = tf_util.conv3d_atrous(
            inputs=conv,
            num_output_channels=32,
            kernel_size=[3,3,3],
            stride=[1, 1, 1],
            padding='same',
            dilation_rate=2,
            is_training=is_training,
            bn=True,
            scope='atrous_2')
    end_points['atrous_conv2'] = conv

    conv = tf_util.conv3d_atrous(
            inputs=conv,
            num_output_channels=32,
            kernel_size=[3,3,3],
            stride=[1, 1, 1],
            padding='same',
            dilation_rate=4,
            is_training=is_training,
            bn=True,
            scope='atrous_3')
    end_points['atrous_conv3'] = conv

    conv = tf_util.conv3d_atrous(
            inputs=conv,
            num_output_channels=64,
            kernel_size=[3,3,3],
            stride=[1, 1, 1],
            padding='same',
            dilation_rate=4,
            is_training=is_training,
            bn=True,
            scope='atrous_4')
    end_points['atrous_conv4'] = conv

    features = conv
    # Decoding phase
    decoded = tf_util.decoder(features, end_points, is_training,
            scope='gen_decoder', num_part=part_num)

    net = tf_util.conv3d(decoded, 64, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=True,
                is_training=is_training, scope='pred_1')
    net = tf_util.conv3d(net, 64, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=True,
                is_training=is_training, scope='pred_2')
    predicted = tf_util.conv3d(net, part_num, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=False, activation_fn=tf.nn.sigmoid,
                is_training=is_training, scope='pred')
    # net = tf_util.rrb(decoded, None, is_training, 'rrb_pred')
    # predicted = tf_util.conv3d(net, part_num, kernel_size=[1,1,1],
    #             padding='SAME', stride=[1,1,1], bn=False,
    #             activation_fn=tf.nn.sigmoid,
    #             is_training=is_training, scope='gen_mask')

    """ Masks [batch,roi_count,height,width,depth,num_classes] """
    mask = tf.reshape(predicted,
            [batch_size, rois_num,
            predicted.get_shape()[1],
            predicted.get_shape()[2],
            predicted.get_shape()[3],
            predicted.get_shape()[-1]]
           )

    return mask


def get_valid_pred_mask(mask, indice):
    """ return (num_positive, mask_shape**3)"""
    return tf.slice(mask, [indice[0], indice[1], 0, 0, 0], [1,1,-1,-1,-1])


def get_valid_gt_mask(mask, indice):
    """ return (num_positive, mask_shape**3)"""
    return tf.slice(mask, [indice, 0, 0, 0], [1,-1,-1,-1])


def create_discriminator(masks, part_ids, part_num,
                         is_training, is_real, bn_decay=None, weight_decay=0.0):
    """ input (gen part mask); not Pair data
            proposals: [batch_size, num_rois, mask_shape, mask_shape, mask_shape, 1]
            part_mask :
                gt mask (batch, rois_num, mask_shape**3),
                pred mask (batch, rois_num, mask_shape**3, part_num)
            part_ids: (batch, rois_num)
        output if its a real pair: probability(pair is real) = [0~1]
            num_input: the valid data num, without zero paddings
    """
    masks_shape = masks.get_shape()
    batch_size = masks_shape[0]
    rois_num = masks_shape[1]

    # Reshape for simplicity. Merge first two dimensions into one.
    part_ids = tf.reshape(part_ids, [-1])
    positive_ix = tf.range(0,batch_size*rois_num,1)  # all idxs
    # positive_ix = tf.where(part_ids > 0)[:, 0]
    # only positive rois contribute to the loss, and only the class
    # specific mask of each ROI
    positive_class_ids = tf.cast(
        tf.gather(part_ids, positive_ix), tf.int32)  # int64
    tmp_ids = tf.maximum(positive_class_ids - 1, tf.zeros(part_ids.get_shape(), tf.int32))
    indices = tf.stack([positive_ix, tmp_ids], axis=1)

    def get_discrim(masks):
        # pdb.set_trace()
        pred_shape = masks.get_shape()
        pred_masks = tf.reshape(masks,
                [-1, pred_shape[2], pred_shape[3], pred_shape[4], pred_shape[5]])
        pred_masks = tf.transpose(pred_masks, [0,4,1,2,3])
        outputs = []
        for m in range(batch_size*rois_num):
            indice = indices[m]
            cur_pred_mask = tf.squeeze(
                    tf.slice(pred_masks,
                            [indice[0],indice[1],0,0,0], [1,1,-1,-1,-1]))
            outputs.append(tf.expand_dims(tf.expand_dims(cur_pred_mask, 0),-1))
        discrim_targets = tf.concat(outputs, axis=0)
        return tf.cast(discrim_targets, tf.float32)

    discrim_targets = get_discrim(masks)
    # pdb.set_trace()
    input_batch = discrim_targets

    # ########### network structure
    en1 = tf_util.conv3d(input_batch, 32, kernel_size=[3,3,3],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='dis_conv1',
                         bn_decay=bn_decay, weight_decay=weight_decay)
    # (b*num_rois, 16,16,16,1)

    en2 = tf_util.conv3d(en1, 32, kernel_size=[3,3,3],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='dis_conv2',
                         bn_decay=bn_decay, weight_decay=weight_decay)
    en2 = tf_util.max_pool3d(en2, kernel_size=[3,3,3], padding='SAME', scope='dis_pool2')
    # (b*num_rois, 8,8,8,16)

    en3 = tf_util.conv3d(en2, 64, kernel_size=[3,3,3],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='dis_conv3',
                         bn_decay=bn_decay, weight_decay=weight_decay)
    en3 = tf_util.max_pool3d(en3, kernel_size=[3,3,3], padding='SAME', scope='dis_pool3')
    # (b*num_rois, 4,4,4,16)

    en4 = tf_util.conv3d(en3, 64, kernel_size=[2,2,2],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='dis_conv4',
                         bn_decay=bn_decay, weight_decay=weight_decay)
    # (b*num_rois, 4,4,4,16)

    pred = tf_util.conv3d(en4, part_num, kernel_size=[4,4,4],
                         padding='VALID', stride=[1,1,1], bn=False,
                         is_training=is_training, scope='dis_conv5',
                         activation_fn=tf.nn.sigmoid)  # (b*n, 4,4,4,part_num)

    return pred


def create_discriminator_v0(proposals, masks, part_ids,
                         is_training, is_real, bn_decay=None, weight_decay=0.0):
    """ input (proposal voxel, part mask) pair; Pair data
            proposals: [batch_size, num_rois, mask_shape, mask_shape, mask_shape, 1]
            part_mask :
                gt mask (batch, rois_num, mask_shape**3),
                pred mask (batch, rois_num, mask_shape**3, part_num)
            part_ids: (batch, rois_num)
        output if its a real pair: probability(pair is real) = [0~1]
            num_input: the valid data num, without zero paddings
    """
    proposal_shape = proposals.get_shape()
    batch_size = proposal_shape[0]
    rois_num = proposal_shape[1]
    vox_size = proposal_shape[2]
    # Reshape for simplicity. Merge first two dimensions into one.
    part_ids = tf.reshape(part_ids, [-1])
    positive_ix = tf.range(0,batch_size*rois_num,1)  # all idxs
    # positive_ix = tf.where(part_ids > 0)[:, 0]
    # only positive rois contribute to the loss, and only the class
    # specific mask of each ROI
    positive_class_ids = tf.cast(
        tf.gather(part_ids, positive_ix), tf.int32)  # int64
    tmp_ids = tf.maximum(positive_class_ids - 1, tf.zeros(part_ids.get_shape(), tf.int32))
    indices = tf.stack([positive_ix, tmp_ids], axis=1)

    def get_pred_discrim(masks):
        # pdb.set_trace()
        pred_shape = masks.get_shape()
        pred_masks = tf.reshape(masks,
                [-1, pred_shape[2], pred_shape[3], pred_shape[4], pred_shape[5]])
        pred_masks = tf.transpose(pred_masks, [0,4,1,2,3])
        fn = lambda indice: get_valid_pred_mask(pred_masks, indice)
        discrim_targets = tf.squeeze(tf.map_fn(fn, elems=(indices), dtype=tf.float32))
        discrim_targets = tf.expand_dims(discrim_targets, axis=-1)
        return tf.cast(discrim_targets, tf.float32)

    def get_target_discrim(masks):
        mask_shape = masks.get_shape()
        gt_masks = tf.reshape(masks,
                [-1, mask_shape[2], mask_shape[3], mask_shape[4]])
        # fn_gt = lambda indice: get_valid_gt_mask(gt_masks, indice)
        # discrim_targets = tf.squeeze(tf.map_fn(fn_gt, elems=(positive_ix)))
        discrim_targets = gt_masks
        discrim_targets = tf.expand_dims(discrim_targets, axis=-1)
        return discrim_targets

    if is_real:
        discrim_targets = get_target_discrim(masks)
    else:
        discrim_targets = get_pred_discrim(masks)

    discrim_inputs = tf.reshape(proposals,
            [-1, vox_size, vox_size, vox_size, 1])
    # pdb.set_trace()
    input_batch = tf.concat([discrim_inputs, discrim_targets], axis=-1)

    # ########### network structure
    en1 = tf_util.conv3d(input_batch, 32, kernel_size=[3,3,3],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='dis_conv1',
                         bn_decay=bn_decay, weight_decay=weight_decay)
    # (b*num_rois, 16,16,16,1)

    en2 = tf_util.conv3d(en1, 32, kernel_size=[3,3,3],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='dis_conv2',
                         bn_decay=bn_decay, weight_decay=weight_decay)
    en2 = tf_util.max_pool3d(en2, kernel_size=[3,3,3], padding='SAME', scope='dis_pool2')
    # (b*num_rois, 8,8,8,16)

    en3 = tf_util.conv3d(en2, 64, kernel_size=[3,3,3],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='dis_conv3',
                         bn_decay=bn_decay, weight_decay=weight_decay)
    en3 = tf_util.max_pool3d(en3, kernel_size=[3,3,3], padding='SAME', scope='dis_pool3')
    # (b*num_rois, 4,4,4,16)

    en4 = tf_util.conv3d(en3, 64, kernel_size=[2,2,2],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='dis_conv4',
                         bn_decay=bn_decay, weight_decay=weight_decay)
    # (b*num_rois, 4,4,4,16)

    net = tf_util.conv3d(en4, 1, kernel_size=[1,1,1],
                         padding='SAME', stride=[1,1,1], bn=False,
                         is_training=is_training, scope='dis_conv5',
                         activation_fn=tf.nn.sigmoid)  # (b*n, 4,4,4,1)
    return net


def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (x1, y1, z1, x2, y2, z2)].
    """
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 6])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_x1, b1_y1, b1_z1, b1_x2, b1_y2, b1_z2 = tf.split(b1, 6, axis=1)
    b2_x1, b2_y1, b2_z1, b2_x2, b2_y2, b2_z2 = tf.split(b2, 6, axis=1)
    z1 = tf.maximum(b1_z1, b2_z1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    z2 = tf.minimum(b1_z2, b2_z2)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2-x1, 0) * tf.maximum(y2-y1, 0) * tf.maximum(z2-z1, 0)
    # 3. Compute unions
    b1_volume = (b1_z2 - b1_z1) * (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_volume = (b2_z2 - b2_z1) * (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_volume + b2_volume - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def trim_zeros_graph(boxes, name=None):
    """Often boxes are represented with matricies of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def get_target_rois(rois, part_ids, gt_rois, gt_partids):
    """ get target rois, which is gt_rois correspond to rois """
    batch_size = rois.get_shape()[0]
    rois_num = rois.get_shape()[1]
    target_rois_list = []
    target_pids_list = []
    for b in range(batch_size):
        cur_roi, non_zeros = trim_zeros_graph(rois[b], name="trim_roi_"+str(b))
        cur_roi_gt, non_zeros_gt = trim_zeros_graph(gt_rois[b], name="trim_gt"+str(b))
        cur_partid = tf.boolean_mask(part_ids[b], non_zeros)
        cur_partid_gt = tf.boolean_mask(gt_partids[b], non_zeros_gt)

        valid_idx = tf.where(non_zeros)[:, 0]
        valid_gtidx = tf.where(non_zeros_gt)[:, 0]
        cur_roi_ = tf.gather(cur_roi, valid_idx)
        cur_partid_ = tf.gather(cur_partid, valid_idx)
        cur_roi_gt_ = tf.gather(cur_roi_gt, valid_gtidx)
        cur_partid_gt_ = tf.gather(cur_partid_gt, valid_gtidx)
        cur_target_roi_ = cur_roi_gt_
        cur_target_pid_ = cur_partid_gt_
        # cur_target_roi_ = tf.concat([cur_roi_gt_, cur_roi_], axis=0)
        # cur_target_pid_ = tf.concat([cur_partid_gt_, cur_partid_], axis=0)
        # # Compute overlaps matrix [proposals, gt_boxes]
        # pdb.set_trace()
        # overlaps = overlaps_graph(cur_roi_, cur_roi_gt_)
        # # roi_iou_max = tf.reduce_max(overlaps, axis=1)
        # gt_max_idx = tf.argmax(overlaps, axis=1)
        # cur_target_roi_ = tf.gather(cur_roi_gt_, gt_max_idx)
        # cur_target_pid_ = tf.gather(cur_partid_gt_, gt_max_idx)
        # Append and pad bbox and pid that are not used with zeros
        cur_target_roi = tf.pad(cur_target_roi_, [(0,rois_num),(0,0)])
        cur_target_roi = tf.slice(cur_target_roi, [0,0], [rois_num,-1])  # [rois_num,6]
        cur_target_pid = tf.pad(cur_target_pid_, [(0,rois_num)])
        cur_target_pid = tf.slice(cur_target_pid, [0,], [rois_num,])

        target_rois_list.append(tf.expand_dims(cur_target_roi, axis=0))
        target_pids_list.append(tf.expand_dims(cur_target_pid, axis=0))
    target_rois = tf.concat(target_rois_list, axis=0)
    target_pids = tf.concat(target_pids_list, axis=0)
    return target_rois, target_pids


def create_gan_model(rois, part_ids, feature_map, gt_seg,
                     gt_rois, gt_partids,
                     pool_size, part_num,
                     is_training, argparse, global_step,
                     bn_decay=None, weight_decay=0.0):
    """ create GAN model, with loss definitions;
    WZJ:20180510: change real discriminator training(using gt rois)
    for roi pooling:
    INPUTS: roi boxes: (batch, num_rois, (x1,y1,z1,x2,y2,z2))
            input_fea: (batch, 48,48,48,channels), from Unet output
            pool_size:  16, mask_shape
            gt_seg_vox: (batch_size, 48, 48, 48)
            part_ids: for roi boxes, (batch_size, num_rois, 1)
    OUTPUT: proposals: (batch, num_rois, 16, 16, 16, channels)
            targets: (batch_size, num_rois, 16, 16, 16, 1), gt mask
    """
    batch_size = rois.get_shape()[0]
    rois_num = rois.get_shape()[1]
    # ### RoI pooling, and get targets
    proposals = roi_pooling(rois, feature_map, pool_size)  # (batch, num_rois, 16, 16, 16, channels)
    if feature_map.get_shape()[4] == 1:
        proposals = tf.expand_dims(proposals, -1)

    # get targets
    gt_masks = []
    for b in range(batch_size):
        cur_gt_seg = tf.slice(gt_seg,
                            [b,0,0,0], [1,-1,-1,-1])
        masks = []
        for part_label in range(part_num):
            cur_part_id = part_label + 1
            cur_mask = tf.cast(tf.equal(cur_gt_seg, cur_part_id), dtype=tf.float32)
            masks.append(tf.expand_dims(cur_mask, axis=-1))  # [48,48,48,part_num]
        cur_masks = tf.concat(masks, axis=-1)  # [1,48,48,48,part_num]
        gt_masks.append(cur_masks)
    gt_seg_vox = tf.concat(gt_masks, axis=0)  # [batch, 48,48,48,part_num]
    # for L1 loss
    targets_crop = roi_pooling(rois, gt_seg_vox, pool_size)  # (batch, num_rois, 16, 16, 16, part_num)
    # for discriminator real:
    target_rois, target_pids = get_target_rois(rois, part_ids, gt_rois, gt_partids)
    # pdb.set_trace()
    targets = roi_pooling(target_rois, gt_seg_vox, pool_size)

    # output pred mask: [batch_size, num_rois, 16, 16, 16, part_num]

    # for remove zero paddings
    a = argparse
    part_ids = tf.reshape(part_ids, [-1])
    positive_ix = tf.where(part_ids > 0)[:, 0]
    # only positive rois contribute to the loss, and only the class
    # specific mask of each ROI
    positive_class_ids = tf.cast(
        tf.gather(part_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids-1], axis=1)

    target_pids = tf.reshape(target_pids, [-1])
    positive_ix_target = tf.where(target_pids > 0)[:, 0]
    positive_class_ids_target = tf.cast(
        tf.gather(target_pids, positive_ix_target), tf.int64)
    indices_target = tf.stack([positive_ix_target, positive_class_ids_target-1], axis=1)

    with tf.variable_scope("generator"):
        # create_generator_v0 for Unet_gen
        outputs = create_generator(proposals, part_num,
                 is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("discriminator_real"):
        with tf.variable_scope("discriminator"):
            # 2x [batch*n_rois,mask_shape**3, n_ch]=>[batch*n_rois, 4,4,4,part_num]
            predict_real = create_discriminator(targets, target_pids, part_num,
                    is_training=is_training, is_real=True, bn_decay=bn_decay, weight_decay=weight_decay)

    # varslist = tf.all_variables()
    # for v in varslist:
    #     print('var name is: ', v.name)
    # pdb.set_trace()
    # discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
    with tf.name_scope("discriminator_fake"):
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):  # reuse=True):
            # 2x [batch*n_rois,mask_shape**3, n_ch]=>[batch*n_rois, 4,4,4,part_num]
            # default: is_real=False
            predict_fake = create_discriminator(outputs, part_ids, part_num,
                    is_training=is_training, is_real=False, bn_decay=bn_decay, weight_decay=weight_decay)

    # # get valid pred, without zero padding
    predict_real_ = tf.transpose(predict_real, [0,4,1,2,3])
    predict_fake_ = tf.transpose(predict_fake, [0,4,1,2,3])
    pred_real_valid = tf.gather_nd(predict_real_, indices_target)
    pred_fake_valid = tf.gather_nd(predict_fake_, indices)
    # pred_real_valid = tf.gather(predict_real,positive_ix_target)
    # pred_fake_valid = tf.gather(predict_fake,positive_ix)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        # discrim_loss = tf.reduce_mean(-(tf.log(pred_real_valid + EPS) +
        #         tf.log(1 - pred_fake_valid + EPS)))
        discrim_loss = tf.reduce_mean(-tf.log(pred_real_valid + EPS)) +\
                      tf.reduce_mean(-tf.log(1 - pred_fake_valid + EPS))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0

        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        target_shape = targets_crop.get_shape()
        targets_ = tf.reshape(targets_crop,
                [-1,target_shape[2],target_shape[3],target_shape[4],target_shape[5]])
        targets_ = tf.transpose(targets_, [0,4,1,2,3])
        output_shape = outputs.get_shape()
        outputs_ = tf.reshape(outputs,
                [-1,output_shape[2],output_shape[3],output_shape[4],output_shape[5]])
        outputs_ = tf.transpose(outputs_, [0,4,1,2,3])
        y_true = tf.gather_nd(targets_, indices)
        y_out = tf.gather_nd(outputs_, indices)

        gen_loss_GAN = tf.reduce_mean(-tf.log(pred_fake_valid + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(y_true - y_out))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    # pdb.set_trace()
    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):  # make sure discrim_train updated first
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=a.decay_rate_GAN)  # 0.99
    # update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    update_dis_loss = ema.apply([discrim_loss])
    update_gen_loss = ema.apply([gen_loss_GAN, gen_loss_L1])

    # global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    # tf.mod(global_step,3)
    train_op = tf.cond(tf.equal(tf.mod(global_step,4), 0),
        lambda: tf.group(update_dis_loss, incr_global_step, discrim_train),
        lambda: tf.group(update_gen_loss, incr_global_step, gen_train))

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=train_op,  # tf.group(update_losses, incr_global_step, gen_train),
        global_step=global_step,
        proposals=proposals,
    )


def create_gan_model_v1(rois, part_ids, feature_map, gt_seg, pool_size,
                     part_num,
                     is_training, argparse, global_step,
                     bn_decay=None, weight_decay=0.0):
    """ create GAN model, with loss definitions;
    for roi pooling:
    INPUTS: roi boxes: (batch, num_rois, (x1,y1,z1,x2,y2,z2))
            input_fea: (batch, 48,48,48,channels), from Unet output
            pool_size:  16, mask_shape
            gt_seg_vox: (batch_size, 48, 48, 48)
            part_ids: for roi boxes, (batch_size, num_rois, 1)
    OUTPUT: proposals: (batch, num_rois, 16, 16, 16, channels)
            targets: (batch_size, num_rois, 16, 16, 16, 1), gt mask
    """
    batch_size = rois.get_shape()[0]
    rois_num = rois.get_shape()[1]
    # ### RoI pooling, and get targets
    proposals = roi_pooling(rois, feature_map, pool_size)  # (batch, num_rois, 16, 16, 16, channels)
    if feature_map.get_shape()[4] == 1:
        proposals = tf.expand_dims(proposals, -1)

    # get targets

    gt_masks = []
    for b in range(batch_size):
        cur_gt_seg = tf.slice(gt_seg,
                            [b,0,0,0], [1,-1,-1,-1])
        masks = []
        for part_label in range(part_num):
            cur_part_id = part_label + 1
            cur_mask = tf.cast(tf.equal(cur_gt_seg, cur_part_id), dtype=tf.float32)
            masks.append(tf.expand_dims(cur_mask, axis=-1))  # [48,48,48,part_num]
        cur_masks = tf.concat(masks, axis=-1)  # [1,48,48,48,part_num]
        gt_masks.append(cur_masks)
    gt_seg_vox = tf.concat(gt_masks, axis=0)  # [batch, 48,48,48,part_num]
    targets = roi_pooling(rois, gt_seg_vox, pool_size)  # (batch, num_rois, 16, 16, 16, part_num)
    # output pred mask: [batch_size, num_rois, 16, 16, 16, part_num]

    # for remove zero paddings
    a = argparse
    part_ids = tf.reshape(part_ids, [-1])
    positive_ix = tf.where(part_ids > 0)[:, 0]
    # only positive rois contribute to the loss, and only the class
    # specific mask of each ROI
    positive_class_ids = tf.cast(
        tf.gather(part_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids-1], axis=1)

    with tf.variable_scope("generator"):
        outputs = create_generator(proposals, part_num,
                 is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("discriminator_real"):
        with tf.variable_scope("discriminator"):
            # 2x [batch*n_rois,mask_shape**3, n_ch]=>[batch*n_rois, 4,4,4,part_num]
            predict_real = create_discriminator(targets, part_ids, part_num,
                    is_training=is_training, is_real=True, bn_decay=bn_decay, weight_decay=weight_decay)

    # varslist = tf.all_variables()
    # for v in varslist:
    #     print('var name is: ', v.name)
    # pdb.set_trace()
    # discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
    with tf.name_scope("discriminator_fake"):
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):  # reuse=True):
            # 2x [batch*n_rois,mask_shape**3, n_ch]=>[batch*n_rois, 4,4,4,part_num]
            # default: is_real=False
            predict_fake = create_discriminator(outputs, part_ids, part_num,
                    is_training=is_training, is_real=False, bn_decay=bn_decay, weight_decay=weight_decay)

    # get valid pred, without zero padding
    predict_real_ = tf.transpose(predict_real, [0,4,1,2,3])
    predict_fake_ = tf.transpose(predict_fake, [0,4,1,2,3])
    pred_real_valid = tf.gather_nd(predict_real_, indices)
    pred_fake_valid = tf.gather_nd(predict_fake_, indices)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(pred_real_valid + EPS) +
                tf.log(1 - pred_fake_valid + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0

        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        target_shape = targets.get_shape()
        targets_ = tf.reshape(targets,
                [-1,target_shape[2],target_shape[3],target_shape[4],target_shape[5]])
        targets_ = tf.transpose(targets_, [0,4,1,2,3])
        output_shape = outputs.get_shape()
        outputs_ = tf.reshape(outputs,
                [-1,output_shape[2],output_shape[3],output_shape[4],output_shape[5]])
        outputs_ = tf.transpose(outputs_, [0,4,1,2,3])
        y_true = tf.gather_nd(targets_, indices)
        y_out = tf.gather_nd(outputs_, indices)

        gen_loss_GAN = tf.reduce_mean(-tf.log(pred_fake_valid + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(y_true - y_out))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    # pdb.set_trace()
    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):  # make sure discrim_train updated first
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=a.decay_rate_GAN)  # 0.99
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    # global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
        global_step=global_step,
    )


def create_gan_model_v0(proposals, targets, part_ids, part_num,
                     is_training, argparse, global_step,
                     bn_decay=None, weight_decay=0.0):
    """ create GAN model, with loss definitions;"""
    # for remove zero paddings
    a = argparse
    part_ids = tf.reshape(part_ids, [-1])
    positive_ix = tf.where(part_ids > 0)[:, 0]
    # only positive rois contribute to the loss, and only the class
    # specific mask of each ROI
    positive_class_ids = tf.cast(
        tf.gather(part_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids-1], axis=1)

    with tf.variable_scope("generator"):
        outputs = create_generator(proposals, part_num,
                 is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("discriminator_real"):
        with tf.variable_scope("discriminator"):
            # 2x [batch*n_rois,mask_shape**3, n_ch]=>[batch*n_rois, 4,4,4,1]
            predict_real = create_discriminator(targets, part_ids,
                    is_training=is_training, is_real=True, bn_decay=bn_decay, weight_decay=weight_decay)

    # varslist = tf.all_variables()
    # for v in varslist:
    #     print('var name is: ', v.name)
    # pdb.set_trace()
    # discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
    with tf.name_scope("discriminator_fake"):
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):  # reuse=True):
            # 2x [batch*n_rois,mask_shape**3, n_ch]=>[batch*n_rois, 4,4,4,1]
            # default: is_real=False
            predict_fake = create_discriminator(outputs, part_ids,
                    is_training=is_training, is_real=False, bn_decay=bn_decay, weight_decay=weight_decay)

    # get valid pred, without zero padding
    pred_real_valid = tf.gather(predict_real, positive_ix)
    pred_fake_valid = tf.gather(predict_fake, positive_ix)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(pred_real_valid + EPS) +
                tf.log(1 - pred_fake_valid + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0

        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        target_shape = targets.get_shape()
        targets = tf.reshape(targets,
                [-1,target_shape[2],target_shape[3],target_shape[4]])
        output_shape = outputs.get_shape()
        outputs_ = tf.reshape(outputs,
                [-1,output_shape[2],output_shape[3],output_shape[4],output_shape[5]])
        outputs_ = tf.transpose(outputs_, [0,4,1,2,3])
        y_true = tf.gather(targets, positive_ix)
        y_out = tf.gather_nd(outputs_, indices)

        gen_loss_GAN = tf.reduce_mean(-tf.log(pred_fake_valid + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(y_true - y_out))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    # pdb.set_trace()
    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):  # make sure discrim_train updated first
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=a.decay_rate_GAN)  # 0.99
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    # global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
        global_step=global_step,
    )


############################################################
#  main(), for graph validity test
############################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
    parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
    parser.add_argument('--decay_rate_GAN', type=float, default=0.99, help='Decay rate for lr decay [default: 0.8]')
    parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
    parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
    Args = parser.parse_args()
    with tf.Graph().as_default():
        inputs = tf.ones((32, 64, 64, 64, 1))  # dtype default is tf.float32
        seg_labels = tf.ones((32, 64, 64, 64))
        pred_labels = tf.ones((32, 64, 64, 64, 6))
        # input_ph, seg_ph = placeholder_inputs(32, 64)
        # outputs = get_model(input_ph, tf.constant(True))
        # loss = get_loss(outputs, seg_ph)
        outputs, features = build_Unet_FCN(inputs, tf.constant(True), 6)
        loss, batch_seg_loss, _ = Unet_FCN_loss(outputs, seg_labels, 6)
        proposals = tf.zeros((32,15,16,16,16,1))
        targets = tf.zeros((32,15,16,16,16), tf.float32)
        part_ids = tf.zeros((32,15))
        rois = tf.zeros((32,15,6))
        # model = create_gan_model(rois, part_ids, features, seg_labels, 16,
        #             6, tf.constant(True), argparse=Args, global_step=tf.Variable(0))
        boxes, partids = build_bbox_graph(inputs, pred_labels, 2)
        sess = tf.Session()
        out = sess.run([boxes])
        print(out)
        # bbox_res, partid_res = sess.run([boxes, partids])
        # pdb.set_trace()
        # A = np.random.rand(32,32,32,32,1)
        # B = np.random.randint(0, 49, [32,32,32,32], np.uint8)
        # feed_dict = {input_ph: A, seg_ph: B}
        # pdb.set_trace()
        # loss_val, output_val, input_val, seg_val = sess.run([loss, outputs, input_ph, seg_ph], feed_dict=feed_dict)
        # print('good')
        # print('over')
    print('outputs\n', outputs, '\n', 'loss\n', loss)
    print('batch_seg_loss\n', batch_seg_loss)
    print('rois: ', boxes, partids)
