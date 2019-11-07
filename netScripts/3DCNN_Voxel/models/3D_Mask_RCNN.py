""" 3DCNN encode decode U net
    mimick O-CNN segmentation network---20180120
    MASK RCNN: use 3D mask for parts,
    ---20180322---
    #---Ver 0.1---: input gt roi and mask, to check model efficiency
"""
import tensorflow as tf
import numpy as np
import os
import sys
import pdb
import multiprocessing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
import gen_part_box as Box_util
import mask_data_prepare as Mask_util

Common_SET_DIR = os.path.join(BASE_DIR, '../../CommonFile')
sys.path.append(Common_SET_DIR)
import globals as g_

############################################################
#  3DCNN Unet Graph
############################################################

def placeholder_Unet_inputs(batch_size, vox_size):
    vol_ph = tf.placeholder(tf.float32, shape=(batch_size, vox_size, vox_size, vox_size, 1))
    seg_ph = tf.placeholder(tf.float32, shape=(batch_size, vox_size, vox_size, vox_size))
    return vol_ph, seg_ph


def build_Unet_FCN(volumes, is_training, part_num, bn_decay=None, weight_decay=0.0):
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


def crop_and_resize_data(data, boxes, crop_size=15):
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
    boxes: rois factor, shape=(batch, num_rois, (x1,y1,z1,x2,y2,z2))
    inputs: feature map, shape=(batch, 48,48,48, channels)
    pool_size: 15 (output is [batch,15,15,15,15,n_ch])

    Return: pooled regions in the shape: (batch, num_boxes, h,w,d,channels)
        might be zero padded if not enough target ROIs. maybe use tf.map_fn()
    """

    batch_size = inputs.get_shape()[0]
    crop_batch = []
    for b in range(batch_size):
        fea = tf.expand_dims(inputs[b], axis=0)
        rois = boxes[b]
        fn = lambda bboxes: crop_and_resize_data(fea, bboxes, pool_size)
        crop_datas = tf.squeeze(tf.map_fn(fn, elems=(rois)))
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


def get_roi_func(vox_datas, pred_segs, Max_Ins_Num):
    """ np function
        input a batch of voxels and gt_segs, pred_segs
        compute boxes (rois factors)
        output: rois
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
        Boxes, BLabels, _, _ = Box_util.computeBox(pts=pts, plb=plb, alpha=1.5)
        boxes, part_ids = Mask_util.gen_rois_info(Boxes, BLabels)
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


def _compute_roi(vox, seg):
    cur_vox = np.asarray(vox)
    cur_pred_seg = np.asarray(seg)
    Max_Ins_Num = g_.MAX_INS_NUM
    info = {}
    pts = np.transpose(np.array(np.where(cur_vox)))
    plb = cur_pred_seg[pts[:,0], pts[:,1], pts[:,2]]
    pts = pts.astype(float)
    Boxes, BLabels, _, _ = Box_util.computeBox(pts=pts, plb=plb, alpha=1.5)
    boxes, part_ids = Box_util.gen_rois_info(Boxes, BLabels)
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
    info['roi'] = Bbox_data
    info['part_id'] = Label_data
    return info


def get_roi_func_fast(vox_datas, pred_segs, Max_Ins_Num):
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
        pool_list.append(pool.apply_async(_compute_roi, args=(sub_vox,sub_pred_seg)))
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


def build_bbox_graph(volumes, seg_pred, Max_Ins_Num):
    """
        volumes:  (batch_size, vox_size, vox_size, vox_size, 1)
        seg_pred: (batch_size, vox_size, vox_size, vox_size, part_num)
    """
    vol_shape = volumes.get_shape()
    seg_pred_ = tf.cast(tf.argmax(seg_pred, axis=-1) + 1, tf.float32)  # (batch_size, vox_size, vox_size, vox_size)

    boxes, part_ids = tf.py_func(get_roi_func_fast,
            inp=[volumes, seg_pred_, Max_Ins_Num],
            Tout=[tf.float32,tf.float32])
    # part_ids = tf.py_func(get_roi_func,
    #         inp=[volumes, seg_pred_, Max_Ins_Num],
    #         Tout=[tf.float32,tf.float32])[1]
    # pdb.set_trace()
    boxes = tf.reshape(boxes, [vol_shape[0],Max_Ins_Num,6])
    part_ids = tf.reshape(part_ids, [vol_shape[0],Max_Ins_Num])
    # pdb.set_trace()
    return boxes, part_ids


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
    reg_set = tf.get_collection('losses')
    l2_loss = tf.add_n(reg_set)  # if wd=0, no effect

    return seg_loss+l2_loss, batch_seg_loss, seg_shift_labels  # the 'batch_seg_loss' and 'seg_shift_labels' are for debug


def mask_loss_graph(pred_masks, gt_seg, rois, target_class_ids):
    """Mask binary cross-entropy loss for the masks head.
    gt_seg: [batch, h,w,d]
    target_masks: [batch, num_rois, height, width, depth].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, depth, num_classes] float32 tensor
                with values from 0 to 1.
    """
    part_num = pred_masks.get_shape()[-1]
    pool_size = pred_masks.get_shape()[2]
    batch_size = gt_seg.get_shape()[0]

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

    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = tf.reshape(target_class_ids, [-1])
    mask_shape = tf.shape(targets)
    target_masks = tf.reshape(targets,
               [-1,mask_shape[2],mask_shape[3],mask_shape[4],mask_shape[5]])
    target_masks = tf.transpose(target_masks, [0, 4, 1, 2, 3])
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
    y_true = tf.gather_nd(target_masks, indices)
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
#  Miscellenous Graph Functions
############################################################

def trim_zeros_graph(boxes, name=None):
    """Often boxes are represented with matricies of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 6] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 64, 64, 64, 1))  # dtype default is tf.float32
        seg_labels = tf.zeros((32, 64, 64, 64), dtype=tf.int32)
        # input_ph, seg_ph = placeholder_inputs(32, 64)
        # outputs = get_model(input_ph, tf.constant(True))
        # loss = get_loss(outputs, seg_ph)
        outputs, features = build_Unet_FCN(inputs, tf.constant(True), 6)
        loss, batch_seg_loss, _ = Unet_FCN_loss(outputs, seg_labels, 6)

        # sess = tf.Session()
        # A = np.random.rand(32,32,32,32,1)
        # B = np.random.randint(0, 49, [32,32,32,32], np.uint8)
        # feed_dict = {input_ph: A, seg_ph: B}
        # pdb.set_trace()
        # loss_val, output_val, input_val, seg_val = sess.run([loss, outputs, input_ph, seg_ph], feed_dict=feed_dict)
        # print('good')
        # print('over')
    print('outputs\n', outputs, '\n', 'loss\n', loss)
    print('batch_seg_loss\n', batch_seg_loss)
