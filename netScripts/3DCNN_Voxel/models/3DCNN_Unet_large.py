""" 3DCNN encode decode U net
    mimick O-CNN segmentation network---20180120
"""
import tensorflow as tf
import numpy as np
import os
import sys
import pdb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util


def placeholder_inputs(batch_size, vox_size):
    vol_ph = tf.placeholder(tf.float32, shape=(batch_size, vox_size, vox_size, vox_size, 1))
    seg_ph = tf.placeholder(tf.float32, shape=(batch_size, vox_size, vox_size, vox_size))
    return vol_ph, seg_ph


def get_model(volumes, is_training, part_num, bn_decay=None, weight_decay=0.0):
    """ 3D FCN for voxel wise label prediction """
    batch_size = volumes.get_shape()[0].value  # for TF tensor
    end_volumes = {}
    input_batch = volumes

    # conv3d with batch normalization, Encoding phase
    en1 = tf_util.conv3d(input_batch, 16, kernel_size=[3,3,3],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='conv1_1', bn_decay=bn_decay, weight_decay=weight_decay)  # relu
    en1_ = en1
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

    de2 = tf_util.conv3d_transpose(de1, 64, kernel_size=[3,3,3],
                                   padding='SAME', stride=[2,2,2], bn=True,
                                   is_training=is_training, scope='deconv_2', bn_decay=bn_decay, weight_decay=weight_decay)  # shape=(12,12,12)
    de2 = tf.concat(axis=-1, values=[de2, en2])

    de3 = tf_util.conv3d_transpose(de2, 32, kernel_size=[3,3,3],
                                   padding='SAME', stride=[2,2,2], bn=True,
                                   is_training=is_training, scope='deconv_3', bn_decay=bn_decay, weight_decay=weight_decay)  # shape=(24,24,24)
    de3 = tf.concat(axis=-1, values=[de3, en1])

    de4 = tf_util.conv3d_transpose(de3, 16, kernel_size=[3,3,3],
                                   padding='SAME', stride=[2,2,2], bn=True,
                                   is_training=is_training, scope='deconv_4', bn_decay=bn_decay, weight_decay=weight_decay)  # shape=(48,48,48)
    de4 = tf.concat(axis=-1, values=[de4, en1_])
    end_volumes['deconv_features_layer4'] = de4

    # predicting phase
    net = tf_util.conv3d(de4, 64, kernel_size=[1,1,1],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='conv_pred_1', bn_decay=bn_decay, weight_decay=weight_decay)  # conv size 1
    net = tf_util.conv3d(net, 64, kernel_size=[1,1,1],
                         padding='SAME', stride=[1,1,1], bn=True,
                         is_training=is_training, scope='conv_pred_2', bn_decay=bn_decay, weight_decay=weight_decay)
    end_volumes['conv_pred_2'] = net
    net = tf_util.conv3d(net, part_num, kernel_size=[1,1,1],  # predict 50 labels
                         padding='SAME', stride=[1,1,1], bn=False, activation_fn=None,
                         is_training=is_training, scope='conv_pred_4', bn_decay=bn_decay, weight_decay=weight_decay)

    return net, end_volumes


def get_loss(pred, seg_labels, part_num):  # seg_labels: type uint8; pred: type float32
    """ pred: B * vol_size*vol_size*vol_size * num_pid,
        seg_label: B * vol_size*vol_size*vol_size
        the gt seg has label from 1 to part_num, the 0 label is for empty voxels
    """
    # shift the seg_labels data, ignore the background labels
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
    # reg_set = tf.get_collection('losses')
    # l2_loss = tf.add_n(reg_set)  # if wd=0, no effect

    return seg_loss  # the 'batch_seg_loss' and 'seg_shift_labels' are for debug


def get_loss_withback(pred, seg_labels, part_num):
    """ part_num also take background empty voxel into count
        seg_labels no longer ignore label for empty voxel
    """
    gt_labels = tf.one_hot(tf.to_int32(seg_labels), depth=part_num)
    batch_seg_loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=gt_labels)
    per_instance_seg_loss = tf.reduce_sum(batch_seg_loss, axis=[1,2,3])
    seg_loss = tf.reduce_mean(per_instance_seg_loss, axis=None)

    return seg_loss


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 64, 64, 64, 1))  # dtype default is tf.float32
        seg_labels = tf.zeros((32, 64, 64, 64), dtype=tf.int32)
        input_ph, seg_ph = placeholder_inputs(32, 64)
        # outputs = get_model(input_ph, tf.constant(True))
        # loss = get_loss(outputs, seg_ph)
        outputs, feature, feature2 = get_model(inputs, tf.constant(True), 6)
        loss, batch_seg_loss, _ = get_loss(outputs, seg_labels, 6)

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
