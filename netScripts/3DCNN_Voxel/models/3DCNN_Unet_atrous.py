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
    #features = tf_util.conv3d_atrous(conv, 64, kernel_size=[3,3,3],
    #                     padding='same', stride=[1,1,1], dilation_rate=1, bn=True,
    #                     is_training=is_training, scope='conv5')  # relu
    features = conv

    # no Decoding phase
    end_volumes['atrous_features_layer4'] = features
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
    

def get_loss(pred, seg_labels, part_num):  # seg_labels: type uint8; pred: type float32
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
