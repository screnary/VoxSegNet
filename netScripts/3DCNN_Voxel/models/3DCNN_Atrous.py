""" 3DCNN encode decode U net
    mimick O-CNN segmentation network---20180120
    get_model_2block_v1_afa ---20181121, TVCG response letter, additional ablation study
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
    """ 3D FCN for voxel wise label prediction, 2 block """
    batch_size = volumes.get_shape()[0].value  # for TF tensor
    end_volumes = {}
    input_batch = volumes

    # Detail Block, r=[1,2,3]
    conv = tf_util.conv3d_atrous(input_batch, 16, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=1, bn=True,
                         is_training=is_training, use_xavier=True, scope='B1_conv1')  # relu
    conv = tf_util.conv3d_atrous(conv, 32, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=2, bn=True,
                         is_training=is_training, use_xavier=True, scope='B1_conv2')  # relu
    conv = tf_util.conv3d_atrous(conv, 32, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=3, bn=True,
                         is_training=is_training, use_xavier=True, scope='B1_conv3')  # relu
    fea_detail = tf_util.rrb(conv, None, is_training, 'residual_detail_block')
    end_volumes['detail_feature'] = fea_detail

    # Long Range Block, r=[1,2,3]
    conv = tf_util.conv3d_atrous(conv, 32, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=1, bn=True,
                         is_training=is_training, use_xavier=True, scope='B2_conv1')  # relu
    conv = tf_util.conv3d_atrous(conv, 32, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=2, bn=True,
                         is_training=is_training, use_xavier=True, scope='B2_conv2')  # relu
    conv = tf_util.conv3d_atrous(conv, 64, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=3, bn=True,
                         is_training=is_training, use_xavier=True, scope='B2_conv3')  # relu
    fea_LRange = tf_util.rrb(conv, 32, is_training, 'residual_LRange_block')
    end_volumes['LRange_feature'] = fea_LRange

    # Combine Detail and Long Range features
    # # reweight fea_detail, and then added to fea_LRange
    feature = tf_util.cab(fea_LRange, fea_detail, 'channel_attention_layer')

    # Predict Phase
    net = tf_util.conv3d(feature, 64, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=True,
                is_training=is_training, scope='pred_1')
    net = tf_util.conv3d(net, 64, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=True,
                is_training=is_training, scope='pred_2')
    predicted = tf_util.conv3d(net, part_num, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=False, activation_fn=None,
                is_training=is_training, scope='pred')
    return predicted, end_volumes


def get_model_2block_att(volumes, is_training, part_num, bn_decay=None, weight_decay=0.0):
    """ 3D FCN for voxel wise label prediction
        not use concate to combine different scales features
        use resnet structures to ease training task
    """
    # batch_size = volumes.get_shape()[0].value  # for TF tensor
    end_notes = {}
    input_batch = volumes

    # Detail Block, r=[1,1,1]
    conv = tf_util.arb(input_batch, 32, rate=1, is_training=is_training,
            name='block_1_conv1', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    end_notes['block_1_conv1'] = conv
    conv = tf_util.arb(conv, 32, rate=1, is_training=is_training,
            name='block_1_conv2', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    conv = tf_util.arb(conv, 64, rate=1, is_training=is_training,
            name='block_1_conv3', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    end_notes['block_1'] = conv

    # Midlevel Block, r=[1,3,5]
    conv = tf_util.arb(conv, 32, rate=1, is_training=is_training,
            name='block_2_conv1', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    conv = tf_util.arb(conv, 32, rate=3, is_training=is_training,
            name='block_2_conv2', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    conv = tf_util.arb(conv, 64, rate=5, is_training=is_training,
            name='block_2_conv3', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    end_notes['block_2'] = conv

    # Combine Detail and Long Range features
    # # reweight fea_detail, and then added to fea_LRange
    feature = tf_util.cab(end_notes['block_2'], end_notes['block_1'], 'channel_attention_layer_1')
    # Predict Phase
    net = tf_util.conv3d(feature, 64, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=True, bn_decay=bn_decay,
                is_training=is_training, scope='pred_1')
    net = tf_util.conv3d(net, 64, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=True, bn_decay=bn_decay,
                is_training=is_training, scope='pred_2')
    predicted = tf_util.conv3d(net, part_num, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=False, activation_fn=None,
                is_training=is_training, scope='pred')
    return predicted, end_notes


def get_model_2block_v0(volumes, is_training, part_num, bn_decay=None, weight_decay=0.0):
    """ 3D FCN for voxel wise label prediction, 2 block """
    batch_size = volumes.get_shape()[0].value  # for TF tensor
    end_volumes = {}
    input_batch = volumes

    # Detail Block, r=[1,2,3]
    conv = tf_util.conv3d_atrous(input_batch, 16, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=1, bn=True,
                         is_training=is_training, use_xavier=True, scope='B1_conv1')  # relu
    conv = tf_util.conv3d_atrous(conv, 32, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=2, bn=True,
                         is_training=is_training, use_xavier=True, scope='B1_conv2')  # relu
    conv = tf_util.conv3d_atrous(conv, 32, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=3, bn=True,
                         is_training=is_training, use_xavier=True, scope='B1_conv3')  # relu
    fea_detail = tf_util.rrb(conv, None, is_training, 'residual_detail_block')
    end_volumes['detail_feature'] = fea_detail

    # Long Range Block, r=[1,3,5]
    conv = tf_util.conv3d_atrous(conv, 32, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=1, bn=True,
                         is_training=is_training, use_xavier=True, scope='B2_conv1')  # relu
    conv = tf_util.conv3d_atrous(conv, 32, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=3, bn=True,
                         is_training=is_training, use_xavier=True, scope='B2_conv2')  # relu
    conv = tf_util.conv3d_atrous(conv, 64, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=5, bn=True,
                         is_training=is_training, use_xavier=True, scope='B2_conv3')  # relu
    fea_LRange = tf_util.rrb(conv, 32, is_training, 'residual_LRange_block')
    end_volumes['LRange_feature'] = fea_LRange

    # Combine Detail and Long Range features
    # # reweight fea_detail, and then added to fea_LRange
    feature_sumed, fea_detail_weighted = tf_util.cab_new_0(fea_LRange, fea_detail, 'channel_attention_layer')
    # pdb.set_trace()
    feature = tf.concat([feature_sumed, fea_detail_weighted], axis=-1)

    # Predict Phase
    net = tf_util.conv3d(feature, 64, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=True,
                is_training=is_training, scope='pred_1')
    net = tf_util.conv3d(net, 64, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=True,
                is_training=is_training, scope='pred_2')
    predicted = tf_util.conv3d(net, part_num, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=False, activation_fn=None,
                is_training=is_training, scope='pred')
    return predicted, end_volumes


def get_model_2block_v1(volumes, is_training, part_num, bn_decay=None, weight_decay=0.0):
    """ 3D FCN for voxel wise label prediction
        use concate to combine different scales features
        use resnet structures to ease training task
        rate [2,3,4]+[1,1,1], RF=25, same as [1,1,1+[1,3,5]
    """
    # batch_size = volumes.get_shape()[0].value  # for TF tensor
    end_notes = {}
    input_batch = volumes

    # Detail Block, r=[2,3,4]
    conv = tf_util.arb(input_batch, 32, rate=2, is_training=is_training,
            name='block_1_conv1', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    end_notes['aconv_1'] = conv
    conv = tf_util.arb(conv, 32, rate=3, is_training=is_training,
            name='block_1_conv2', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    end_notes['aconv_2'] = conv
    conv = tf_util.arb(conv, 64, rate=4, is_training=is_training,
            name='block_1_conv3', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    end_notes['aconv_3'] = conv

    # Midlevel Block, r=[1,1,1]
    conv = tf_util.arb(conv, 32, rate=1, is_training=is_training,
            name='block_2_conv1', activation_fn=tf.nn.relu, bn_decay=bn_decay)

    conv = tf.concat([conv, end_notes['aconv_2']], axis=-1)
    conv = tf_util.arb(conv, 32, rate=1, is_training=is_training,
            name='block_2_conv2', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    end_notes['aconv_5'] = conv

    conv = tf.concat([conv, end_notes['aconv_1']], axis=-1)
    conv = tf_util.arb(conv, 32, rate=1, is_training=is_training,
            name='block_2_conv3', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    end_notes['aconv_6'] = conv

    # Predict Phase
    net = tf_util.conv3d(conv, 64, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=True, bn_decay=bn_decay,
                is_training=is_training, scope='pred_1')
    net = tf_util.conv3d(net, 64, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=True, bn_decay=bn_decay,
                is_training=is_training, scope='pred_2')
    predicted = tf_util.conv3d(net, part_num, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=False, activation_fn=None,
                is_training=is_training, scope='pred')
    return predicted, end_notes


def get_model_2block_v1_afa(volumes, is_training, part_num, bn_decay=None, weight_decay=0.0):
    """ 3D FCN for voxel wise label prediction
        use concate to combine different scales features
        use resnet structures to ease training task
        rate [2,3,4]+[1,1,1], RF=25, same as [1,1,1+[1,3,5]
    """
    # batch_size = volumes.get_shape()[0].value  # for TF tensor
    end_notes = {}
    input_batch = volumes

    # Detail Block, r=[2,3,4]
    conv = tf_util.arb(input_batch, 32, rate=2, is_training=is_training,
            name='block_1_conv1', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    end_notes['aconv_1'] = conv
    conv = tf_util.arb(conv, 32, rate=3, is_training=is_training,
            name='block_1_conv2', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    end_notes['aconv_2'] = conv
    conv = tf_util.arb(conv, 64, rate=4, is_training=is_training,
            name='block_1_conv3', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    end_notes['aconv_3'] = conv

    # Midlevel Block, r=[1,1,1]
    conv = tf_util.arb(conv, 32, rate=1, is_training=is_training,
            name='block_2_conv1', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    end_notes['aconv_4'] = conv
    
    feature_1, weight_1 = tf_util.cab_new(1.0*end_notes['aconv_4'], 10.0*end_notes['aconv_2'], 'channel_attention_layer_1')
    feature_2, weight_2 = tf_util.cab_new(1.0*feature_1, 10.0*end_notes['aconv_1'], 'channel_attention_layer_2')
    end_notes['combine_1'] = feature_1
    end_notes['combine_weight_1'] = weight_1
    end_notes['combine_2'] = feature_2
    end_notes['combine_weight_2'] = weight_2
    feature = tf.concat(
            [feature_1,feature_2],
            axis=-1)
    end_notes['combined_fea'] = feature

    # Predict Phase
    net = tf_util.conv3d(feature, 64, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=True, bn_decay=bn_decay,
                is_training=is_training, scope='pred_1')
    net = tf_util.conv3d(net, 64, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=True, bn_decay=bn_decay,
                is_training=is_training, scope='pred_2')
    predicted = tf_util.conv3d(net, part_num, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=False, activation_fn=None,
                is_training=is_training, scope='pred')
    return predicted, end_notes


def get_model_2block(volumes, is_training, part_num, bn_decay=None, weight_decay=0.0):
    """ 3D FCN for voxel wise label prediction
        not use concate to combine different scales features
        use resnet structures to ease training task
        - 2 SDE-concate
    """
    # batch_size = volumes.get_shape()[0].value  # for TF tensor
    end_notes = {}
    input_batch = volumes

    # Detail Block, r=[1,1,1]
    conv = tf_util.arb(input_batch, 32, rate=1, is_training=is_training,
            name='block_1_conv1', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    end_notes['block_1_conv1'] = conv
    conv = tf_util.arb(conv, 32, rate=1, is_training=is_training,
            name='block_1_conv2', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    conv = tf_util.arb(conv, 64, rate=1, is_training=is_training,
            name='block_1_conv3', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    end_notes['block_1'] = conv

    # Midlevel Block, r=[1,3,5]
    conv = tf_util.arb(conv, 32, rate=1, is_training=is_training,
            name='block_2_conv1', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    conv = tf_util.arb(conv, 32, rate=3, is_training=is_training,
            name='block_2_conv2', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    conv = tf_util.arb(conv, 64, rate=5, is_training=is_training,
            name='block_2_conv3', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    end_notes['block_2'] = conv

    # Combine Detail and Long Range features
    # # reweight fea_detail, and then added to fea_LRange
    # feature = tf_util.cab(end_notes['block_2'], end_notes['block_1'], 'channel_attention_layer_1')
    # # for concate directly
    feature = tf.concat([end_notes['block_2'], end_notes['block_1']], axis=-1)
    end_notes['combine'] = feature
    # Predict Phase
    net = tf_util.conv3d(feature, 64, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=True, bn_decay=bn_decay,
                is_training=is_training, scope='pred_1')
    net = tf_util.conv3d(net, 64, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=True, bn_decay=bn_decay,
                is_training=is_training, scope='pred_2')
    predicted = tf_util.conv3d(net, part_num, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=False, activation_fn=None,
                is_training=is_training, scope='pred')
    return predicted, end_notes


def get_model_3block(volumes, is_training, part_num, bn_decay=None, weight_decay=0.0):
    """ 3D FCN for voxel wise label prediction
        not use concate to combine different scales features
        use resnet structures to ease training task
        -3 SDE AFA
    """
    # batch_size = volumes.get_shape()[0].value  # for TF tensor
    end_notes = {}
    input_batch = volumes

    # Detail Block, r=[1,1,1]
    conv = tf_util.arb(input_batch, 32, rate=1, is_training=is_training,
            name='block_1_conv1', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    conv = tf_util.arb(conv, 32, rate=1, is_training=is_training,
            name='block_1_conv2', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    conv = tf_util.arb(conv, 64, rate=1, is_training=is_training,
            name='block_1_conv3', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    end_notes['block_1'] = conv

    # Midlevel Block, r=[1,3,5]
    conv = tf_util.arb(conv, 32, rate=1, is_training=is_training,
            name='block_2_conv1', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    conv = tf_util.arb(conv, 32, rate=3, is_training=is_training,
            name='block_2_conv2', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    conv = tf_util.arb(conv, 64, rate=5, is_training=is_training,
            name='block_2_conv3', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    end_notes['block_2'] = conv

    # LongRange Block, r=[1,3,5]
    conv = tf_util.arb(conv, 32, rate=1, is_training=is_training,
            name='block_3_conv1', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    conv = tf_util.arb(conv, 32, rate=3, is_training=is_training,
            name='block_3_conv2', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    conv = tf_util.arb(conv, 64, rate=5, is_training=is_training,
            name='block_3_conv3', activation_fn=tf.nn.relu, bn_decay=bn_decay)
    end_notes['block_3'] = conv
    # pdb.set_trace()

    # Combine Detail and Long Range features
    # # reweight fea_detail, and then added to fea_LRange
    # feature_1 = tf_util.cab(end_notes['block_3'], end_notes['block_2'], 'channel_attention_layer_1')
    # feature_2 = tf_util.cab(feature_1, end_notes['block_1'], 'channel_attention_layer_2')
    feature_1, weight_1 = tf_util.cab_new(end_notes['block_3'], end_notes['block_2'], 'channel_attention_layer_1')
    feature_2, weight_2 = tf_util.cab_new(feature_1, end_notes['block_1'], 'channel_attention_layer_2')
    end_notes['combine_1'] = feature_1
    end_notes['combine_weight_1'] = weight_1
    end_notes['combine_2'] = feature_2
    end_notes['combine_weight_2'] = weight_2
    feature = tf.concat(
            [feature_1,feature_2],
            axis=-1)
    end_notes['combined_fea'] = feature

    # Predict Phase
    net = tf_util.conv3d(feature, 64, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=True, bn_decay=bn_decay,
                is_training=is_training, scope='pred_1')
    net = tf_util.conv3d(net, 64, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=True, bn_decay=bn_decay,
                is_training=is_training, scope='pred_2')
    predicted = tf_util.conv3d(net, part_num, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=False, activation_fn=None,
                is_training=is_training, scope='pred')
    return predicted, end_notes


def get_model_3block_v0(volumes, is_training, part_num, bn_decay=None, weight_decay=0.0):
    """ 3D FCN for voxel wise label prediction
        use concate to combine different scales features
    """
    batch_size = volumes.get_shape()[0].value  # for TF tensor
    end_volumes = {}
    input_batch = volumes

    # Detail Block, r=[1,2,3]
    conv = tf_util.conv3d_atrous(input_batch, 16, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=1, bn=True, bn_decay=bn_decay,
                         is_training=is_training, use_xavier=True, scope='B1_conv1')  # relu
    conv = tf_util.conv3d_atrous(conv, 32, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=1, bn=True, bn_decay=bn_decay,
                         is_training=is_training, use_xavier=True, scope='B1_conv2')  # relu
    conv = tf_util.conv3d_atrous(conv, 32, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=1, bn=True, bn_decay=bn_decay,
                         is_training=is_training, use_xavier=True, scope='B1_conv3')  # relu
    fea_detail = tf_util.rrb(conv, 32, is_training, 'residual_detail_block')
    end_volumes['detail_feature'] = fea_detail

    # r=[1,3,5]
    conv = tf_util.conv3d_atrous(conv, 32, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=1, bn=True, bn_decay=bn_decay,
                         is_training=is_training, use_xavier=True, scope='B2_conv1')  # relu
    conv = tf_util.conv3d_atrous(conv, 32, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=3, bn=True, bn_decay=bn_decay,
                         is_training=is_training, use_xavier=True, scope='B2_conv2')  # relu
    conv = tf_util.conv3d_atrous(conv, 64, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=5, bn=True, bn_decay=bn_decay,
                         is_training=is_training, use_xavier=True, scope='B2_conv3')  # relu
    fea_MRange = tf_util.rrb(conv, 32, is_training, 'residual_MRange_block')
    end_volumes['MRange_feature'] = fea_MRange

    # Long Range Block, r=[2,3,7]
    conv = tf_util.conv3d_atrous(conv, 32, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=1, bn=True, bn_decay=bn_decay,
                         is_training=is_training, use_xavier=True, scope='B3_conv1')  # relu
    conv = tf_util.conv3d_atrous(conv, 32, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=3, bn=True, bn_decay=bn_decay,
                         is_training=is_training, use_xavier=True, scope='B3_conv2')  # relu
    conv = tf_util.conv3d_atrous(conv, 64, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=5, bn=True, bn_decay=bn_decay,
                         is_training=is_training, use_xavier=True, scope='B3_conv3')  # relu
    fea_LRange = tf_util.rrb(conv, 32, is_training, 'residual_LRange_block')
    end_volumes['LRange_feature'] = fea_LRange

    # Combine Detail and Long Range features
    # # reweight fea_detail, and then added to fea_LRange
    feature_sumed_L, fea_weighted_M = tf_util.cab_new(fea_LRange, fea_MRange, 'channel_attention_layer_1')
    feature_sumed_M, fea_weighted_d = tf_util.cab_new(fea_weighted_M, fea_detail, 'channel_attention_layer_2')
    feature =tf.concat([feature_sumed_L, feature_sumed_M], axis=-1)

    # Predict Phase
    net = tf_util.conv3d(feature, 64, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=True, bn_decay=bn_decay,
                is_training=is_training, scope='pred_1')
    net = tf_util.conv3d(net, 64, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=True, bn_decay=bn_decay,
                is_training=is_training, scope='pred_2')
    predicted = tf_util.conv3d(net, part_num, kernel_size=[1,1,1],
                padding='SAME', stride=[1,1,1], bn=False, activation_fn=None,
                is_training=is_training, scope='pred')
    return predicted, end_volumes


def get_model_1block(volumes, is_training, part_num, bn_decay=None, weight_decay=0.0):
    """ 3D FCN for voxel wise label prediction """
    batch_size = volumes.get_shape()[0].value  # for TF tensor
    end_volumes = {}
    input_batch = volumes

    # Detail Block, r=[1,2,3]
    conv = tf_util.conv3d_atrous(input_batch, 16, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=1, bn=True,
                         is_training=is_training, use_xavier=True, scope='B1_conv1')  # relu
    conv = tf_util.conv3d_atrous(conv, 32, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=2, bn=True,
                         is_training=is_training, use_xavier=True, scope='B1_conv2')  # relu
    conv = tf_util.conv3d_atrous(conv, 32, kernel_size=[3,3,3],
                         padding='same', stride=[1,1,1], dilation_rate=3, bn=True,
                         is_training=is_training, use_xavier=True, scope='B1_conv3')  # relu
    fea_detail = tf_util.rrb(conv, None, is_training, 'residual_detail_block')
    end_volumes['detail_feature'] = fea_detail
    feature = fea_detail

    # Predict Phase
    net = tf_util.conv3d(feature, 64, kernel_size=[1,1,1],
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
