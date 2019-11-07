""" Wrapper functions for TensorFlow layers.

Author: Charles R. Qi
Date: November 2016
"""

import numpy as np
import tensorflow as tf

import pdb


def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
      use_xavier: bool, whether to use xavier initializer

    Returns:
      Variable Tensor
    """
    if use_xavier:
        initializer = tf.contrib.layers.xavier_initializer()
    else:
        initializer = tf.truncated_normal_initializer(stddev=stddev)
    # pdb.set_trace()
    var = _variable_on_cpu(name, shape, initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=1,
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
    """ 1D convolution with non-linear operation.

    Args:
      inputs: 3-D tensor variable BxLxC
      num_output_channels: int
      kernel_size: int
      scope: string
      stride: int
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true
      stddev: float, stddev for truncated_normal init
      weight_decay: float
      activation_fn: function
      bn: bool, whether to use batch norm
      bn_decay: float or float tensor variable in [0,1]
      is_training: bool Tensor variable

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_size,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        outputs = tf.nn.conv1d(inputs, kernel,
                               stride=stride,
                               padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv1d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
    """ 2D convolution with non-linear operation.

    Args:
      inputs: 4-D tensor variable BxHxWxC
      num_output_channels: int
      kernel_size: a list of 2 ints
      scope: string
      stride: a list of 2 ints
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true
      stddev: float, stddev for truncated_normal init
      weight_decay: float
      activation_fn: function
      bn: bool, whether to use batch norm
      bn_decay: float or float tensor variable in [0,1]
      is_training: bool Tensor variable

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(inputs, kernel,
                               [1, stride_h, stride_w, 1],
                               padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def conv2d_transpose(inputs,
                     num_output_channels,
                     kernel_size,
                     scope,
                     stride=[1, 1],
                     padding='SAME',
                     use_xavier=True,
                     stddev=1e-3,
                     weight_decay=0.0,
                     activation_fn=tf.nn.relu,
                     bn=False,
                     bn_decay=None,
                     is_training=None):
    """ 2D convolution transpose with non-linear operation.

    Args:
      inputs: 4-D tensor variable BxHxWxC
      num_output_channels: int
      kernel_size: a list of 2 ints
      scope: string
      stride: a list of 2 ints
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true
      stddev: float, stddev for truncated_normal init
      weight_decay: float
      activation_fn: function
      bn: bool, whether to use batch norm
      bn_decay: float or float tensor variable in [0,1]
      is_training: bool Tensor variable

    Returns:
      Variable tensor

    Note: conv2d(conv2d_transpose(a, num_out, ksize, stride), a.shape[-1], ksize, stride) == a
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_output_channels, num_in_channels]  # reversed to conv2d
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_h, stride_w = stride

        # from slim.convolution2d_transpose
        def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
            dim_size *= stride_size

            if padding == 'VALID' and dim_size is not None:
                dim_size += max(kernel_size - stride_size, 0)
            return dim_size

        # caculate output shape
        batch_size = inputs.get_shape()[0].value
        height = inputs.get_shape()[1].value
        width = inputs.get_shape()[2].value
        out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
        out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
        output_shape = [batch_size, out_height, out_width, num_output_channels]

        outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape,
                                         [1, stride_h, stride_w, 1],
                                         padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def conv3d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
    """ 3D convolution with non-linear operation.

    Args:
      inputs: 5-D tensor variable BxDxHxWxC
      num_output_channels: int
      kernel_size: a list of 3 ints
      scope: string
      stride: a list of 3 ints
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true
      stddev: float, stddev for truncated_normal init
      weight_decay: float
      activation_fn: function
      bn: bool, whether to use batch norm
      bn_decay: float or float tensor variable in [0,1]
      is_training: bool Tensor variable

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_d, kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)  # initialize the kernel
        stride_d, stride_h, stride_w = stride
        outputs = tf.nn.conv3d(inputs, kernel,
                               [1, stride_d, stride_h, stride_w, 1],
                               padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv3d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def conv3d_atrous(inputs,
           num_output_channels,
           kernel_size,
           scope,
           dilation_rate=1,
           stride=[1, 1, 1],
           padding='same',
           use_xavier=True,
           activation_fn=tf.nn.relu,
           use_bias=True,
           bn=False,
           bn_decay=None,
           is_training=None):
    """using atrous convolution---20180520
    """
    # initializer = tf.truncated_normal_initializer(stddev=stddev)
    with tf.variable_scope(scope) as sc:
        outputs = tf.layers.conv3d(
                  inputs=inputs,
                  filters=num_output_channels,
                  kernel_size=kernel_size,
                  strides=stride,
                  padding=padding,
                  dilation_rate=dilation_rate,
                  activation=None,
                  use_bias=use_bias
                  )
        if bn:
            outputs = batch_norm_for_conv3d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def decoder(features, end_points, is_training, scope, num_part):
    """ Decode features to mask predict
    Arg:
        features (tensor): features produced by unet.
        end_points (list of tenosr): the end points of unet
        is_training (bool): Is the network in training mode?
    Return:
        mask (tensor): mask decoded from features.
    """
    with tf.variable_scope(scope) as sc:
        output_c = [16,16,32,32]
        layers = []

        unet_point = []
        unet_point.append(end_points['atrous_conv1'])
        unet_point.append(end_points['atrous_conv2'])
        unet_point.append(end_points['atrous_conv3'])
        unet_point.append(end_points['atrous_conv4'])

        for i in range(3, -1, -1):
            if i == 3:
                curr = rrb(features, output_c[i], is_training, 'rrb_%d_1' % i)
                # curr = dropout(curr, keep_prob=0.7, is_training=is_training,
                #           scope='drop_%d' % i)
                layers.append(curr)
                continue
            curr = rrb(unet_point[i], None, is_training, 'rrb_%d_1' % i)
            curr = cab(layers[-1], curr, 'cab_%d' % i)
            curr = rrb(curr, output_c[i], is_training, 'rrb_%d_2' % i)
            # if i >= 1:
            #     curr = dropout(curr, keep_prob=0.7, is_training=is_training,
            #               scope='drop_%d' % i)
            layers.append(curr)

        # conv = rrb(layers[-1], None, is_training, 'rrb_0')
        conv = layers[-1]

        return conv


def rrb(feature, output_c, is_training, name, activation_fn=tf.nn.relu):
    """ RRB: Refinement Residual Block """
    if output_c == None:
        output_c = feature.shape.as_list()[-1]

    with tf.variable_scope(name):
        feature = tf.layers.conv3d(
            inputs=feature,
            filters=output_c,
            kernel_size=1,
            padding='same',
            activation=None
        )

        conv = tf.layers.conv3d(
            inputs=feature,
            filters=output_c,
            kernel_size=3,
            padding='same',
            activation=None,
            use_bias=False
        )

        bn = batch_norm_for_conv3d(conv, is_training,
                                   bn_decay=None, scope='bn')

        nonlinear = tf.nn.relu(bn)

        conv = tf.layers.conv3d(
            inputs=nonlinear,
            filters=output_c,
            kernel_size=3,
            padding='same',
            activation=None
        )

        sumed = feature + conv

        if activation_fn is not None:
            nonlinear = activation_fn(sumed)

    return nonlinear


def arb_v1(feature, output_c, rate, is_training, name, activation_fn=tf.nn.relu):
    """ ARB: Atrous Residual Block---not correct residual usage
        rate is [1,2,3] or [1,1,1] ... : the dilated rate of atrous_convs
    """
    if output_c is None:
        output_c = feature.shape.as_list()[-1]

    with tf.variable_scope(name):
        feature = tf.layers.conv3d(
            inputs=feature,
            filters=output_c,
            kernel_size=1,
            padding='same',
            activation=None,
        )

        conv = tf.layers.conv3d(
            inputs=feature,
            filters=output_c,
            kernel_size=3,
            dilation_rate=rate[0],
            padding='same',
            activation=None,
            use_bias=False,
            name='atrous_conv1'
        )

        bn = batch_norm_for_conv3d(conv, is_training,
                                   bn_decay=None, scope='bn_1')

        nonlinear = tf.nn.relu(bn)

        conv = tf.layers.conv3d(
            inputs=nonlinear,
            filters=output_c,
            kernel_size=3,
            dilation_rate=rate[1],
            padding='same',
            activation=None,
            use_bias=False,
            name='atrous_conv2'
        )

        bn = batch_norm_for_conv3d(conv, is_training,
                                   bn_decay=None, scope='bn_2')

        nonlinear = tf.nn.relu(bn)

        conv = tf.layers.conv3d(
            inputs=nonlinear,
            filters=output_c,
            kernel_size=3,
            dilation_rate=rate[2],
            padding='same',
            activation=None,
            name='atrous_conv3'
        )

        sumed = feature + conv

        if activation_fn is not None:
            nonlinear = activation_fn(sumed)

    return nonlinear


def arb(feature, output_c, rate, is_training, name, activation_fn=tf.nn.relu, bn_decay=None):
    """ ARB: Atrous Residual Block---Bottle net structure
        rate is an integer ... : the dilated rate of atrous_convs
    """
    if output_c is None:
        output_c = feature.shape.as_list()[-1]

    with tf.variable_scope(name) as sc:
        # this layer for change feature map channels
        feature_ = conv3d_atrous(feature,
            output_c,
            kernel_size=1,
            scope='conv0_ks1',
            dilation_rate=1,
            stride=[1, 1, 1],
            padding='same',
            use_bias=False,
            activation_fn=None,
            bn=True,
            bn_decay=bn_decay,
            is_training=is_training)
        conv = conv3d_atrous(feature,
            32,
            kernel_size=1,
            scope='conv1_ks1',
            dilation_rate=1,
            stride=[1, 1, 1],
            padding='same',
            use_bias=False,
            activation_fn=tf.nn.relu,
            bn=True,
            bn_decay=bn_decay,
            is_training=is_training)
        # dropout before atrous
        # conv = dropout(conv,
        #     is_training=is_training,
        #     scope='drop_1',
        #     keep_prob=0.7)
        # only this layer for atrous conv
        conv = conv3d_atrous(conv,
            32,
            kernel_size=3,
            scope='conv2_atrous',
            dilation_rate=rate,
            stride=[1, 1, 1],
            padding='same',
            use_bias=False,
            activation_fn=tf.nn.relu,
            bn=True,
            bn_decay=bn_decay,
            is_training=is_training)
        # dropout after atrous
        # conv = dropout(conv,
        #     is_training=is_training,
        #     scope='drop_2',
        #     keep_prob=0.7)
        conv = conv3d_atrous(conv,
            output_c,
            kernel_size=1,
            scope='conv3_ks1',
            dilation_rate=1,
            stride=[1, 1, 1],
            padding='same',
            use_bias=False,
            activation_fn=None,
            bn=True,
            bn_decay=bn_decay,
            is_training=is_training)

        output = feature_ + conv

        if activation_fn is not None:
            output = activation_fn(output)

    return output


def cab(prev, curr, name):
    """ CAB: Channel Attention Block """
    with tf.variable_scope(name):
        output_c = curr.shape.as_list()[-1]
        concat = tf.concat([curr, prev], axis=-1)
        global_pool = tf.reduce_mean(concat, axis=[1, 2, 3], keep_dims=True)

        conv = tf.layers.conv3d(
            inputs=global_pool,
            filters=output_c,
            kernel_size=1,
            padding='same',
            activation=tf.nn.relu
        )

        conv = tf.layers.conv3d(
            inputs=conv,
            filters=output_c,
            kernel_size=1,
            padding='same',
            activation=tf.nn.sigmoid
        )

        weighted = curr * conv

        sumed = weighted + prev

    return sumed


def cab_new(prev, curr, name):
    """ CAB: Channel Attention Block """
    with tf.variable_scope(name):
        output_c = curr.shape.as_list()[-1]
        concat = tf.concat([curr, prev], axis=-1)
        global_pool = tf.reduce_mean(concat, axis=[1, 2, 3], keep_dims=True)

        conv = tf.layers.conv3d(
            inputs=global_pool,
            filters=output_c,
            kernel_size=1,
            padding='same',
            activation=tf.nn.relu
        )

        conv = tf.layers.conv3d(
            inputs=conv,
            filters=output_c,
            kernel_size=1,
            padding='same',
            activation=tf.nn.sigmoid
        )

        weighted = curr * conv

        sumed = weighted + prev

    return sumed, weighted


def conv3d_transpose(inputs,
                     num_output_channels,
                     kernel_size,
                     scope,
                     stride=[1, 1, 1],
                     padding='SAME',
                     use_xavier=True,
                     stddev=1e-3,
                     weight_decay=0.0,
                     activation_fn=tf.nn.relu,
                     bn=False,
                     bn_decay=None,
                     is_training=None):
    """ 3D convolution transpose with non-linear operation.

    Args:
      inputs: 5-D tensor variable BxHxWxC
      num_output_channels: int
      kernel_size: a list of 3 ints
      scope: string
      stride: a list of 3 ints
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true
      stddev: float, stddev for truncated_normal init
      weight_decay: float
      activation_fn: function
      bn: bool, whether to use batch norm
      bn_decay: float or float tensor variable in [0,1]
      is_training: bool Tensor variable

    Returns:
      Variable tensor

    Note: conv3d(conv3d_transpose(a, num_out, ksize, stride), a.shape[-1], ksize, stride) == a
    why? have the above Note?? the shape should be equal but the value?? -- WZJ
    """
    with tf.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_d, kernel_h, kernel_w,
                        num_output_channels, num_in_channels]  # reversed to conv3d
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_d, stride_h, stride_w = stride

        # from slim.convolution3d_transpose
        def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
            dim_size *= stride_size

            if padding == 'VALID' and dim_size is not None:
                dim_size += max(kernel_size - stride_size, 0)
            return dim_size

        # caculate output shape
        batch_size = inputs.get_shape()[0].value
        depth = inputs.get_shape()[1].value
        height = inputs.get_shape()[2].value
        width = inputs.get_shape()[3].value
        out_depth = get_deconv_dim(depth, stride_d, kernel_d, padding)
        out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
        out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
        output_shape = [batch_size, out_depth, out_height, out_width, num_output_channels]

        outputs = tf.nn.conv3d_transpose(inputs, kernel, output_shape,
                                         [1, stride_d, stride_h, stride_w, 1],
                                         padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv3d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.0,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None):
    """ Fully connected layer with non-linear operation.

    Args:
      inputs: 2-D tensor BxN
      num_outputs: int

    Returns:
      Variable tensor of size B x num_outputs.
    """
    with tf.variable_scope(scope) as sc:
        num_input_units = inputs.get_shape()[-1].value
        weights = _variable_with_weight_decay('weights',
                                              shape=[num_input_units, num_outputs],
                                              use_xavier=use_xavier,
                                              stddev=stddev,
                                              wd=weight_decay)
        outputs = tf.matmul(inputs, weights)
        biases = _variable_on_cpu('biases', [num_outputs],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
    """ 2D max pooling.

    Args:
      inputs: 4-D tensor BxHxWxC
      kernel_size: a list of 2 ints
      stride: a list of 2 ints

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.max_pool(inputs,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 name=sc.name)
        return outputs


def avg_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
    """ 2D avg pooling.

    Args:
      inputs: 4-D tensor BxHxWxC
      kernel_size: a list of 2 ints
      stride: a list of 2 ints

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.avg_pool(inputs,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 name=sc.name)
        return outputs


def max_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
    """ 3D max pooling.

    Args:
      inputs: 5-D tensor BxDxHxWxC
      kernel_size: a list of 3 ints
      stride: a list of 3 ints

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        stride_d, stride_h, stride_w = stride
        outputs = tf.nn.max_pool3d(inputs,
                                   ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                                   strides=[1, stride_d, stride_h, stride_w, 1],
                                   padding=padding,
                                   name=sc.name)
        return outputs


def avg_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
    """ 3D avg pooling.

    Args:
      inputs: 5-D tensor BxDxHxWxC
      kernel_size: a list of 3 ints
      stride: a list of 3 ints

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        stride_d, stride_h, stride_w = stride
        outputs = tf.nn.avg_pool3d(inputs,
                                   ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                                   strides=[1, stride_d, stride_h, stride_w, 1],
                                   padding=padding,
                                   name=sc.name)
        return outputs


def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
    """ Batch normalization on convolutional maps and beyond...
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

    Args:
        inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
        is_training:   boolean tf.Varialbe, true indicates training phase
        scope:         string, variable scope
        moments_dims:  a list of ints, indicating dimensions for moments calculation
        bn_decay:      float or float tensor variable, controling moving average weight
    Return:
        normed:        batch-normalized maps
    """
    with tf.variable_scope(scope) as sc:  # change to get_variable
        num_channels = inputs.get_shape()[-1].value
        # beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
        #                    name='beta', trainable=True)
        # gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
        #                     name='gamma', trainable=True)
        beta = tf.get_variable(name='beta', shape=[num_channels],
                initializer=tf.constant_initializer(0.0))
        gamma = tf.get_variable(name='gamma', shape=[num_channels],
                initializer=tf.constant_initializer(1.0))
        # caculate the mean and variance of inputs
        batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
        decay = bn_decay if bn_decay is not None else 0.6  # before 0.9
        # maintains moving averages of variables by employing an exponential decay
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        # Operator that maintains moving averages of variables.
        ema_apply_op = tf.cond(is_training,
                               lambda: ema.apply([batch_mean, batch_var]),  # update the mean and var
                               lambda: tf.no_op())  # do nothing, while testing.

        # Update moving average and return current batch's avg and var.
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # ema.average returns the Variable holding the average of var.
        mean, var = tf.cond(is_training,
                            mean_var_with_update,  # when training, shadow_var = decay*shadow_var + (1-decay)*var; update the mean and var
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))  # when evaluating, no need to maintain moving averages; returns the shadow variable for a given variable
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
    return normed


# ##### original version
def batch_norm_template_v0(inputs, is_training, scope, moments_dims, bn_decay):
    """ Batch normalization on convolutional maps and beyond...
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

    Args:
        inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
        is_training:   boolean tf.Varialbe, true indicates training phase
        scope:         string, variable scope
        moments_dims:  a list of ints, indicating dimensions for moments calculation
        bn_decay:      float or float tensor variable, controling moving average weight
    Return:
        normed:        batch-normalized maps
    """
    with tf.variable_scope(scope) as sc:
        num_channels = inputs.get_shape()[-1].value
        beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                            name='gamma', trainable=True)
        # caculate the mean and variance of inputs
        batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
        decay = bn_decay if bn_decay is not None else 0.9
        # maintains moving averages of variables by employing an exponential decay
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        # Operator that maintains moving averages of variables.
        ema_apply_op = tf.cond(is_training,
                               lambda: ema.apply([batch_mean, batch_var]),  # update the mean and var
                               lambda: tf.no_op())  # do nothing, while testing.

        # Update moving average and return current batch's avg and var.
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # ema.average returns the Variable holding the average of var.
        mean, var = tf.cond(is_training,
                            mean_var_with_update,  # when training, shadow_var = decay*shadow_var + (1-decay)*var; update the mean and var
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))  # when evaluating, no need to maintain moving averages; returns the shadow variable for a given variable
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
    return normed


def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
    """ Batch normalization on FC data.

    Args:
        inputs:      Tensor, 2D BxC input
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0,], bn_decay)


def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 1D convolutional maps.

    Args:
        inputs:      Tensor, 3D BLC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0,1], bn_decay)


def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 2D convolutional maps.

    Args:
        inputs:      Tensor, 4D BHWC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0,1,2], bn_decay)


def batch_norm_for_conv3d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 3D convolutional maps.

    Args:
        inputs:      Tensor, 5D BDHWC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    # [0,1,2,3], 'global normalization', aggregating the contents of inputs across axes 0,1,2,3; without channel axis
    return batch_norm_template(inputs, is_training, scope, [0,1,2,3], bn_decay)


def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
    """ Dropout layer.

    Args:
      inputs: tensor
      is_training: boolean tf.Variable
      scope: string
      keep_prob: float in [0,1]
      noise_shape: list of ints

    Returns:
      tensor variable
    """
    with tf.variable_scope(scope) as sc:
        outputs = tf.cond(is_training,
                          lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                          lambda: inputs)
        return outputs
