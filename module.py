import tensorflow as tf
import numpy as np
from ops import conv2d, deconv2d, build_mapping_resnet_block, build_feature_resnet_block, pool, element_wise_linear_add, PReLU


def network(depth, rgb, reuse=False, dim1=56, dim2=12, sr_times=16, name='network', do_norm=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        height, width = depth.get_shape()[1], depth.get_shape()[2]
        depth = tf.image.resize_bicubic(depth, size=[height * sr_times, width*sr_times])
        depth_conv1 = conv2d(depth, dim1, 3, 1, 0.02, fn='prelu', do_norm=do_norm,
                             name='depth_conv1', name_bn='bn1', name_prelu='alpha1')
        depth_feature_resnet1 = build_feature_resnet_block(depth_conv1, dim1, 'depth_feature_resnet1')
        depth_pool1 = pool(depth_feature_resnet1, 'pool1')
        depth_mapping_resnet1 = build_mapping_resnet_block(depth_feature_resnet1, dim1, dim2, 'depth_mapping_resnet1')
        depth_feature_resnet2 = build_feature_resnet_block(depth_pool1, dim1, 'depth_feature_resnet2')
        depth_pool2 = pool(depth_feature_resnet2, 'pool2')
        depth_mapping_resnet2 = build_mapping_resnet_block(depth_feature_resnet2, dim1, dim2, 'depth_mapping_resnet2')
        depth_feature_resnet3 = build_feature_resnet_block(depth_pool2, dim1, 'depth_feature_resnet3')
        depth_pool3 = pool(depth_feature_resnet3, 'pool3')
        depth_mapping_resnet3 = build_mapping_resnet_block(depth_feature_resnet3, dim1, dim2, 'depth_mapping_resnet3')
        depth_feature_resnet4 = build_feature_resnet_block(depth_pool3, dim1, 'depth_feature_resnet4')
        depth_mapping_resnet4 = build_mapping_resnet_block(depth_feature_resnet4, dim1, dim2, 'depth_mapping_resnet4')
        depth_deconv1 = deconv2d(depth_mapping_resnet4, dim1,  do_norm=do_norm,
                                 name='depth_deconv1', name_bn='bn1', name_prelu='alpha1')
        depth_deconv1 = element_wise_linear_add(depth_deconv1, depth_mapping_resnet3, 'alpha1')
        depth_deconv2 = deconv2d(depth_deconv1, dim1,  do_norm=do_norm,
                                 name='depth_deconv2', name_bn='bn1', name_prelu='alpha1')
        depth_deconv2 = element_wise_linear_add(depth_deconv2, depth_mapping_resnet2, 'alpha2')
        depth_deconv3 = deconv2d(depth_deconv2, dim1,  do_norm=do_norm,
                                 name='depth_deconv3', name_bn='bn1', name_prelu='alpha1')
        depth_deconv3 = element_wise_linear_add(depth_deconv3, depth_mapping_resnet1, 'alpha3')

        rgb_conv1 = conv2d(rgb, dim1, 3, 1, 0.02, fn='prelu', do_norm=do_norm,
                           name='rgb_conv1', name_bn='bn1', name_prelu='alpha1')
        rgb_feature_resnet1 = build_feature_resnet_block(rgb_conv1, dim1, 'rgb_feature_resnet1')
        rgb_mapping_resnet1 = build_mapping_resnet_block(rgb_feature_resnet1, dim1, dim2, 'rgb_mapping_resnet1')
        rgb_pool1 = pool(rgb_feature_resnet1, 'pool1')
        rgb_feature_resnet2 = build_feature_resnet_block(rgb_pool1, dim1, 'rgb_feature_resnet2')
        rgb_mapping_resnet2 = build_mapping_resnet_block(rgb_feature_resnet2, dim1, dim2, 'rgb_mapping_resnet2')
        rgb_pool2 = pool(rgb_feature_resnet2, 'pool2')
        rgb_feature_resnet3 = build_feature_resnet_block(rgb_pool2, dim1, 'rgb_feature_resnet3')
        rgb_mapping_resnet3 = build_mapping_resnet_block(rgb_feature_resnet3, dim1, dim2, 'rgb_mapping_resnet3')
        rgb_pool3 = pool(rgb_feature_resnet3, 'pool3')
        rgb_feature_resnet4 = build_feature_resnet_block(rgb_pool3, dim1, 'rgb_feature_resnet4')
        rgb_mapping_resnet4 = build_mapping_resnet_block(rgb_feature_resnet4, dim1, dim2, 'rgb_mapping_resnet4')
        rgb_deconv1 = deconv2d(rgb_mapping_resnet4, dim1, do_norm=do_norm,
                                 name='rgb_deconv1', name_bn='bn1', name_prelu='alpha1')
        rgb_deconv1 = element_wise_linear_add(rgb_deconv1, rgb_mapping_resnet3, 'alpha4')
        rgb_deconv2 = deconv2d(rgb_deconv1, dim1,  do_norm=do_norm,
                                 name='rgb_deconv2', name_bn='bn1', name_prelu='alpha1')
        rgb_deconv2 = element_wise_linear_add(rgb_deconv2, rgb_mapping_resnet2, 'alpha5')
        rgb_deconv3 = deconv2d(rgb_deconv2, dim1,  do_norm=do_norm,
                                 name='rgb_deconv3', name_bn='bn1', name_prelu='alpha1')
        rgb_deconv3 = element_wise_linear_add(rgb_deconv3, rgb_mapping_resnet1, 'alpha6')

        out = tf.concat([depth_deconv3, rgb_deconv3], 3)
        out = conv2d(out, dim1, 3, 1, 0.02, fn='prelu', do_norm=do_norm,
                     name='out1', name_bn='bn1', name_prelu='alpha1')
        out = conv2d(out, 1, 3, 1, 0.02, do_norm=do_norm,
                     name='out2', name_bn='bn1', name_prelu='alpha1')
        out = tf.concat([depth, out], 3)
        out = PReLU(out, 'alpha7')
        out = conv2d(out, 1, 3, 1, 0.02, do_norm=False, name='out')
        print(out)
        return out


def mse_loss(output, ground):
    return tf.reduce_mean(tf.square(output-ground))


def gradient_loss(output, ground):
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])
    output_x = tf.nn.conv2d(output, sobel_x_filter, strides=[1, 1, 1, 1], padding='SAME')
    output_y = tf.nn.conv2d(output, sobel_y_filter, strides=[1, 1, 1, 1], padding='SAME')
    ground_x = tf.nn.conv2d(ground, sobel_x_filter, strides=[1, 1, 1, 1], padding='SAME')
    ground_y = tf.nn.conv2d(ground, sobel_y_filter, strides=[1, 1, 1, 1], padding='SAME')
    return tf.reduce_mean(tf.square(output_x-ground_x) + tf.square(output_y-ground_y) +
                          tf.sqrt(tf.square(output_x-ground_x) + tf.square(output_y-ground_y) + 0.01))


def cal_rmse(output, ground):
    output_ = np.array(output)
    ground_ = np.array(ground)
    output_ = (output_ + 1.0) * 127.5
    output_[output_ > 255.] = 255.0
    output_[output_ < 0.0] = 0.0
    ground_ = (ground_ + 1.0) * 127.5
    ground_[ground_ > 255.] = 255.0
    ground_[ground_ < 0.0] = 0.0
    return np.sqrt(np.mean(np.square(output_ - ground_)))
