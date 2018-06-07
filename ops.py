import tensorflow as tf
import tensorflow.contrib.slim as slim


def batch_norm(x, name='batch_norm'):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)


def instance_norm(x, name='instance_norm'):
    with tf.variable_scope(name):
        depth = x.get_shape()[-1]
        mean, var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        scale = tf.get_variable('scale', [depth],
                                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02, dtype=tf.float32))
        offset = tf.get_variable('offset', [depth], initializer=tf.constant_initializer(0.0))
        return scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset


def PReLU(x, name):
    with tf.variable_scope(name):
        alphas = tf.get_variable(name, x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alphas * (x - abs(x)) * 0.5
        return pos + neg


def conv2d(input_, output_dim, ks=3, s=1, stddev=0.02, padding="SAME",
           fn=None, do_norm=True, name="conv2d", name_bn="bn", name_prelu=None):
    with tf.variable_scope(name):
        conv = slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                           weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                           biases_initializer=None)
        if do_norm:
            conv = instance_norm(conv, name=name_bn)
        if fn == "relu":
            conv = tf.nn.relu(conv)
        elif fn == "lrelu":
            conv = tf.maximum(conv, 0.2*conv)
        elif fn == "tanh":
            conv = tf.nn.tanh(conv)
        elif fn == 'prelu':
            conv = PReLU(conv, name_prelu)
        return conv


def deconv2d(input, output_dim, ks=5, s=2, stddev=0.02, fn=None, do_norm=True,
             name="deconv2d", name_bn=None, name_prelu=None):
    with tf.variable_scope(name):
        conv = slim.conv2d_transpose(input, output_dim, ks, s, padding="SAME", activation_fn=None,
                                     weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                     biases_initializer=None)
        if do_norm:
            conv = instance_norm(conv, name_bn)
        if fn == "relu":
            conv = tf.nn.relu(conv)
        elif fn == "lrelu":
            conv = tf.maximum(conv, 0.2 * conv)
        elif fn == "tanh":
            conv = tf.nn.tanh(conv)
        elif fn == 'prelu':
            conv = PReLU(conv, name_prelu)
        return conv


def element_wise_linear_add(x1, x2, name):
    alpha = tf.get_variable(name=name, shape=[1], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.1, dtype=tf.float32))
    return tf.add(alpha * x1, (1 - alpha) * x2)


def build_feature_resnet_block(x, dim, name='resnet', do_norm=False):
    with tf.variable_scope(name):
        out_res = conv2d(x, dim, 3, 1, 0.02, fn="prelu", do_norm=do_norm, name="conv1",
                         name_bn='bn1', name_prelu='alpha1')
        out_res = conv2d(out_res, dim, 3, 1, 0.02, do_norm=do_norm, name="conv2",
                         name_bn='bn2', name_prelu='alpha2')
        out_res = element_wise_linear_add(x, out_res, 'liner1')
        return out_res


def build_mapping_resnet_block(x, dim1, dim2, name='resnet', do_norm=False):
    with tf.variable_scope(name):
        out_res = PReLU(x, "alpha0")
        out_res = conv2d(out_res, dim2, 1, 1, 0.02, fn="prelu", do_norm=do_norm, name="conv1",
                         name_bn='bn1', name_prelu='alpha1')
        out_res = conv2d(out_res, dim2, 3, 1, 0.02, fn="prelu", do_norm=do_norm, name="conv2",
                         name_bn='bn2', name_prelu='alpha2')
        out_res = conv2d(out_res, dim1, 1, 1, 0.02, do_norm=do_norm, name="conv3",
                         name_bn='bn3', name_prelu='alpha3')
        out_res = element_wise_linear_add(x, out_res, 'liner1')
        out_res = PReLU(out_res, 'alpha4')
        return out_res


def pool(x, name):
    return slim.max_pool2d(x, [2, 2], 2, padding='SAME', scope=name)
