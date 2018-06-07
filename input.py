import tensorflow as tf
import numpy as np
import os
import scipy.misc


def get_all_files(file_path):
    filename_list = []

    for item in os.listdir(file_path):
        path = os.path.join(file_path, item)
        if os.path.isdir(path):     # 如果是文件夹
            filename_list.extend(get_all_files(path))
        elif os.path.isfile(path):  # 如果是文件
            filename_list.append(path)

    return filename_list


def get_image_list(depth_dir, ground_dir, rgb_dir, phase='train'):
    depth_list = get_all_files(depth_dir)       # 彩色图片
    ground_list = get_all_files(ground_dir)       # 颜色主题
    rgb_list = get_all_files(rgb_dir)       # 颜色标签

    print("训练目录%s, 文件个数%d" % (depth_dir, len(depth_list)))
    print("训练目录%s, 文件个数%d" % (ground_dir, len(ground_list)))
    print("训练目录%s, 文件个数%d" % (rgb_dir, len(rgb_list)))

    temp = np.array([depth_list, ground_list, rgb_list])
    temp = temp.transpose()
    if phase == 'train':
        np.random.shuffle(temp)

    depth_list = list(temp[:, 0])
    ground_list = list(temp[:, 1])
    rgb_list = list(temp[:, 2])

    return [depth_list, ground_list, rgb_list]


def get_batch(image_list, depth_list, ground_list, width, height, batch_size, SR_times, phase, capacity=2000):
    image = tf.cast(image_list, tf.string)
    ground = tf.cast(ground_list, tf.string)
    depth = tf.cast(depth_list, tf.string)
    input_queue = tf.train.slice_input_producer([image, ground, depth], shuffle=False)
    image_contents = tf.read_file(input_queue[0])
    ground_contents = tf.read_file(input_queue[1])
    depth_contants = tf.read_file(input_queue[2])
    image = tf.image.decode_png(image_contents)
    ground = tf.image.decode_png(ground_contents)
    depth = tf.image.decode_png(depth_contants)
    image = tf.reshape(image, [width*SR_times, height*SR_times, 3])
    depth = tf.reshape(depth, [width, height, 1])
    ground = tf.reshape(ground, [width*SR_times, height*SR_times, 1])
    image = tf.cast(image, tf.float32)
    image = image/255.0
    ground = tf.cast(ground, tf.float32)
    ground = ground/255.0
    depth = tf.cast(depth, tf.float32)
    depth = depth/255.0
    if phase == 'train':
        image_batch, ground_batch, depth_batch = tf.train.batch([image, ground, depth],
                                                                batch_size=batch_size,
                                                                num_threads=64,  # 线程
                                                                capacity=capacity)
    else:
        image_batch, ground_batch, depth_batch = tf.train.batch([image, ground, depth],
                                                                batch_size=batch_size,
                                                                num_threads=1,  # 线程
                                                                capacity=capacity)
    return image_batch, depth_batch, ground_batch


def get_image(image_path, image_height, image_width, image_channel, phase='train'):
    image = scipy.misc.imread(image_path)
    image_resize = scipy.misc.imresize(image, [image_height, image_width, image_channel])
    image_resize = image_resize/127.5 - 1  # pixel range in [-1, 1]
    # if phase == 'train' and np.random.random() >= 0.5:
    #     image_resize = np.flip(image_resize, 1)
    return image_resize


def process_image(depth_list, rgb_list, ground_list, image_height, image_width, sr_times, phase='train'):
    depth = [get_image(image_path, image_height, image_width, 1) for image_path in depth_list]
    depth = np.array(depth)
    depth = np.expand_dims(depth, axis=3)

    rgb = [get_image(image_path, image_height * sr_times, image_width * sr_times, 3) for image_path in rgb_list]
    rgb = np.array(rgb)

    ground = [get_image(image_path, image_height * sr_times, image_width * sr_times, 1) for image_path in ground_list]
    ground = np.array(ground)
    ground = np.expand_dims(ground, axis=3)

    return depth, rgb, ground




