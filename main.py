# coding=utf-8
import argparse
import os
import tensorflow as tf
from model import srcnn


parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase',                 type=str,     default='train', help='train or test')
parser.add_argument('--rgb_image_dir',         type=str,     default='rgb_128_128')  # 彩色图路径
parser.add_argument('--depth_image_dir',       type=str,     default='depth_8_8_2')  # 深度图路径
parser.add_argument('--ground_image_dir',      type=str,     default='depth_128_128')  # 原图路径
parser.add_argument('--log_dir',               type=str,     default=os.path.join('.', 'log'))
parser.add_argument('--ckpt_dir',              type=str,     default=os.path.join('.', 'checkpoint'))
parser.add_argument('--sample_dir',            type=str,     default=os.path.join('.', 'sample'))  # 训练中输出
parser.add_argument('--test_dir',              type=str,     default=os.path.join('.', 'test'))  # 测试输出路径
parser.add_argument('--image_height',          type=int,     default=8)
parser.add_argument('--image_width',           type=int,     default=8)
parser.add_argument('--test_height',           type=int,     default=30)
parser.add_argument('--test_width',            type=int,     default=40)
parser.add_argument('--gpu_number',            type=str,     default='0')
parser.add_argument('--SR_times',              type=int,     default=16)   # 超分辨率倍数
parser.add_argument('--SR_times_dir',          type=str,     default='train_16x')
parser.add_argument('--batch_size',            type=int,     default=16)
parser.add_argument('--epoch',                 type=int,     default=16)
parser.add_argument('--learning_rate',         type=int,     default=0.001)
parser.add_argument('--continue_train',        type=bool,    default=False)
parser.add_argument('--test_depth_dir',        type=str,     default='depth_16x')  # 测试深度图路径
parser.add_argument('--test_rgb_dir',          type=str,     default='rgb')   # 测试rgb图路径
parser.add_argument('--test_ground_dir',       type=str,     default='ground')  # 测试原图路径
args = parser.parse_args()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number
    tf.reset_default_graph()
    image_dir = os.path.join('E:/depth_datasets', args.SR_times_dir)
    args.rgb_image_dir = os.path.join(image_dir, args.rgb_image_dir)  # 低分辨率彩色图路径
    args.depth_image_dir = os.path.join(image_dir, args.depth_image_dir)  # 低分辨深度图路径
    args.ground_image_dir = os.path.join(image_dir, args.ground_image_dir)  # 高分辨率深度图路径
    args.sample_dir = os.path.join(args.sample_dir, args.SR_times_dir)  # 检验样本输出路径
    args.log_dir = os.path.join(args.log_dir, args.SR_times_dir)
    args.ckpt_dir = os.path.join(args.ckpt_dir, args.SR_times_dir)
    args.test_dir = os.path.join(args.test_dir, args.SR_times_dir)
    args.test_depth_dir = os.path.join(args.test_dir, args.test_depth_dir)
    args.test_rgb_dir = os.path.join(args.test_dir, args.test_rgb_dir)
    args.test_ground_dir = os.path.join(args.test_dir, args.test_ground_dir)


    try:
        os.makedirs(args.sample_dir)
    except:
        pass
    try:
        os.makedirs(args.log_dir)
    except:
        pass
    try:
        os.makedirs(args.ckpt_dir)
    except:
        pass
    try:
        os.makedirs(args.test_dir)
    except:
        pass

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = srcnn(sess, args)
        model.train() if args.phase == 'train' else model.test()


if __name__ == '__main__':
    main()
