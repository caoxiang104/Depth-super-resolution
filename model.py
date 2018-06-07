import tensorflow as tf
import numpy as np
import os
import time
import scipy.misc

from module import network, mse_loss, gradient_loss, cal_rmse
from input import get_image_list, process_image


class srcnn(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.learning_rate = args.learning_rate
        self.phase = args.phase
        self.log_dir = args.log_dir
        self.rgb_image_dir = args.rgb_image_dir
        self.depth_image_dir = args.depth_image_dir
        self.ground_image_dir = args.ground_image_dir
        self.ckpt_dir = args.ckpt_dir
        self.sample_dir = args.sample_dir
        self.test_dir = args.test_dir
        self.image_height = args.image_height
        self.image_width = args.image_width
        self.sr_times = args.SR_times
        self.continue_train = args.continue_train
        self.test_depth_dir = args.test_depth_dir
        self.test_rgb_dir = args.test_rgb_dir
        self.test_ground_dir = args.test_ground_dir
        self.test_height = args.test_height
        self.test_width = args.test_width

        # build model and make checkpoint saver
        self.build_model()
        self.saver = tf.train.Saver()

    def build_model(self):
        # placeholder
        # input_image: depth, rgb
        self.depth_image = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, 1], name='depth_image')
        self.rgb_image = tf.placeholder(tf.float32, [None, self.image_height * self.sr_times,
                                                     self.image_width * self.sr_times, 3], name='rgb_image')
        self.ground_image = tf.placeholder(tf.float32, [None, self.image_height * self.sr_times,
                                                        self.image_width * self.sr_times, 1], name='ground_image')
        self.test_depth_image = tf.placeholder(tf.float32,
                                               [None, self.test_height, self.test_width, 1], name='test_depth')
        self.test_rgb_image = tf.placeholder(tf.float32, [None, self.test_height * self.sr_times,
                                                          self.test_width * self.sr_times, 3], name='test_rgb')
        self.test_ground_image = tf.placeholder(tf.float32, [None, self.test_height * self.sr_times,
                                                self.test_width * self.sr_times, 1], name='test_ground')
        # generator high resolution image
        self.hr_image = network(self.depth_image, self.rgb_image, sr_times=self.sr_times, reuse=False)

        # loss
        self.mse_loss = mse_loss(self.hr_image, self.ground_image)
        self.gradient_loss = gradient_loss(self.hr_image, self.ground_image)
        self.loss =  self.mse_loss + self.gradient_loss

        # learning rate decay
        self.lr_decay = tf.placeholder(tf.float32, [], name='lr_decay')

        # trainable variables
        t_vars = tf.trainable_variables()
        self.vars = [var for var in t_vars if 'network' in var.name]
        # print var name in trainable variables
        for var in t_vars: print(var.name)
        # optimize
        self.optim = tf.train.AdamOptimizer(self.learning_rate * self.lr_decay, beta1=0.5).minimize(self.loss,
                                                                                                    var_list=self.vars)

    def train(self):
        self.summary()
        depth_file, rgb_file, ground_file = get_image_list(self.depth_image_dir,
                                                           self.rgb_image_dir, self.ground_image_dir)
        self.sess.run(tf.global_variables_initializer())
        # load or not checkpoint
        if self.continue_train and self.checkpoint_load():
            print(" [*] before training, Load SUCCESS ")
        else:
            print(" [!] before training, no need to Load ")
        batch_idexs = len(depth_file) // self.batch_size
        count = 0
        # train
        start_time = time.time()
        for epoch in range(self.epoch):
            print("epoch:{}".format(epoch + 1))
            if epoch < 5:
                lr_decay = 1.0
            else:
                lr_decay = 1.0 - (epoch - 5)/5
            for idx in range(batch_idexs):
                depth_list = depth_file[idx * self.batch_size: (idx + 1) * self.batch_size]
                rgb_list = rgb_file[idx * self.batch_size: (idx + 1) * self.batch_size]
                ground_list = ground_file[idx * self.batch_size: (idx + 1) * self.batch_size]
                count += 1
                depth, rgb, ground = process_image(depth_list, rgb_list, ground_list, self.image_height,
                                                   self.image_width, self.sr_times)
                feed = {self.depth_image: depth, self.rgb_image: rgb, self.ground_image: ground, self.lr_decay: lr_decay}
                _, global_loss, mse_l, gradient_l, summ = self.sess.run([self.optim, self.loss, self.mse_loss,
                                                                        self.gradient_loss, self.sum], feed_dict=feed)
                if idx % 100 == 0:
                    durarion = time.time() - start_time
                    start_time = time.time()
                    print("Iter: %06d, global_loss: %4.4f, mse_loss: %4.4f, gradient_loss: %4.4f, time: %4.4f"
                          % (count, global_loss, mse_l, gradient_l, durarion))
                    self.writer.add_summary(summ, count)
            self.checkpoint_save(count)
            self.sample_save(self.sess, count)

    def test(self):
        # variable initialize
        self.sess.run(tf.global_variables_initializer())

        # load or not checkpoint
        if self.phase == 'test' and self.checkpoint_load():
            print(" [*] before training, Load SUCCESS ")
        else:
            print(" [!] before training, no need to Load ")

        test_depth_file, test_rgb_file, test_ground_file = get_image_list(self.test_depth_dir, self.test_rgb_dir,
                                                                          self.test_ground_dir, phase='test')
        depth_test, rgb_test, ground_test = process_image(test_depth_file, test_rgb_file, test_ground_file,
                                                          self.test_height, self.test_width, self.sr_times)
        self.test_hr_image = network(self.test_depth_image, self.test_rgb_image, sr_times=self.sr_times, reuse=True)
        hr_image_temp = self.sess.run([self.test_hr_image],
                                      feed_dict={self.test_depth_image: depth_test, self.test_rgb_image: rgb_test})
        print("average RMSE is: %4.4f" % cal_rmse(hr_image_temp, ground_test))
        for i in range(6):
            temp = hr_image_temp[0][i]
            temp = (temp.reshape([self.test_height * self.sr_times, self.test_width * self.sr_times]) + 1.) * 127.5
            temp[temp > 255] = 255.
            temp[temp < 0] = 0.
            temp = temp.astype(np.uint8)
            scipy.misc.imsave(self.test_dir + '/' + str(i) + '.png', temp)

    def sample_save(self, sess, step):
        test_depth_file, test_rgb_file, test_ground_file = get_image_list(self.test_depth_dir, self.test_rgb_dir,
                                                                          self.test_ground_dir, phase='test')
        depth_test, rgb_test, ground_test = process_image(test_depth_file, test_rgb_file, test_ground_file,
                                                          self.test_height, self.test_width, self.sr_times)
        self.test_hr_image = network(self.test_depth_image, self.test_rgb_image, sr_times=self.sr_times, reuse=True)
        hr_image_temp = sess.run([self.test_hr_image],
                                 feed_dict={self.test_depth_image: depth_test, self.test_rgb_image: rgb_test})
        print("average RMSE is: %4.4f" % cal_rmse(hr_image_temp, ground_test))
        for i in range(6):
            temp = hr_image_temp[0][i]
            temp = (temp.reshape([self.test_height * self.sr_times, self.test_width * self.sr_times]) + 1.) * 127.5
            temp[temp > 255] = 255.
            temp[temp < 0] = 0.
            temp = temp.astype(np.uint8)
            scipy.misc.imsave(self.sample_dir + '/' + str(step) + '_' + str(i) + '.png', temp)

    def checkpoint_load(self):
        print(" [*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            # ckpt_name = 'srcnn.model-80000'
            self.saver.restore(self.sess, os.path.join(self.ckpt_dir, ckpt_name))
            return True
        else:
            return False

    def checkpoint_save(self, step):
        model_name = "srcnn.model"
        self.saver.save(self.sess,
                        os.path.join(self.ckpt_dir, model_name),
                        global_step=step)

    def summary(self):
        mse = tf.summary.scalar('mse', self.mse_loss)
        gradient = tf.summary.scalar('gradient', self.gradient_loss)
        total = tf.summary.scalar('total_loss', self.loss)
        self.sum = tf.summary.merge([mse, gradient, total])
        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)




