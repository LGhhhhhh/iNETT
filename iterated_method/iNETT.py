# -*- coding:utf-8 -*-
"""
作者:admin
日期:2021年12月26
"""

import tensorflow as tf
import numpy as np
import odl
from time import time
import matplotlib.pyplot as plt

path1 = 'E:/GH/data1/'
path2 = 'E:/GH/GH_code_final/train_convex_22/'
k = 200
size = 256
s = 0.1
alpha = 2
num = 1
a = 1.e-3
lambda_regu = 5.e-4

space = odl.uniform_discr([-128, -128], [128, 128], [size, size], dtype='float32')
INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL = 256, 256, 1
OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT, OUTPUT_IMG_CHANNEL = 256, 256, 1
EPS = 10e-5

train_set = np.load(path1 + 'train_set.npy')
label_set = np.load(path1 + 'train_set_label.npy')

test_set = np.load(path1 + 'validation_set.npy')
test_label_set = np.load(path1 + 'validation_set_label.npy')

predict_set = np.load(path1 + 'predict_set.npy')
predict_label_set = np.load(path1 + 'predict_set_label.npy')
predict_set_data = np.load(path1 + 'predict_set_data.npy')
predict_true_image = np.load(path1 + 'predict_true_image.npy')
train_sample_size = 1000
test_sample_size = 200
train_set_use = np.empty([train_sample_size, 256, 256, 1])
label_set_use = np.empty([train_sample_size, 256, 256, 1])
test_set_use = np.empty([test_sample_size, 256, 256, 1])
test_label_set_use = np.empty([test_sample_size, 256, 256, 1])

for i in range(train_sample_size):
    train_set_use[i, ..., 0] = train_set[i, ..., 0]
    label_set_use[i, ..., 0] = np.abs(label_set[i, ..., 0])

for i in range(test_sample_size):
    test_set_use[i, ..., 0] = test_set[i, ..., 0]
    test_label_set_use[i, ..., 0] = np.abs(test_label_set[i, ..., 0])


class Unet:

    def __init__(self):
        print('New U-net Network')
        self.input_image = None
        self.input_label = None
        # self.keep_prob = None
        self.lamb = None
        self.loss, self.loss_mean, self.loss_all, self.train_step = [None] * 4
        self.prediction, self.correct_prediction, self.accuracy = [None] * 3
        self.result_conv = {}
        self.result_relu = {}
        self.result_maxpool = {}
        self.result_from_contract_layer = {}
        self.w = {}
        self.b = {}
        self.grad = {}
        self.RV = {}

    def init_w(self, shape, name):
        with tf.name_scope('init_w'):
            stddev = tf.sqrt(x=2 / (shape[0] * shape[1] * shape[2]))
            w = tf.Variable(initial_value=tf.truncated_normal(shape=shape, mean=0.0, stddev=stddev, dtype=tf.float32),
                            name=name)
            tf.add_to_collection(name='loss', value=tf.contrib.layers.l2_regularizer(self.lamb)(w))
            return w

    @staticmethod
    def init_b(shape, name):
        with tf.name_scope('init_b'):
            return tf.Variable(initial_value=tf.random_normal(shape=shape, dtype=tf.float32), name=name)

    @staticmethod
    def batch_norm(x, is_training, eps=EPS, decay=0.9, affine=True, name='BatchNorm2d'):
        from tensorflow.python.training.moving_averages import assign_moving_average

        with tf.variable_scope(name):
            params_shape = x.shape[-1:]
            moving_mean = tf.get_variable(name='mean', shape=params_shape, initializer=tf.zeros_initializer,
                                          trainable=False)
            moving_var = tf.get_variable(name='variance', shape=params_shape, initializer=tf.ones_initializer,
                                         trainable=False)

            def mean_var_with_update():
                mean_this_batch, variance_this_batch = tf.nn.moments(x, list(range(len(x.shape) - 1)), name='moments')
                with tf.control_dependencies([
                    assign_moving_average(moving_mean, mean_this_batch, decay),
                    assign_moving_average(moving_var, variance_this_batch, decay)
                ]):
                    return tf.identity(mean_this_batch), tf.identity(variance_this_batch)

            mean, variance = tf.cond(is_training, mean_var_with_update, lambda: (moving_mean, moving_var))
            if affine:  # 如果要用beta和gamma进行放缩
                beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)
                gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
                normed = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=beta, scale=gamma,
                                                   variance_epsilon=eps)
                tf.add_to_collection('gammas', gamma)
            else:
                normed = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=None, scale=None,
                                                   variance_epsilon=eps)
            return normed

    @staticmethod
    def copy_and_crop_and_merge(result_from_contract_layer, result_from_upsampling):

        result_from_contract_layer_crop = result_from_contract_layer
        return tf.concat(values=[result_from_contract_layer_crop, result_from_upsampling], axis=-1)

    def set_up_unet(self, batch_size):
        with tf.name_scope('input'):
            self.input_image = tf.placeholder(
                dtype=tf.float32, shape=[batch_size, INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL],
                name='input_images'
            )

            self.input_label = tf.placeholder(
                dtype=tf.float32, shape=[batch_size, OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT, OUTPUT_IMG_CHANNEL],
                name='input_labels'
            )
            #   self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
            self.lamb = tf.placeholder(dtype=tf.float32, name='lamb')
            self.is_traing = tf.placeholder(dtype=tf.bool, name='is_traing')
            normed_batch = self.batch_norm(x=self.input_image, is_training=self.is_traing, name='input')

        # layer 1
        with tf.name_scope('layer_1'):
            # conv_1
            self.w[1] = self.init_w(shape=[3, 3, INPUT_IMG_CHANNEL, 64], name='w_1')
            self.b[1] = self.init_b(shape=[64], name='b_1')
            result_conv_1 = tf.nn.conv2d(
                input=self.input_image, filter=self.w[1],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            result_conv_1 = tf.nn.bias_add(result_conv_1, self.b[1], name='conv_1_add')  # 返回特征图
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_traing,
                                           name='layer_1_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')

            # conv_2
            self.w[2] = self.init_w(shape=[3, 3, 64, 64], name='w_2')
            tf.add_to_collection('weightss', self.w[2])
            self.b[2] = self.init_b(shape=[64], name='b_2')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[2],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            result_conv_2 = tf.nn.bias_add(result_conv_2, self.b[2], name='conv_2_add')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_traing,
                                           name='layer_1_conv_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')
            self.result_from_contract_layer[1] = result_relu_2  # 该层结果临时保存, 供上采样使用

            # maxpool
            result_maxpool = tf.nn.max_pool(
                value=result_relu_2, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

        # dropout
        #   result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)

        # layer 2
        with tf.name_scope('layer_2'):
            # conv_1
            self.w[3] = self.init_w(shape=[3, 3, 64, 128], name='w_3')
            tf.add_to_collection('weightss', self.w[3])
            self.b[3] = self.init_b(shape=[128], name='b_3')
            result_conv_1 = tf.nn.conv2d(
                input=result_maxpool, filter=self.w[3],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            result_conv_1 = tf.nn.bias_add(result_conv_1, self.b[3], name='conv_1_add')
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_traing,
                                           name='layer_2_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')

            # conv_2
            self.w[4] = self.init_w(shape=[3, 3, 128, 128], name='w_4')
            tf.add_to_collection('weightss', self.w[4])
            self.b[4] = self.init_b(shape=[128], name='b_4')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[4],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            result_conv_2 = tf.nn.bias_add(result_conv_2, self.b[4], name='conv_2_add')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_traing,
                                           name='layer_2_conv_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')
            self.result_from_contract_layer[2] = result_relu_2  # 该层结果临时保存, 供上采样使用

            # maxpool
            result_maxpool = tf.nn.max_pool(
                value=result_relu_2, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

        # dropout
        # result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)

        # layer 3
        with tf.name_scope('layer_3'):
            # conv_1
            self.w[5] = self.init_w(shape=[3, 3, 128, 256], name='w_5')
            tf.add_to_collection('weightss', self.w[5])
            self.b[5] = self.init_b(shape=[256], name='b_5')
            result_conv_1 = tf.nn.conv2d(
                input=result_maxpool, filter=self.w[5],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            result_conv_1 = tf.nn.bias_add(result_conv_1, self.b[5], name='conv_1_add')
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_traing,
                                           name='layer_3_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')

            # conv_2
            self.w[6] = self.init_w(shape=[3, 3, 256, 256], name='w_6')
            tf.add_to_collection('weightss', self.w[6])
            self.b[6] = self.init_b(shape=[256], name='b_6')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[6],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            result_conv_2 = tf.nn.bias_add(result_conv_2, self.b[6], name='conv_2_add')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_traing,
                                           name='layer_3_conv_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')
            self.result_from_contract_layer[3] = result_relu_2  # 该层结果临时保存, 供上采样使用

            # maxpool
            result_maxpool = tf.nn.max_pool(
                value=result_relu_2, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

        # dropout
        # result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)

        # layer 4
        with tf.name_scope('layer_4'):
            # conv_1
            self.w[7] = self.init_w(shape=[3, 3, 256, 512], name='w_7')
            tf.add_to_collection('weightss', self.w[7])
            self.b[7] = self.init_b(shape=[512], name='b_7')
            result_conv_1 = tf.nn.conv2d(
                input=result_maxpool, filter=self.w[7],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            result_conv_1 = tf.nn.bias_add(result_conv_1, self.b[7], name='conv_1_add')
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_traing,
                                           name='layer_4_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')

            # conv_2
            self.w[8] = self.init_w(shape=[3, 3, 512, 512], name='w_8')
            tf.add_to_collection('weightss', self.w[8])
            self.b[8] = self.init_b(shape=[512], name='b_8')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[8],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            result_conv_2 = tf.nn.bias_add(result_conv_2, self.b[8], name='conv_2_add')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_traing,
                                           name='layer_4_conv_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')
            self.result_from_contract_layer[4] = result_relu_2  # 该层结果临时保存, 供上采样使用

            # maxpool
            result_maxpool = tf.nn.max_pool(
                value=result_relu_2, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

        # dropout
        # result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)

        # layer 5 (bottom)
        with tf.name_scope('layer_5'):
            # conv_1
            self.w[9] = self.init_w(shape=[3, 3, 512, 1024], name='w_9')
            tf.add_to_collection('weightss', self.w[9])
            self.b[9] = self.init_b(shape=[1024], name='b_9')
            result_conv_1 = tf.nn.conv2d(
                input=result_maxpool, filter=self.w[9],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            result_conv_1 = tf.nn.bias_add(result_conv_1, self.b[9], name='conv_1_add')
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_traing,
                                           name='layer_5_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')

            # conv_2
            self.w[10] = self.init_w(shape=[3, 3, 1024, 1024], name='w_10')
            tf.add_to_collection('weightss', self.w[10])
            self.b[10] = self.init_b(shape=[1024], name='b_10')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[10],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            result_conv_2 = tf.nn.bias_add(result_conv_2, self.b[10], name='conv_1_add')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_traing,
                                           name='layer_5_conv_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')

            # up sample
            self.w[11] = self.init_w(shape=[2, 2, 512, 1024], name='w_11')
            tf.add_to_collection('weightss', self.w[11])
            self.b[11] = self.init_b(shape=[512], name='b_11')
            result_up = tf.nn.conv2d_transpose(
                value=result_relu_2, filter=self.w[11],
                output_shape=[batch_size, 32, 32, 512],
                strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
            result_up = tf.nn.bias_add(result_up, self.b[11], name='add_bias')
            normed_batch = self.batch_norm(x=result_up, is_training=self.is_traing, name='layer_5_conv_3')
            result_relu_3 = tf.nn.relu(normed_batch, name='relu_3')

        # dropout
        # result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)

        # layer 6
        with tf.name_scope('layer_6'):
            # copy, crop and merge
            result_merge = self.copy_and_crop_and_merge(
                result_from_contract_layer=self.result_from_contract_layer[4], result_from_upsampling=result_relu_3)
            # print(result_merge)

            # conv_1
            self.w[12] = self.init_w(shape=[3, 3, 1024, 512], name='w_12')
            tf.add_to_collection('weightss', self.w[12])
            self.b[12] = self.init_b(shape=[512], name='b_12')
            result_conv_1 = tf.nn.conv2d(
                input=result_merge, filter=self.w[12],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            result_conv_1 = tf.nn.bias_add(result_conv_1, self.b[12], name='conv_1_add')
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_traing,
                                           name='layer_6_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')

            # conv_2
            self.w[13] = self.init_w(shape=[3, 3, 512, 512], name='w_13')
            tf.add_to_collection('weightss', self.w[13])
            self.b[13] = self.init_b(shape=[512], name='b_13')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[13],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            result_conv_2 = tf.nn.bias_add(result_conv_2, self.b[13], name='conv_2_add')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_traing,
                                           name='layer_6_conv_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')
            # print(result_relu_2.shape[1])

            # up sample
            self.w[14] = self.init_w(shape=[2, 2, 256, 512], name='w_14')
            tf.add_to_collection('weightss', self.w[14])
            self.b[14] = self.init_b(shape=[256], name='b_14')
            result_up = tf.nn.conv2d_transpose(
                value=result_relu_2, filter=self.w[14],
                output_shape=[batch_size, 64, 64, 256],
                strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
            result_up = tf.nn.bias_add(result_up, self.b[14], name='add_bias')
            normed_batch = self.batch_norm(x=result_up, is_training=self.is_traing, name='layer_6_conv_3')
            result_relu_3 = tf.nn.relu(normed_batch, name='relu_3')

        # dropout
        # result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)

        # layer 7
        with tf.name_scope('layer_7'):
            # copy, crop and merge
            result_merge = self.copy_and_crop_and_merge(
                result_from_contract_layer=self.result_from_contract_layer[3], result_from_upsampling=result_relu_3)

            # conv_1
            self.w[15] = self.init_w(shape=[3, 3, 512, 256], name='w_15')
            tf.add_to_collection('weightss', self.w[15])
            self.b[15] = self.init_b(shape=[256], name='b_15')
            result_conv_1 = tf.nn.conv2d(
                input=result_merge, filter=self.w[15],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            result_conv_1 = tf.nn.bias_add(result_conv_1, self.b[15], name='conv_1_add')
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_traing,
                                           name='layer_7_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')

            # conv_2
            self.w[16] = self.init_w(shape=[3, 3, 256, 256], name='w_16')
            tf.add_to_collection('weightss', self.w[16])
            self.b[16] = self.init_b(shape=[256], name='b_16')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[16],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            result_conv_2 = tf.nn.bias_add(result_conv_2, self.b[16], name='conv_2_add')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_traing,
                                           name='layer_7_conv_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')

            # up sample
            self.w[17] = self.init_w(shape=[2, 2, 128, 256], name='w_17')
            tf.add_to_collection('weightss', self.w[17])
            self.b[17] = self.init_b(shape=[128], name='b_17')
            result_up = tf.nn.conv2d_transpose(
                value=result_relu_2, filter=self.w[17],
                output_shape=[batch_size, 128, 128, 128],
                strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
            result_up = tf.nn.bias_add(result_up, self.b[17], name='add_bias')
            normed_batch = self.batch_norm(x=result_up, is_training=self.is_traing, name='layer_7_conv_3')
            result_relu_3 = tf.nn.relu(normed_batch, name='relu_3')

        # dropout
        # result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)

        # layer 8
        with tf.name_scope('layer_8'):
            # copy, crop and merge
            result_merge = self.copy_and_crop_and_merge(
                result_from_contract_layer=self.result_from_contract_layer[2], result_from_upsampling=result_relu_3)
            # conv_1
            self.w[18] = self.init_w(shape=[3, 3, 256, 128], name='w_18')
            tf.add_to_collection('weightss', self.w[18])
            self.b[18] = self.init_b(shape=[128], name='b_18')
            result_conv_1 = tf.nn.conv2d(
                input=result_merge, filter=self.w[18],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            result_conv_1 = tf.nn.bias_add(result_conv_1, self.b[18], name='conv_1_add')
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_traing,
                                           name='layer_8_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')

            # conv_2
            self.w[19] = self.init_w(shape=[3, 3, 128, 128], name='w_19')
            tf.add_to_collection('weightss', self.w[19])
            self.b[19] = self.init_b(shape=[128], name='b_19')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[19],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            result_conv_2 = tf.nn.bias_add(result_conv_2, self.b[19], name='conv_2_add')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_traing,
                                           name='layer_8_conv_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')

            # up sample
            self.w[20] = self.init_w(shape=[2, 2, 64, 128], name='w_20')
            tf.add_to_collection('weightss', self.w[20])
            self.b[20] = self.init_b(shape=[64], name='b_20')
            result_up = tf.nn.conv2d_transpose(
                value=result_relu_2, filter=self.w[20],
                output_shape=[batch_size, 256, 256, 64],
                strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
            result_up = tf.nn.bias_add(result_up, self.b[20], name='add_bias')
            normed_batch = self.batch_norm(x=result_up, is_training=self.is_traing, name='layer_8_conv_3')
            result_relu_3 = tf.nn.relu(normed_batch, name='relu_3')

        # dropout
        # result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)

        # layer 9
        with tf.name_scope('layer_9'):
            # copy, crop and merge
            result_merge = self.copy_and_crop_and_merge(
                result_from_contract_layer=self.result_from_contract_layer[1], result_from_upsampling=result_relu_3)

            # conv_1
            self.w[21] = self.init_w(shape=[3, 3, 128, 64], name='w_21')
            tf.add_to_collection('weightss', self.w[21])
            self.b[21] = self.init_b(shape=[64], name='b_21')
            result_conv_1 = tf.nn.conv2d(
                input=result_merge, filter=self.w[21],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            result_conv_1 = tf.nn.bias_add(result_conv_1, self.b[21], name='conv_1_add')
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_traing,
                                           name='layer_9_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')

            # conv_2
            self.w[22] = self.init_w(shape=[3, 3, 64, 64], name='w_22')
            tf.add_to_collection('weightss', self.w[22])
            self.b[22] = self.init_b(shape=[64], name='b_22')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[22],
                strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            result_conv_2 = tf.nn.bias_add(result_conv_2, self.b[22], name='conv_2_add')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_traing,
                                           name='layer_9_conv_2')
            result_relu_2 = tf.nn.relu(normed_batch, name='relu_2')

            # convolution to [batch_size, OUTPIT_IMG_WIDE, OUTPUT_IMG_HEIGHT, CLASS_NUM]
            self.w[23] = self.init_w(shape=[1, 1, 64, 1], name='w_23')
            tf.add_to_collection('weightss', self.w[23])
            self.b[23] = self.init_b(shape=[1], name='b_23')
            tf.add_to_collection('betas', self.b[23])
            result_conv_3 = tf.nn.conv2d(
                input=result_relu_2, filter=self.w[23],
                strides=[1, 1, 1, 1], padding='VALID', name='conv_3')
            self.prediction = tf.nn.bias_add(result_conv_3, self.b[23], name='conv_3_add')

            self.RV = tf.square(tf.norm(self.prediction, ord=2))

        # softmax loss
        with tf.name_scope('squre_loss'):
            self.loss_mean = tf.reduce_mean((self.input_label - self.prediction) ** 2)
            tf.add_to_collection(name='loss', value=self.loss_mean)
            self.loss_all = tf.add_n(inputs=tf.get_collection(key='loss'))

        # accuracy
        with tf.name_scope('accuracy'):
            self.accuracy = self.loss

        # gradients
        with tf.name_scope('gradients'):
            self.grad = tf.gradients(ys=self.RV, xs=self.input_image, name='gradients')

        # Gradient Descent
        with tf.name_scope('Gradient_Descent'):
            learning_rate = 0.0005
            self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_all)

    def train(self, batch_size):
        valid_X = test_set_use
        valid_Y = test_label_set_use
        train_num = 20000

        train_loss_mean = np.empty([train_num], dtype=np.float32)
        train_loss_all = np.empty([train_num], dtype=np.float32)
        valid_loss_mean = np.empty([train_num], dtype=np.float32)
        valid_loss_all = np.empty([train_num], dtype=np.float32)

        tf.summary.scalar("loss_mean", self.loss_mean)
        tf.summary.scalar('loss_all', self.loss_all)
        tf.summary.image('prediction', self.prediction)
        merged_summary = tf.summary.merge_all()

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            writer = tf.summary.FileWriter(path2 + "result/log_train01", tf.get_default_graph())
            D_var = tf.get_collection('weightss')
            clip_ops = []
            for var in D_var:
                print("I AM gv:", var)
                print(var.get_shape())
                print("-----------")
                clip_ops.append(
                    tf.assign(var, tf.clip_by_value(var, 0.0, tf.reduce_max(var)))
                )
            E_var = tf.get_collection('gammas')
            for v in E_var:
                print("I AM gamma:", v)
                clip_ops.append(
                    tf.assign(v, tf.clip_by_value(v, 0.0, tf.reduce_max(v)))
                )
            clip_disc_weights = tf.group(*clip_ops)
            for i in range(train_num):
                start = (i * batch_size) % train_sample_size
                end = min(start + batch_size, train_sample_size)

                if i % train_sample_size == 0:
                    index = np.arange(train_sample_size)
                    np.random.shuffle(index)

                index_m = index[start:end]

                x = train_set_use[index_m]
                y = label_set_use[index_m]

                v_start = (i * batch_size) % test_sample_size
                v_end = min(v_start + batch_size, test_sample_size)
                valid_x = valid_X[v_start:v_end]
                valid_y = valid_Y[v_start:v_end]
                _ = sess.run(
                    self.train_step,
                    feed_dict={self.input_image: x, self.input_label: y, self.lamb: lambda_regu, self.is_traing: True}
                )
                _ = sess.run(clip_disc_weights)

                t_loss_mean, t_loss_all, y_, summary_str = sess.run(
                    [self.loss_mean, self.loss_all, self.prediction, merged_summary],
                    feed_dict={self.input_image: x, self.input_label: y, self.lamb: lambda_regu, self.is_traing: True}
                )

                v_loss_mean, v_loss_all = sess.run(
                    [self.loss_mean, self.loss_all],
                    feed_dict={self.input_image: valid_x, self.input_label: valid_y, self.lamb: lambda_regu,
                               self.is_traing: False})

                train_loss_all[i] = t_loss_all
                train_loss_mean[i] = t_loss_mean
                valid_loss_all[i] = v_loss_all
                valid_loss_mean[i] = v_loss_mean

                writer.add_summary(summary_str, i)

                if True:
                    print(i)
                    print("train: loss:%.6f,  loss_mean:%.6f ; valid: loss:%.6f, loss_mean:%.6f" % (
                        t_loss_all, t_loss_mean, v_loss_all, v_loss_mean))
                    print("------------------------------------------------------------")
        writer.close()
        print("Done training")

    def predict(self, batch_size, A, b, k, s, alpha, num, eta):
        # tf.summary.scalar("loss_mean", self.loss_mean)
        # tf.summary.scalar('accuracy', self.accuracy)
        # tf.summary.image('prediction', self.prediction)
        # merged_summary = tf.summary.merge_all()
        ckpt_path = path2 + "result/19999unet01.ckpt"
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess=sess, save_path=ckpt_path)

            m = np.size(A, 0)  # 15360（256*60）
            n = np.size(A, 1)  # 65536（256*256）
            x = np.ones(shape=(n, 1)) / n  # 图像列向量
            x_input = np.ones(shape=(1, 256, 256, 1)) / n  # 对应图像的神经网络输入
            y = np.zeros(shape=(1, 256, 256, 1))  # 神经网络需要的形式参数，因为没有训练神经网络，事实上未使用
            E1 = np.zeros(shape=(k * num, 1))  # 损失函数的值：数据拟合项
            E2 = np.zeros(shape=(k * num, 1))  # 损失函数的值： 正则项，包括正则化参数
            count = 0

            # iter2对应n，iter1对应k
            for iter2 in range(num):
                alpha_n = 1 / np.power(alpha, iter2 + 1)
                xn = x  # x_{n-1}
                for iter1 in range(k):
                    x_input[0, ..., 0] = np.reshape(x, [256, 256])  # 对应图像的神经网络输入
                    gradients, RV = sess.run(
                        [self.grad, self.RV],
                        feed_dict={self.input_image: x_input, self.input_label: y, self.lamb: 1.e-4,
                                   self.is_traing: False}
                    )

                    grad = np.reshape(gradients, [n, 1]) + 2.0 * a * x

                    cha = np.dot(A, x) - b  # FX-b
                    tidu = np.dot(A.T, cha)  # F.T(FX-b)

                    if iter2 < 1 and iter1 < 1:
                        xi = grad

                    if iter1 < 1:  # k==0
                        R_previous = RV + a * np.linalg.norm(xn) ** 2

                    dx = x - xn
                    temp = np.dot(xi.T, dx)
                    bregman = RV + a * np.linalg.norm(x) ** 2 - R_previous - temp
                    E1[count] = 0.5 * np.linalg.norm(cha) ** 2 * 1 / m
                    E2[count] = alpha_n * bregman
                    print(count, E1[count], E2[count], temp, np.linalg.norm(cha))

                    x = x - s * (1 / m * tidu + alpha_n * (grad - xi))

                    count = count + 1

                NUM = str((iter2 + 1) * k)
                np.save(path2 + 'result/prediction' + NUM + '.npy', x_input[0, ..., 0])

                ### 迭代xi ###
                xi_cha = np.dot(A, x) - b  # FX-b
                xi_tidu = np.dot(A.T, xi_cha)  # F.T(FX-b)
                xi = xi - 1 / m / alpha_n * xi_tidu
                ################
                A_ = np.linalg.norm(xi_cha)
                if A_ <= eta:
                    print(A_, iter2)
                    break

        np.save(path2 + 'result/19999/E2.npy', E2)
        np.save(path2 + 'result/19999/E1.npy', E1)
        np.save(path2 + 'result/19999/prediction.npy', x_input[0, ..., 0])
        print("Done predicting")


if __name__ == '__main__':
    start_time = time()
    batch_size = 1
    net = Unet()
    net.set_up_unet(batch_size)
    i = 14
    k = 2000
    size = 256
    s = 0.01
    alpha = 1.1
    num = 10
    p = predict_set_data[i]
    y = np.dot(w, np.reshape(predict_true_image[i, ..., 0], [256 * 256, 1]))
    y_delta = predict_set_data[i]
    eta = np.linalg.norm(y_delta - y)
    # y = np.dot(w, np.reshape(IMG1, [256 * 256, 1]))
    # p = np.load(path2 + 'result/heart_data.npy')
    # y_delta = np.load(path2 + 'result/heart_data.npy')
    # eta = np.linalg.norm(y_delta - y)
    net.predict(batch_size, w, p, k, s, alpha, num, 1.01 * eta)
    end_time = time()
    print(end_time - start_time)
