# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import paddle
import paddle.fluid as fluid
import contextlib
import math
import sys
import numpy
import unittest
import os
import numpy as np
import time


def resnet_cifar10(input, depth=32):
    def conv_bn_layer(input,
                      ch_out,
                      filter_size,
                      stride,
                      padding,
                      act='relu',
                      bias_attr=False):
        tmp = fluid.layers.conv2d(
            input=input,
            filter_size=filter_size,
            num_filters=ch_out,
            stride=stride,
            padding=padding,
            act=None,
            bias_attr=bias_attr)
        return fluid.layers.batch_norm(input=tmp, act=act)

    def shortcut(input, ch_in, ch_out, stride):
        if ch_in != ch_out:
            return conv_bn_layer(input, ch_out, 1, stride, 0, None)
        else:
            return input

    def basicblock(input, ch_in, ch_out, stride):
        tmp = conv_bn_layer(input, ch_out, 3, stride, 1)
        tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, act=None, bias_attr=True)
        short = shortcut(input, ch_in, ch_out, stride)
        return fluid.layers.elementwise_add(x=tmp, y=short, act='relu')

    def layer_warp(block_func, input, ch_in, ch_out, count, stride):
        tmp = block_func(input, ch_in, ch_out, stride)
        for i in range(1, count):
            tmp = block_func(tmp, ch_out, ch_out, 1)
        return tmp

    assert (depth - 2) % 6 == 0
    n = (depth - 2) / 6
    conv1 = conv_bn_layer(
        input=input, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 16, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 32, 64, n, 2)
    pool = fluid.layers.pool2d(
        input=res3, pool_size=8, pool_type='avg', pool_stride=1)
    return pool


def vgg16_bn_drop(input):
    def conv_block(input, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=input,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max')

    conv1 = conv_block(input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)
    return fc2


def train(net_type, use_cuda, use_reader_op):
    classdim = 10
    data_shape = [3, 32, 32]
    BATCH_SIZE = 128
    print("Batch size is {}".format(BATCH_SIZE))

    train_file_path = "/tmp/cifar10_train.recordio"

    if use_reader_op:
        print("use reader op from {}".format(train_file_path))
        train_data_file = fluid.layers.open_recordio_file(
            filename=train_file_path,
            shapes=[[-1, 3, 32, 32], [-1, 1]],
            lod_levels=[0, 0],
            dtypes=["float32", "int64"],
            pass_num=50,
            for_parallel=False)
        train_data_file = fluid.layers.double_buffer(
            fluid.layers.batch(
                train_data_file, batch_size=BATCH_SIZE))
        images, label = fluid.layers.read_file(train_data_file)
    else:
        images = fluid.layers.data(
            name='pixel', shape=data_shape, dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    if net_type == "vgg":
        print("train vgg net")
        net = vgg16_bn_drop(images)
    else:
        raise ValueError("%s network is not supported" % net_type)

    predict = fluid.layers.fc(input=net, size=classdim, act='softmax')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    acc = fluid.layers.accuracy(input=predict, label=label)

    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_cost)

    # train_reader = paddle.batch(
    #     paddle.reader.shuffle(
    #         paddle.dataset.cifar.train10(), buf_size=BATCH_SIZE * 10),
    #     batch_size=BATCH_SIZE)

    train_reader = paddle.batch(
        paddle.dataset.cifar.train10(), batch_size=BATCH_SIZE)

    test_reader = paddle.batch(
        paddle.dataset.cifar.test10(), batch_size=BATCH_SIZE)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    if not use_reader_op:
        feeder = fluid.DataFeeder(place=place, feed_list=[images, label])

    exe.run(fluid.default_startup_program())

    PASS = 50
    iters = 1050
    skip_batch_num = 50

    print('start')

    iter_num, num_samples, start = 0, 0, time.time()
    for pass_id in range(PASS):
        if not use_reader_op:
            reader_generator = train_reader()
        data = None
        while True:
            if not use_reader_op:
                data = next(reader_generator, None)
                if data is None:
                    break
            if iter_num == iters:
                break
            if iter_num == skip_batch_num:
                start = time.time()
                num_samples = 0
            if use_reader_op:
                try:
                    exe.run(fluid.default_main_program(),
                            use_program_cache=True)
                except fluid.core.EnforceNotMet as ex:
                    break
            else:
                exe.run(fluid.default_main_program(), feed=feeder.feed(data))
            iter_num += 1
            if use_reader_op:
                num_samples += BATCH_SIZE
            else:
                num_samples += len(data)
            print("Pass: %d, Iter: %d" % (pass_id, iter_num))
        if iter_num == iters:
            break

    end = time.time()
    elapsed_time = end - start
    print('{} iteratios takes {} seconds wall clock time'.format(
        iters - skip_batch_num, elapsed_time))
    print('Total examples: %d; Throughput: %.5f examples per sec' %
          (num_samples, num_samples / elapsed_time))


if __name__ == '__main__':
    train('vgg', use_cuda=True, use_reader_op=True)
