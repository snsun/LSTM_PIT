#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang     Xiaomi

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time

import numpy as np
import tensorflow as tf

from tfrecords_io import get_padded_batch

tf.logging.set_verbosity(tf.logging.INFO)


class TfrecordsIoTest(tf.test.TestCase):

    def testReadTfrecords(self):
        name = 'train'
        config_file = open(os.path.join(FLAGS.config_dir, name +  ".lst"))
        tfrecords_lst = []
        for line in config_file:
            utt_id, inputs_path, labels_path = line.strip().split()
            tfrecords_name = os.path.join(FLAGS.data_dir, name,
                                          utt_id + ".tfrecords")
            tfrecords_lst.append(tfrecords_name)

        with tf.Graph().as_default():
            inputs, labels, lengths = get_padded_batch(
                tfrecords_lst, FLAGS.batch_size, FLAGS.input_dim,
                FLAGS.output_dim, num_enqueuing_threads=FLAGS.num_threads,
                num_epochs=FLAGS.num_epochs)

            init = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())

            sess = tf.Session()

            sess.run(init)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                time_start = time.time()
                while not coord.should_stop():
                    # Print an overview fairly often.
                    tr_inputs, tr_labels, tr_lengths = sess.run([
                        inputs, labels, lengths])
                    tf.logging.info('inputs shape : '+ str(tr_inputs.shape))
                    tf.logging.info('labels shape : ' + str(tr_labels.shape))
                    tf.logging.info('actual lengths : ' + str(tr_lengths))
            except tf.errors.OutOfRangeError:
                tf.logging.info('Done training -- epoch limit reached')
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)
            sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Mini-batch size.'
    )
    parser.add_argument(
        '--input_dim',
        type=int,
        default=145,
        help='The dimension of inputs.'
    )
    parser.add_argument(
        '--output_dim',
        type=int,
        default=51,
        help='The dimension of outputs.'
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        default=8,
        help='The num of threads to read tfrecords files.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=1,
        help='The num of epochs to read tfrecords files.'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/tfrecords/',
        help='Directory of train, val and test data.'
    )
    parser.add_argument(
        '--config_dir',
        type=str,
        default='config/',
        help='Directory to load train, val and test lists.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.test.main()
