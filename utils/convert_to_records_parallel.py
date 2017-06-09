#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang     Xiaomi

"""Converts data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import struct
import sys
import multiprocessing

import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(sys.path[0]))
from io_funcs.tfrecords_io import make_sequence_example

tf.logging.set_verbosity(tf.logging.INFO)

def convert_cmvn_to_numpy(inputs_cmvn, labels_cmvn):
    """Convert global binary ark cmvn to numpy format."""
    tf.logging.info("Convert %s and %s to numpy format" % (
        inputs_cmvn, labels_cmvn))
    inputs_filename = os.path.join(FLAGS.data_dir, inputs_cmvn + '.cmvn')
    labels_filename = os.path.join(FLAGS.data_dir, labels_cmvn + '.cmvn')

    inputs = read_binary_file(inputs_filename, 0)
    labels = read_binary_file(labels_filename, 0)

    inputs_frame = inputs[0][-1]
    labels_frame = labels[0][-1]

    assert inputs_frame == labels_frame

    cmvn_inputs = np.hsplit(inputs, [inputs.shape[1]-1])[0]
    cmvn_labels = np.hsplit(labels, [labels.shape[1]-1])[0]

    mean_inputs = cmvn_inputs[0] / inputs_frame
    stddev_inputs = np.sqrt(cmvn_inputs[1] / inputs_frame - mean_inputs ** 2)
    mean_labels = cmvn_labels[0] / labels_frame
    stddev_labels = np.sqrt(cmvn_labels[1] / labels_frame - mean_labels ** 2)

    cmvn_name = os.path.join(FLAGS.output_dir, "train_cmvn.npz")
    np.savez(cmvn_name,
             mean_inputs=mean_inputs,
             stddev_inputs=stddev_inputs,
             mean_labels=mean_labels,
             stddev_labels=stddev_labels)

    tf.logging.info("Write to %s" % cmvn_name)

def read_binary_file(filename, offset=0):
    """Read data from matlab binary file (row, col and matrix).

    Returns:
        A numpy matrix containing data of the given binary file.
    """
    read_buffer = open(filename, 'rb')
    read_buffer.seek(int(offset), 0)
    header = struct.unpack('<xcccc', read_buffer.read(5))
    if header[0] != 'B':
        print("Input .ark file is not binary")
        sys.exit(-1)
    if header[1] == 'C':
        print("Input .ark file is compressed, exist now.")
        sys.exit(-1)

    rows = 0; cols= 0
    _, rows = struct.unpack('<bi', read_buffer.read(5))
    _, cols = struct.unpack('<bi', read_buffer.read(5))

    if header[1] == "F":
        tmp_mat = np.frombuffer(read_buffer.read(rows * cols * 4),
                                dtype=np.float32)
    elif header[1] == "D":
        tmp_mat = np.frombuffer(read_buffer.read(rows * cols * 8),
                                dtype=np.float64)
    mat = np.reshape(tmp_mat, (rows, cols))

    read_buffer.close()

    return mat

def process_in_each_thread(line, name, apply_cmvn):
    if name != 'test':
        utt_id, inputs_path, labels_path = line.strip().split()
        inputs_path, inputs_offset = inputs_path.split(':')
        labels_path, labels_offset = labels_path.split(':')
    else:
        utt_id, inputs_path = line.strip().split()
        inputs_path, inputs_offset = inputs_path.split(':')
    tfrecords_name = os.path.join(FLAGS.output_dir, name,
                                  utt_id + ".tfrecords")
    with tf.python_io.TFRecordWriter(tfrecords_name) as writer:
        tf.logging.info(
            "Writing utterance %s to %s" % (utt_id, tfrecords_name))
        inputs = read_binary_file(inputs_path, inputs_offset).astype(np.float64)
        if name != 'test':
            labels = read_binary_file(labels_path, labels_offset).astype(np.float64)
        else:
            labels = None
        if apply_cmvn:
            cmvn = np.load(os.path.join(FLAGS.output_dir, "train_cmvn.npz"))
            inputs = (inputs - cmvn["mean_inputs"]) / cmvn["stddev_inputs"]
            if labels is not None:
                labels = (labels - cmvn["mean_labels"]) / cmvn["stddev_labels"]
        ex = make_sequence_example(inputs, labels)
        writer.write(ex.SerializeToString())

def convert_to(name, apply_cmvn=True):
    """Converts a dataset to tfrecords."""
    config_file = open(os.path.join(FLAGS.config_dir, name + ".lst"))
    if not os.path.exists(os.path.join(FLAGS.output_dir, name)):
        os.makedirs(os.path.join(FLAGS.output_dir, name))

    pool = multiprocessing.Pool(FLAGS.num_threads)
    workers= []
    for line in config_file:
        workers.append(pool.apply_async(
            process_in_each_thread, (line, name, apply_cmvn)))
    pool.close()
    pool.join()

    config_file.close()


def main(unused_argv):
    # Convert to Examples and write the result to TFRecords.
    convert_cmvn_to_numpy('inputs', 'labels')

    #convert_to("myTest", apply_cmvn=True)
    convert_to("train", apply_cmvn=True)
    convert_to("val", apply_cmvn=True)
    convert_to("test", apply_cmvn=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/raw/cmvn',
        help='Directory to load global cmvn.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/tfrecords',
        help='Directory to write the converted result'
    )
    parser.add_argument(
        '--config_dir',
        type=str,
        default='config',
        help='Directory to load train, val and test lists'
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        default='10',
        help='The number of threads to convert tfrecords files.'
    )
    FLAGS, unparsed = parser.parse_known_args()

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
