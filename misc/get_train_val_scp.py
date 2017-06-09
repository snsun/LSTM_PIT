#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang     Xiaomi

from __future__ import absolute_import
from __future__ import print_function

import random
import sys
import os

raw = 'data/raw'
val_size = 5000

input_scp_dir = raw + '/prepared_input/'
label_scp_dir = raw + '/prepared_label/'
test_scp_dir = raw + '/prepared_test/'
lst_dir = 'config/'

if not os.path.exists(lst_dir):
    os.makedirs(lst_dir)

input_scp = open(input_scp_dir + 'all.scp')
label_scp = open(label_scp_dir + 'all.scp')
test_scp = open(test_scp_dir + 'all.scp')

input_train = open(input_scp_dir + 'train.scp','w')
input_val = open(input_scp_dir + 'val.scp','w')
input_test = open(test_scp_dir + 'test.scp','w')
label_train = open(label_scp_dir + 'train.scp','w')
label_val = open(label_scp_dir + 'val.scp','w')

lst_train = open(lst_dir + 'train.lst','w')
lst_val = open(lst_dir + 'val.lst','w')
lst_test = open(lst_dir + 'test.lst','w')

lists_input = input_scp.readlines()
lists_label = label_scp.readlines()
lists_test = test_scp.readlines()

if len(lists_input) != len(lists_label):
    print("scp files %s and %s have unequal lengths" % (input_scp, label_scp))
    sys.exit(1)

# Train and validation sets.
lists = range(len(lists_input))
random.shuffle(lists)
for i in xrange(len(lists)):
    line_input = lists_input[i]
    line_label = lists_label[i]
    line_lst = line_input.strip() + ' ' + line_label.split()[1] + '\n'
    if i < val_size:
        input_val.write(line_input)
        label_val.write(line_label)
        lst_val.write(line_lst)
    else:
        input_train.write(line_input)
        label_train.write(line_label)
        lst_train.write(line_lst)

# Test sets.
for i in xrange(len(lists_test)):
    line_input = lists_test[i]
    input_test.write(line_input)
    line_lst = line_input.strip() + '\n'
    lst_test.write(line_lst)
