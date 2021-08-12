#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 13:47:25 2021

@author: root
"""

import os
import numpy as np

num_data_each_class = 50
training_data_dir = 'data/modelnet40_normal_resampled/'
SHAPE_NAMES = [line.rstrip() for line in \
    open('data/shape_names.txt')] 
data_files = os.listdir(training_data_dir)

with open("Adv_suc_test_list.txt","w") as t:
    for f in data_files:
        if f in SHAPE_NAMES:
            cur_dir = training_data_dir + f
            num_cur_files = len(os.listdir(cur_dir))
            selected_data = np.random.choice(num_cur_files,num_data_each_class)
            for i in range(num_data_each_class):
                cur_file_name = os.path.join(training_data_dir+f,os.listdir(cur_dir)[selected_data[i]])
                t.write(cur_file_name+'\n')
            