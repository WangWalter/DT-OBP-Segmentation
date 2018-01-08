#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 22:41:50 2017

@author: ncuwkc
"""
import os
import shutil
source_dir = '/msl/home/ncuwkc/deeplab_v2/data/VOC2012/'
train = open(source_dir+'trainval.txt')
data = train.readlines()
src_dir = source_dir+'JPEGImages/'
target_dir = source_dir+'train_2012/'

for n, line in enumerate(data,1):
    shutil.copyfile(src_dir+line.rstrip()+'.jpg', target_dir+line.rstrip()+'.jpg')
