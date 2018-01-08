#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 20:20:02 2017

@author: ncuwkc
"""
import numpy as np
import PIL as Image
import os
from os import listdir, makedirs
from os.path import isfile, join, exists
import scipy.io as sio
def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)
 
 
def color_map(N=256, normalized=False):
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3
 
        cmap[i] = np.array([r, g, b])
 
    cmap = cmap/255 if normalized else cmap
    return cmap

def toRGBarray(ar): 
    cmap=color_map(N=256, normalized=False)
    rows=ar.shape[0] 
    cols=ar.shape[1] 
    r=np.zeros(ar.size*3, dtype=np.uint8).reshape(rows,cols,3) 
    for i in range(rows): 
        for j in range(cols): 
            r[i,j]=cmap[ar[i,j]] 

    return r


if __name__ == '__main__':
    root_dir = '/msl/home/ncuwkc/deeplab_v2/voc12/features2/ResNet_FCN_OBG/test/'
    src_dir = root_dir + 'fc2/'
    dst_dir = root_dir + 'ResNet_FCN_OBG5000/'
    pic_dir = '/msl/home/ncuwkc/deeplab_v2/data/test2012/JPEGImages/'
    files = [f for f in listdir(src_dir) if isfile(join(src_dir,f))]
    colormap = color_map()
    if not os.path.exists(dst_dir):
        makedirs(dst_dir)
    
    for i in files:
        mat_file = sio.loadmat(src_dir+i)
        mat_file = mat_file['data']
        im = Image.Image.open(pic_dir+i[0:11]+'.jpg')
        (w,h) = im.size
        label = mat_file.argmax(axis = 2)
        rgb = toRGBarray(label[:,:,0])
        rgb = rgb.transpose((1, 0, 2))
        rgb = Image.Image.fromarray(rgb)
        
        rgb.crop((0,0,w,h)).save(dst_dir + i[0:11]+'.png')
        
        