#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 20:53:09 2017

@author: ncuwkc
"""

import caffe
import numpy as np
import PIL as Image

if __name__ == '__main__':
    caffe.set_mode_cpu()
    print("CPU mode")
    model_name = 'ResNet_FCN_OBG/';
    #model_name = 'vgg-endtoend-withOBG/'
    proto_dir = '/msl/home/ncuwkc/deeplab_v2/voc12/config/'+ model_name;
    model_dir = '/msl/home/ncuwkc/deeplab_v2/voc12/model/' +model_name;
    net_model = proto_dir +'deploy_slice.prototxt';
    net_weights = model_dir +'init.caffemodel';
    #load image 
    model_name_1 = 'single_ResNet/';
    #model_name = 'vgg-endtoend-withOBG/'
    proto_dir_1 = '/msl/home/ncuwkc/deeplab_v2/voc12/config/'+ model_name_1;
    model_dir_1 = '/msl/home/ncuwkc/deeplab_v2/voc12/model/' +model_name_1;
    net_model_1 = proto_dir_1 +'deploy.prototxt';
    net_weights_1 = model_dir_1 +'train_iter_3.caffemodel';
    net = caffe.Net(net_model, net_weights, caffe.TEST)
    net_1 = caffe.Net(net_model_1, net_weights_1, caffe.TEST)
    net_key = [key for key in net.params]
    net_key1 = [key for key in net_1.params]
    intersect = set(net_key).intersection(net_key1)
    for element in intersect:
        if net_1.params[element]:
            for index in range(len(net.params[element])) :
                if net_1.params[element][index].data.shape == net.params[element][index].data.shape:
                    net_1.params[element][index].data[...] = net.params[element][index].data[...]
    net_1.save(model_dir_1+'init1.caffemodel');