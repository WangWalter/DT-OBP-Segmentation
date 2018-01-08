#!/usr/bin/env python
"""
detector.py is an out-of-the-box windowed detector
callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
Note that this model was trained for image classification and not detection,
and finetuning for detection can be expected to improve results.

The selective_search_ijcv_with_python code required for the selective search
proposal mode is available at
    https://github.com/sergeyk/selective_search_ijcv_with_python

TODO:
- batch up image filenames as well: don't want to load all of them into memory
- come up with a batching scheme that preserved order / keeps a unique ID
"""
import caffe
import numpy as np
import PIL as Image
import matplotlib.pyplot as plt
import os
import os.path
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
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    #
#    print(data)
    img = Image.Image.fromarray(np.uint8(data*255))
    img.save('/msl/home/ncuwkc/deeplab_v2/data/5class/test_sam_result/'+im_name+'_data.png')
    
    plt.imshow(data)
if __name__ == '__main__':
    GPU_ID = 2
    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)
#    print("CPU mode")
    model_name = 'single_ResNet/';
    #model_name = 'vgg-endtoend-withOBG/'
    proto_dir = '/msl/home/ncuwkc/deeplab_v2/voc12/config/'+ model_name;
    model_dir = '/msl/home/ncuwkc/deeplab_v2/voc12/model/' +model_name;
    net_model = proto_dir +'deploy.prototxt';
    net_weights = model_dir+'5class/train_iter_25000.caffemodel';
    net = caffe.Net(net_model, net_weights, caffe.TEST)
    #load image 
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.interpolation'] = 'nearest'
    input_size = 513
    image_dir = '/msl/home/ncuwkc/deeplab_v2/data/5class/test_sam/'
#    for i,im in enumerate(image):
    for im_name in os.listdir(image_dir):
        im = Image.Image.open(image_dir + im_name)
        width, height = im.size 
        """input size is 513, if size>513 need resize,if size<513 need padding""" 
        if  width > input_size and width>height:
            ratio = float(input_size)/width
            im = im.resize([int(width*ratio), int(height*ratio)], Image.Image.BILINEAR)
        elif height > input_size :
            ratio = float(input_size)/height
            im = im.resize([int(width*ratio), int(height*ratio)], Image.Image.BILINEAR)
        newwidth, newheight = im.size
        in_temp = np.array(im, dtype=np.float32)
        temp = np.ones((input_size, input_size, 3))
        temp[:newheight,:newwidth,:] = in_temp
        in_temp = temp
        #in_ = np.array(im, dtype=np.float32)
        """transform im_array to chx"""
        in_ = in_temp[:,:,::-1]
        in_ -= np.array((104.00698793, 116.66876762, 122.67891434))
        in_ = in_.transpose((2, 0, 1))
        data_dim = [[[[input_size,input_size]]]]
    #test        
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_
  #      net.blobs['data_dim'].reshape(1,1,1,2)
  #      net.blobs['data_dim'].data[...] = data_dim
        res = net.forward()
        out = net.blobs['fc1_interp'].data
        vis_square(out[0], padval=1)
        c = out.argmax(axis=1)
        out1 = toRGBarray(c[0,:,:])
        img = Image.Image.fromarray(out1[:newheight,:newwidth,:])
        img.save('/msl/home/ncuwkc/deeplab_v2/data/5class/test_sam_result/'+im_name+'.png')

