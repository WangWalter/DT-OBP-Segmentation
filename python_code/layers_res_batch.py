import caffe

import numpy as np
from PIL import Image

import random

class VOCSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - voc_dir: path to PASCAL VOC year dir
        - split: train / val / test
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for PASCAL VOC semantic segmentation.

        example

        params = dict(voc_dir="/path/to/PASCAL/VOC2011",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="val")
        """
        # config
        params = eval(self.param_str)
        self.voc_dir = params['voc_dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.crop_size = params.get('crop_size', None)
        self.scale_factors = params.get('scale_factors', None)

        # two tops: data and label
        if len(top) != 4:
            raise Exception("Need to define three tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/ImageSets/Segmentation/{}.txt'.format(self.voc_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0
        self.scale = 1
        # make eval deterministic
        if 'train1' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)
        if self.scale_factors:
            random.seed(self.seed)
            self.scale = random.randint(0, len(self.scale_factors)-1)

    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])
        self.label2 = self.load_label2(self.indices[self.idx])
        self.data_dim = [[[321,321]]]
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, 3,self.crop_size,self.crop_size)
        top[1].reshape(1, 1,self.crop_size,self.crop_size)
        top[2].reshape(1, 1,self.crop_size,self.crop_size)
        top[3].reshape(1, 1, 1, 2)


    def forward(self, bottom, top):
        # assign output
	# random crop and input to blob
        im_size = self.data.shape
        w_off = random.randint(0, im_size[2]-self.crop_size)
        h_off = random.randint(0, im_size[1]-self.crop_size)
        top[0].data[...] = self.data[:,h_off:(h_off+self.crop_size),w_off:(w_off+self.crop_size)]
        top[1].data[...] = self.label[:,h_off:(h_off+self.crop_size),w_off:(w_off+self.crop_size)]
        top[2].data[...] = self.label2[:,h_off:(h_off+self.crop_size),w_off:(w_off+self.crop_size)]
        top[3].data[...] = self.data_dim
        

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0
        if self.scale_factors:
            self.scale = random.randint(0, len(self.scale_factors)-1)

    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/JPEGImages/{}.jpg'.format(self.voc_dir, idx))
        #im.save('/msl/home/ncuwkc/deeplab_v2/data/VOC2012/ImageSets/Segmentation/1.jpg');
        width, height = im.size
        if self.scale_factors[self.scale]!=1:
            width = width * self.scale_factors[self.scale]
            height = height * self.scale_factors[self.scale]
            im = im.resize( (int(width), int(height)), Image.BILINEAR )
			
		
        in_temp = np.array(im, dtype=np.float32)
        """padding"""
        if self.crop_size > width and self.crop_size > height :
            temp = np.ones((self.crop_size, self.crop_size, 3))
            temp[:height,:width,:] = in_temp
            in_temp = temp
        elif self.crop_size > width :
            temp = np.ones((height, self.crop_size, 3))
            temp[:height,:width,:] = in_temp
            in_temp = temp
        elif self.crop_size > height :
            temp = np.ones((self.crop_size, width, 3))
            temp[:height,:width,:] = in_temp
            in_temp = temp
	#temp1 = (255.0 / in_temp.max() * (in_temp - in_temp.min())).astype(np.uint8)
        in_ = in_temp[:,:,::-1]
	#img = Image.fromarray(in_temp,'RGB')
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
	
	#img.save('/msl/home/ncuwkc/deeplab_v2/data/VOC2012/ImageSets/Segmentation/1.png');
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open('{}/SegmentClass/{}.png'.format(self.voc_dir, idx))
        width, height = im.size
        if self.scale_factors[self.scale]!=1:
            width = width * self.scale_factors[self.scale]
            height = height * self.scale_factors[self.scale]
            im = im.resize( (int(width), int(height)), Image.BILINEAR )
        in_temp = np.array(im, dtype=np.float32)
        """padding"""
        print ('{} {} {}/n',int(width), int(height), idx)
        if self.crop_size > width and self.crop_size > height :
            temp = 255*np.ones((self.crop_size, self.crop_size))
            temp[:height,:width] = in_temp
            in_temp = temp
        elif self.crop_size > width :
            temp = 255*np.ones((height, self.crop_size))
            temp[:height,:width] = in_temp
            in_temp = temp
        elif self.crop_size > height : 
            temp = 255*np.ones(( self.crop_size, width))
            temp[:height,:width] = in_temp
            in_temp = temp

        label = in_temp[np.newaxis, ...]
	#I8 = (((in_temp - in_temp.min()) / (in_temp.max() - in_temp.min())) * 255.9).astype(np.uint8)
	#img = Image.fromarray(I8)
	#img.save('/msl/home/ncuwkc/deeplab_v2/data/VOC2012/ImageSets/Segmentation/2.png');
        return label

    def load_label2(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open('{}/SegmentOBG_v2/{}.png'.format(self.voc_dir, idx))
	#im.save('/msl/home/ncuwkc/deeplab_v2/data/VOC2012/ImageSets/Segmentation/3.jpg');
        width, height = im.size
        if self.scale_factors[self.scale]!=1:
            width = width * self.scale_factors[self.scale]
            height = height * self.scale_factors[self.scale]
            im = im.resize( (int(width), int(height)), Image.BILINEAR )
            in_temp = np.array(im, dtype=np.float32)
        """padding"""
        if self.crop_size > width and self.crop_size > height :
            temp = 255*np.ones((self.crop_size, self.crop_size))
            temp[:height,:width] = in_temp
            in_temp = temp
        elif self.crop_size > width :
            temp = 255*np.ones((height, self.crop_size))
            temp[:height,:width] = in_temp
            in_temp = temp
        elif self.crop_size > height :
            temp = 255*np.ones((self.crop_size, width))
            temp[:height,:width] = in_temp
            in_temp = temp
        label2 = in_temp[np.newaxis, ...]
	#I8 = (((in_temp - in_temp.min()) / (in_temp.max() - in_temp.min())) * 255.9).astype(np.uint8)
	#img = Image.fromarray(I8)
	#img.save('/msl/home/ncuwkc/deeplab_v2/data/VOC2012/ImageSets/Segmentation/3.png');
        return label2



class SBDDSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from the SBDD extended labeling
    of PASCAL VOC for semantic segmentation
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - sbdd_dir: path to SBDD `dataset` dir
        - split: train / seg11valid
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for SBDD semantic segmentation.

        N.B.segv11alid is the set of segval11 that does not intersect with SBDD.
        Find it here: https://gist.github.com/shelhamer/edb330760338892d511e.

        example

        params = dict(sbdd_dir="/path/to/SBDD/dataset",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="valid")
        """
        # config
        params = eval(self.param_str)
        self.sbdd_dir = params['sbdd_dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/{}.txt'.format(self.sbdd_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/img/{}.jpg'.format(self.sbdd_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        import scipy.io
        mat = scipy.io.loadmat('{}/cls/{}.mat'.format(self.sbdd_dir, idx))
        label = mat['GTcls'][0]['Segmentation'][0].astype(np.uint8)
        label = label[np.newaxis, ...]
        return label
