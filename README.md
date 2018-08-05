# DT-OBP-Segmentation 

## Introduction
proposed architecture improves the DT-EdgeNet (Domain Transform with EdgeNet)[1]. Here, we combined the OBG-FCN [2] mask 
network and replaced the [1] edge network. The used mask network can predict background, object, and object edge reference diagrams. In addition, our architecture uses multi-scale ResNet-101 as the base network and introduces multi-scale Atrous Convolution to architecture training to preserve the dimensions of the feature map, which increases the receptive and further to enhance the accuracy of semantic segmentation. 

## Installation
### Requirements

  - Python 2.7 
  - Linux
  - Cuda 8.0
  - Cudnn 5 (optional)
  - Caffe (follow [here](https://github.com/xmyqsh/deeplab-v2))
  - anaconda (recommend)
  - matio (To use the mat_read_layer and mat_write_layer, please download [here](https://sourceforge.net/projects/matio/files/matio/1.5.2/))

### caffe (brief compile description)
```sh
1. $mkdir build

2. $cd build

3. $cmake -DUSE_CUDNN=ON -DCMAKE_INSTALL_PREFIX={install path} -DBLAS=open -DBUILD_matlab=ON ..

4. $make –j8

5. $make pycaffe

6. $make install
```

### model
trained model can be download:

[Deeplabev2](https://drive.google.com/open?id=1BNBlWfQ9dtiJMdD360y5GGId_Eag7w5o) reference(https://bitbucket.org/aquariusjay/deeplab-public-ver2)

[Proposed model](https://drive.google.com/open?id=1IjNv1qADPg40ZoqvkTOWeDChHhho3AKR)

### code 
>run_pascal.sh : setting {NUM_LABELS} and caffe command 

>python_code/train.prototxt : for caffe training, defined training layer. if you need testing on your own dataset，modify data layer {root_folder} and {sourse}, where {root_folder} is dataset path, {sourse} is txt file full path, the txt format is image name correspond a label every row.

>python_code/test.prototxt : 
for caffe testing, defined testing layers，if you need testing on your own dataset，modify data layer {root_folder} and {sourse}, where {root_folder} is dataset path，{sourse} is txt file full path, the txt format is image name every row

>python_code/solver.prototxt : initial network setting

>python_code/deploy.prototxt : for matlab、python testing, the layers is same with testing layers

>python_code/copy_net.py : copy parameters of pretraind model to new defined architecture

>python_code/run_segtest.py : run testing for python 

### Caffe Training 
setting run_pascal.sh (RUN_TRAIN = 1、MODEL = {init model}，if not setting it will random initial parameters of netwrok)
```sh
$ sh run_pascal.sh
```
### Caffe Testing 
setting run_pascal.sh (RUN_TRAIN = 1、MODEL = {testing model})
```sh
$ sh run_pascal.sh
```
### Testing (python)
```sh
$ python run_segtest.py
```

## Reference
[1] L. Chen, J. Barron, G. Papandreou, K. Murphy, and A. Yuille, “Semantic image segmentation with task-specific edge detection using CNNs and a discriminatively trained domain transform,” arXiv preprint arXiv:1511.03328, 2015. 

[2] Q. Huang, C. Xia, W. Zheng, Y. Song, H. Xu, and C. C. J. Kuo, “Object Boundary Guided Semantic Segmentation” arXiv preprint arXiv:1603.09742, 2016. 




