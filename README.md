# DT-OBP-Segmentation
##Installation
###Requirements
>• Python 2.7
>• Linux
>• Cuda 8.0
>• Cudnn 5 (optional)
>• Caffe (follow here)
>• anaconda (recommend)
>• matio 5(To use the mat_read_layer and mat_write_layer, please download here)
###caffe (brief compile description)
>1. $mkdir build

>2. $cd build

>3. $cmake -DUSE_CUDNN=ON -DCMAKE_INSTALL_PREFIX={install path} -DBLAS=open -DBUILD_matlab=ON ..

>4. $make –j8

>5. $make pycaffe 

>6. $make install


###code 
>run_pascal.sh : setting {NUM_LABELS} and caffe command 
>python_code/train.prototxt :
for caffe training, defined training layer. if you need testing on your own dataset，modify data layer {root_folder} and {sourse}, where {root_folder} is dataset path, {sourse} is txt file full path, the txt format is image name correspond a label every row.
>python_code/test.prototxt : 
for caffe testing, defined testing layers，if you need testing on your own dataset，modify data layer {root_folder} and {sourse}, where {root_folder} is dataset path，{sourse} is txt file full path, the txt format is image name every row

>python_code/solver.prototxt : initial network setting

>python_code/deploy.prototxt : for matlab、python testing, the layers is same with testing layers

>python_code/copy_net.py : copy parameters of pretraind model to new defined architecture
>python_code/run_segtest.py : run testing for python 

###Caffe Training 
>setting run_pascal.sh (RUN_TRAIN = 1、MODEL = {init model}，if not setting it will random initial parameters of netwrok)
>$ sh run_pascal.sh

###Caffe Testing 
>setting run_pascal.sh (RUN_TRAIN = 1、MODEL = {testing model})
>$ sh run_pascal.sh

###Testing (python)
>$ python run_segtest.py

