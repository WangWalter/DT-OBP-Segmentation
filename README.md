"# DT-OBP-Segmentation" 
Deeplab/voc12/config /{network}/ train.prototxt :caffe �V�m�ΡA�w�qtraining layers�Y�����dataset�A�ק�data layer��{root_folder}��{sourse}�A{root_folder}��dataset��m�A{sourse}�����|+txt file�A��txt�榡��(�C�欰�v���W�٤ι���label�A�Ѧ�Deeplab/voc12/list/train.txt)

Deeplab/voc12/config /{network}/test.prototxt : caffe ���եΡA�w�qtesting layers�A�Y����ʴ���dataset�A�ק�data layer��{root_folder}��{sourse}�A{root_folder}��dataset��m�A{sourse}�����|+txt file�A��txt�榡��(�C�欰�v���W�١A�Ѧ�Deeplab/voc12/list/test.txt)

Deeplab/voc12/config /{network}/solver.prototxt :�]�w�򥻰Ѽ�
Deeplab/voc12/config /{network}/deploy.prototxt : matlab�Bpython���եΡA�w�qtesting layers
Deeplab/voc12/model /{network}/: model�x�s��m�A��m��screenshot��solver.prototxt�]�w

Deeplab/run_pascal.sh : �]�w{NUM_LABELS}��caffe command �]�w

Deeplab/voc12/python: copy_net.py(�ƻspretraind model ���ѼƦܷs�w�q�[�c)�Brun_segtest.py(�]���պ����A�����w�q�bdeploy.prototxt)

Training 
�]�wrun_pascal.sh�̪��Ѽ� (RUN_TRAIN:�]��1�BMODEL:init model�A���]�w���ܧY�H����l�ưѼ�)
$ sh run_pascal.sh

Testing 
�]�wrun_pascal.sh�̪��Ѽ� (RUN_TRAIN:�]��1�BMODEL:init model�A���]�w���ܧY�H����l�ưѼ�)
$ sh run_pascal.sh

Testing (python)
run_segtest.py

