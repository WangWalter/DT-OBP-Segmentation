"# DT-OBP-Segmentation" 
Deeplab/voc12/config /{network}/ train.prototxt :caffe 訓練用，定義training layers若須改動dataset，修改data layer的{root_folder}及{sourse}，{root_folder}為dataset位置，{sourse}為路徑+txt file，此txt格式為(每行為影像名稱及對應label，參考Deeplab/voc12/list/train.txt)

Deeplab/voc12/config /{network}/test.prototxt : caffe 測試用，定義testing layers，若須改動測試dataset，修改data layer的{root_folder}及{sourse}，{root_folder}為dataset位置，{sourse}為路徑+txt file，此txt格式為(每行為影像名稱，參考Deeplab/voc12/list/test.txt)

Deeplab/voc12/config /{network}/solver.prototxt :設定基本參數
Deeplab/voc12/config /{network}/deploy.prototxt : matlab、python測試用，定義testing layers
Deeplab/voc12/model /{network}/: model儲存位置，位置及screenshot由solver.prototxt設定

Deeplab/run_pascal.sh : 設定{NUM_LABELS}及caffe command 設定

Deeplab/voc12/python: copy_net.py(複製pretraind model 的參數至新定義架構)、run_segtest.py(跑測試網路，網路定義在deploy.prototxt)

Training 
設定run_pascal.sh裡的參數 (RUN_TRAIN:設為1、MODEL:init model，不設定的話即隨機初始化參數)
$ sh run_pascal.sh

Testing 
設定run_pascal.sh裡的參數 (RUN_TRAIN:設為1、MODEL:init model，不設定的話即隨機初始化參數)
$ sh run_pascal.sh

Testing (python)
run_segtest.py

