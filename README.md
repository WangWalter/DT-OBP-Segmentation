"# DT-OBP-Segmentation" 
code解說
run_pascal.sh : 設定{NUM_LABELS}及caffe command 設定
python_code/train.prototxt :
caffe 訓練用，定義training layers若須改動dataset，修改data layer的{root_folder}及{sourse}，{root_folder}為dataset位置，{sourse}為路徑+txt file，此txt格式為(每行為影像名稱及對應label，參考Deeplab/voc12/list/train.txt)
python_code/test.prototxt : 
caffe 測試用，定義testing layers，若須改動測試dataset，修改data layer的{root_folder}及{sourse}，{root_folder}為dataset位置，{sourse}為路徑+txt file，此txt格式為(每行為影像名稱，參考Deeplab/voc12/list/test.txt)

python_code/solver.prototxt :設定網路基本參數
python_code/deploy.prototxt : matlab、python測試用，定義testing layers

python_code/copy_net.py : 複製pretraind model 的參數至新定義架構
python_code/run_segtest.py : 跑測試網路，網路定義在deploy.prototxt

Training 
設定run_pascal.sh裡的參數 (RUN_TRAIN:設為1、MODEL:init model，不設定的話即隨機初始化參數)
 $ sh run_pascal.sh

Testing 
設定run_pascal.sh裡的參數 (RUN_TRAIN:設為1、MODEL:init model，不設定的話即隨機初始化參數)
 $ sh run_pascal.sh

Testing (python)
run_segtest.py

