#!/bin/sh

## MODIFY PATH for YOUR SETTING
export LD_PRELOAD=/msl/home/ncuwkc/deeplab-v2-cudnnv5/matio/lib/libmatio.so.4
export LD_LIBRARY_PATH=/home/ncuwkc/anaconda2/lib:${LD_LIBRARY_PATH}
ROOT_DIR=/msl/home/ncuwkc/deeplab_v2/data/5class/testimg/

CAFFE_BIN=/home/ncuwkc/anaconda2/bin/caffe 


EXP=voc12

if [ "${EXP}" = "voc12" ]; then
    NUM_LABELS=6				#total of class (including background)
    NUM_LABELS_OBG=3    #3
    DATA_ROOT=${ROOT_DIR}
    #/rmt/data/pascal/VOCdevkit/VOC2012
else
    NUM_LABELS=0
    echo "Wrong exp name"
fi
 

## Specify which model to train
########### voc12 ################
NET_ID=single_ResNet
#NET_ID=ResNet_FCN_OBG
## Variables used for weakly or semi-supervisedly training
TRAIN_SET_SUFFIX= #

TRAIN_SET_STRONG=train

DEV_ID=2  #GPU device

#####

## Create dirs

CONFIG_DIR=${EXP}/config/${NET_ID}
MODEL_DIR=${EXP}/model/${NET_ID}
mkdir -p ${MODEL_DIR}
LOG_DIR=${EXP}/log/${NET_ID}
mkdir -p ${LOG_DIR}
export GLOG_log_dir=${LOG_DIR}

## Run

RUN_TRAIN=0 #if training set to 1 and it will run train.prototxt
RUN_TEST=1  #if testing set to 1 and it will run test.prototxt

## Training #1 (on train_aug)

if [ ${RUN_TRAIN} -eq 1 ]; then
    #
    LIST_DIR=${EXP}/list
    TRAIN_SET=train${TRAIN_SET_SUFFIX}
    TRAIN_SET_OBG=train_obg
    if [ -z ${TRAIN_SET_WEAK_LEN} ]; then
				TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}
				comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
    else
				TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}_head${TRAIN_SET_WEAK_LEN}
				comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt | head -n ${TRAIN_SET_WEAK_LEN} > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
    fi
    #
    MODEL=${EXP}/model/${NET_ID}/train_iter_40000.caffemodel#train model here!!!!!!!!!!!!!!
    #
    echo Training net ${EXP}/${NET_ID}
    for pname in train solver; do
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/${pname}.prototxt > ${CONFIG_DIR}/${pname}_${TRAIN_SET}.prototxt
    done
        CMD="${CAFFE_BIN} train \
         --solver=${CONFIG_DIR}/solver_${TRAIN_SET}.prototxt \
		 --gpu=${DEV_ID} "
	     #--snapshot=${EXP}/model/${NET_ID}/5class/train_iter_21854.solverstate"
		if [ -f ${MODEL} ]; then
				CMD="${CMD} --weights=${MODEL}"
		fi
		echo Running ${CMD} && ${CMD}
fi

## Test #1 specification (on val or test)



## Test #1 on official test set

if [ ${RUN_TEST} -eq 1 ]; then
    #
    for TEST_SET in test; do
				TEST_ITER=`cat /msl/home/ncuwkc/deeplab_v2/data/5class/test.txt | wc -l`
				MODEL=/msl/home/ncuwkc/deeplab_v2/voc12/model/${NET_ID}/5class/train_iter_63000.caffemodel
				#MODEL=/msl/home/ncuwkc/deeplab_v2/voc12/model/FCN-8s/train_iter_12000.caffemodel
				if [ ! -f ${MODEL} ]; then
						MODEL=`ls -t ${EXP}/model/${NET_ID}/train_iter_*.caffemodel | head -n 1`
				fi
				#
				echo Testing2 net ${EXP}/${NET_ID}
				FEATURE_DIR=${EXP}/features2/5class/${NET_ID}
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/fcn
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/crf
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/test.prototxt > ${CONFIG_DIR}/test_${TEST_SET}.prototxt
				CMD="${CAFFE_BIN} test \
             --model=${CONFIG_DIR}/test_${TEST_SET}.prototxt \
             --weights=${MODEL} \
             --iterations=${TEST_ITER} \
			 --gpu=${DEV_ID} "
				echo Running ${CMD} && ${CMD}
    done
fi