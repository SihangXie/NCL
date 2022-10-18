#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CONFIG=$1
GPUS=$2
#GPUS='0,1,2,3'

CUDA_VISIBLE_DEVICES=$GPUS $PYTHON main/multi_network_train_final.py --cfg $CONFIG ${@:3}
