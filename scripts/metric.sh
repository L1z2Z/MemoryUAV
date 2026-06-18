#!/bin/bash

# ROOT_DIR='/path/to/your/root/eval/result/dir' # ROOT_DIR="./closeloop_eval/"
# ANALYSIS_LIST="eval dir list" # ANALYSIS_LIST="baseline baseline2"
# PATH_TYPE_LIST="full easy hard" # full easy hard

ROOT_DIR='/home/hhl/TravelUAV/TravelUAV'
ANALYSIS_LIST="eval_results_6.7"
PATH_TYPE_LIST="full easy hard"

# CUDA_VISIBLE_DEVICES=0 python3 ./AirVLN/utils/metric.py \
CUDA_VISIBLE_DEVICES=1 python3 /home/hhl/TravelUAV/TravelUAV/utils/metric.py \
    --root_dir $ROOT_DIR \
    --analysis_list $ANALYSIS_LIST \
    --path_type_list $PATH_TYPE_LIST
