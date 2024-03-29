#!/usr/bin/env bash
set -x

TYPE=$1
FOLD=$2
PERCENT=$3
GPUS=$4
FROM=$5
PORT=${PORT:-29487}

export CUDA_VISIBLE_DEVICES=4,5,6,7
PYTHONPAT=H"$(dirname $0)/..":$PYTHONPATH

if [[ ${TYPE} == 'baseline' ]]; then
    python -W ignore -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        $(dirname "$0")/train.py configs/baseline/faster_rcnn_r50_gradcam.py --launcher pytorch \
        --cfg-options fold=${FOLD} percent=${PERCENT} ${@:5}  \
        --work-dir "/home/danshili/softTeacher/SoftTeacher/work_dirs/gradcam-exp/4GPU-without-cam-stopped-80000" \

else
    python -W ignore -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        $(dirname "$0")/train.py configs/soft_teacher/soft_teacher_faster_rcnn_r50_gradcam.py --launcher pytorch \
        --cfg-options fold=${FOLD} percent=${PERCENT} ${@:5}  \
        --work-dir "/home/danshili/softTeacher/SoftTeacher/work_dirs/gradcam-exp/4GPU-without-cam-stopped-80000" \

fi
