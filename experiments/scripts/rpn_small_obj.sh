#!/bin/bash
# Usage:
# ./experiments/scripts/rpn_small_obj.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is only pascal_voc for now
#
# Example:
# ./experiments/scripts/rpn_small_obj.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400,500,600,700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=80000
    ;;
  coco)
    echo "Not implemented: use experiments/scripts/faster_rcnn_end2end.sh for coco"
    exit
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG_PATH="experiments/logs/rpn_small_obj"
mkdir -p ${LOG_PATH}
LOG="${LOG_PATH}/${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_rpn.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/rpn_small_obj/stage1_rpn_solver60k80k.pt \
  --rpn_test models/${PT_DIR}/${NET}/rpn_small_obj/rpn_test.pt \
  --weights data/imagenet_models/${NET}.v2.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --test_imdb ${TEST_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/rpn_small_obj.yml \
  ${EXTRA_ARGS}

# set +x
# RPN_PROPOSAL_PATH=`grep "Wrote RPN proposals to " ${LOG} | awk '{print $5}'`
# set -x
