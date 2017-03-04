#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_voc_2007_SR.sh GPU SR_METHOD [options args to test_net.py]

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
SR_METHOD=$2

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}
TEST_IMDB="voc_2007_${SR_METHOD}"

# case $DATASET in
#   pascal_voc)
#     TRAIN_IMDB="voc_2007_trainval"
#     TEST_IMDB="voc_2007_" + ${SR_METHOD}
#     PT_DIR="pascal_voc"
#     ITERS=70000
#     ;;
#   coco)
#     # This is a very long and slow training schedule
#     # You can probably use fewer iterations and reduce the
#     # time to the LR drop (set in the solver to 350,000 iterations).
#     echo "COCO test not implemented"
#     exit
#     ;;
#   *)
#     echo "No dataset given"
#     exit
#     ;;
# esac

LOG="experiments/logs/faster_rcnn_voc_2007_vgg16_${SR_METHOD}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# time ./tools/train_net.py --gpu ${GPU_ID} \
#   --solver models/${PT_DIR}/${NET}/faster_rcnn_end2end/solver.prototxt \
#   --weights data/imagenet_models/${NET}.v2.caffemodel \
#   --imdb ${TRAIN_IMDB} \
#   --iters ${ITERS} \
#   --cfg experiments/cfgs/faster_rcnn_end2end.yml \
#   ${EXTRA_ARGS}
#
# set +x
# NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
# set -x

NET_FINAL="output/faster_rcnn_end2end/voc_2007_trainval/vgg16_faster_rcnn_iter_70000.caffemodel"

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}
