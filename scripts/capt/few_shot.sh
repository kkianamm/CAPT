#!/bin/bash
# Few-shot for CAPT on BiomedCLIP (all classes).
# Usage: bash scripts/capt/few_shot.sh <data_dir> <dataset> <shots>
#   e.g. CUDA_VISIBLE_DEVICES=0 bash scripts/capt/few_shot.sh data dermamnist 16

DATA=$1
DATASET=$2
SHOTS=$3
BASE_EP=50
CAPT_EP=25

for SEED in 1 2 3
do
  BASE_DIR=output/few_shot/base_model/${DATASET}/shots_${SHOTS}/seed${SEED}
  if [ ! -d "$BASE_DIR" ]; then
    python train_capt.py \
      --root ${DATA} --seed ${SEED} \
      --trainer BiomedCoOp_BiomedCLIP \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --config-file configs/trainers/BiomedCoOp/few_shot/${DATASET}.yaml \
      --output-dir ${BASE_DIR} \
      DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES all
  fi

  DIR=output/few_shot/CAPT_BiomedCLIP/${DATASET}/shots_${SHOTS}/seed${SEED}
  if [ ! -d "$DIR" ]; then
    python train_capt.py \
      --root ${DATA} --seed ${SEED} \
      --trainer CAPT_BiomedCLIP \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --config-file configs/trainers/CAPT/few_shot/${DATASET}.yaml \
      --output-dir ${DIR} \
      DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES all \
      TRAINER.CAPT.BASE_CKPT ${BASE_DIR} TRAINER.CAPT.BASE_EPOCH ${BASE_EP}
  fi
done
