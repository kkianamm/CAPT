#!/bin/bash
# Few-shot for CAPT on BiomedCLIP (all classes) -- SELF-CONTAINED, no BiomedCoOp.
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/capt/few_shot.sh data dermamnist 16

DATA=$1
DATASET=$2
SHOTS=$3

for SEED in 1 2 3
do
  DIR=output/few_shot/CAPT_BiomedCLIP/${DATASET}/shots_${SHOTS}/seed${SEED}
  if [ -d "$DIR" ]; then
    echo "Results exist at ${DIR} (skip)"
  else
    python train_capt.py \
      --root ${DATA} --seed ${SEED} \
      --trainer CAPT_BiomedCLIP \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --config-file configs/trainers/CAPT/few_shot/${DATASET}.yaml \
      --output-dir ${DIR} \
      DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES all
  fi
done
