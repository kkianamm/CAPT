#!/bin/bash
# Base-to-Novel for CAPT on BiomedCLIP.
# Stage 1: train a base BiomedCoOp model on the BASE classes (builds the model
#          whose predictions seed CAPT's Confusion Bank).
# Stage 2: train CAPT on the BASE classes using that checkpoint, then evaluate
#          on BASE and NOVEL classes.
#
# Usage: bash scripts/capt/base2new.sh <data_dir> <dataset>
#   e.g. CUDA_VISIBLE_DEVICES=0 bash scripts/capt/base2new.sh data dermamnist

DATA=$1
DATASET=$2
SHOTS=16
NCTX=4
CSC=False
CTP=end
BASE_EP=50      # epochs for the Stage-1 BiomedCoOp base model
CAPT_EP=25      # epochs for Stage-2 CAPT

for SEED in 1 2 3
do
  # ---------- Stage 1: base BiomedCoOp model on BASE classes ----------
  BASE_DIR=output/base2new/base_model/${DATASET}/shots_${SHOTS}/seed${SEED}
  if [ -d "$BASE_DIR" ]; then
    echo "Base model exists at ${BASE_DIR} (skip Stage 1)"
  else
    python train_capt.py \
      --root ${DATA} --seed ${SEED} \
      --trainer BiomedCoOp_BiomedCLIP \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --config-file configs/trainers/BiomedCoOp/base_to_novel/${DATASET}.yaml \
      --output-dir ${BASE_DIR} \
      DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES base
  fi

  # ---------- Stage 2: CAPT on BASE classes ----------
  COMMON=${DATASET}/shots_${SHOTS}/CAPT_BiomedCLIP/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
  TRAIN_DIR=output/base2new/train_base/${COMMON}
  if [ -d "$TRAIN_DIR" ]; then
    echo "CAPT results exist at ${TRAIN_DIR} (skip training)"
  else
    python train_capt.py \
      --root ${DATA} --seed ${SEED} \
      --trainer CAPT_BiomedCLIP \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --config-file configs/trainers/CAPT/base_to_novel/${DATASET}.yaml \
      --output-dir ${TRAIN_DIR} \
      DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES base \
      TRAINER.CAPT.BASE_CKPT ${BASE_DIR} TRAINER.CAPT.BASE_EPOCH ${BASE_EP}
  fi

  # ---------- Eval on BASE and NOVEL ----------
  for SUB in base new
  do
    TEST_DIR=output/base2new/test_${SUB}/${COMMON}
    if [ -d "$TEST_DIR" ]; then
      echo "Eval results exist at ${TEST_DIR} (skip)"
    else
      python train_capt.py \
        --root ${DATA} --seed ${SEED} \
        --trainer CAPT_BiomedCLIP \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/CAPT/base_to_novel/${DATASET}.yaml \
        --output-dir ${TEST_DIR} \
        --model-dir ${TRAIN_DIR} --load-epoch ${CAPT_EP} --eval-only \
        DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
  done
done
