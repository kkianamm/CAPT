#!/bin/bash
# Base-to-Novel for CAPT on BiomedCLIP -- SELF-CONTAINED (no BiomedCoOp weights).
# Works for any BiomedCoOp dataset; the eval epoch is chosen per dataset so it
# matches MAX_EPOCH in that dataset's CAPT config.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/capt/base2new.sh data retina
#   CUDA_VISIBLE_DEVICES=0 bash scripts/capt/base2new.sh data dermamnist
#   # optional 3rd arg overrides the eval epoch:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/capt/base2new.sh data retina 50

DATA=$1
DATASET=$2
SHOTS=16
NCTX=4
CSC=False
CTP=end

# eval epoch must equal MAX_EPOCH in configs/trainers/CAPT/base_to_novel/<dataset>.yaml
case $DATASET in
  retina)     DEF_EP=50 ;;
  dermamnist) DEF_EP=25 ;;
  *)          DEF_EP=50 ;;
esac
CAPT_EP=${3:-$DEF_EP}

for SEED in 1 2 3
do
  COMMON=${DATASET}/shots_${SHOTS}/CAPT_BiomedCLIP/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
  TRAIN_DIR=output/base2new/train_base/${COMMON}

  # ---------- Train CAPT on BASE classes ----------
  if [ -d "$TRAIN_DIR" ]; then
    echo "CAPT results exist at ${TRAIN_DIR} (skip training)"
  else
    python train_capt.py \
      --root ${DATA} --seed ${SEED} \
      --trainer CAPT_BiomedCLIP \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --config-file configs/trainers/CAPT/base_to_novel/${DATASET}.yaml \
      --output-dir ${TRAIN_DIR} \
      DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES base
  fi

  # ---------- Evaluate on BASE and NOVEL ----------
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
