#!/usr/bin/env bash
python train_arcface.py \
  --batch-size=256 \
  --log-interval=20 \
  --gpus=0 \
  --lr=0.001 \
  --mode=hybrid \
  --save-type=mxnet
