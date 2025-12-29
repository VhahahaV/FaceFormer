#!/bin/bash
# FaceFormer Multi-Dataset Training Script
# 按照 baseline 规范进行训练：支持 digital_human, MEAD_VHAP, MultiModal200 数据集

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Data root directory (按照规范设置)
export DATA_ROOT="/home/caizhuoqiang/hdd/data"

# Run training with conda environment
cd /home/caizhuoqiang/hdd/code/audio_driven_baseline/FaceFormer && \
PYTHONPATH=/home/caizhuoqiang/hdd/code/audio_driven_baseline/FaceFormer conda run -n faceformer python main/train.py \
  --config config/faceformer.yaml

echo "Multi-dataset training completed!"
echo "Model saved to: ./outputs/faceformer_multi_dataset/"
echo "Run inference with: bash inference_faceformer.sh"

