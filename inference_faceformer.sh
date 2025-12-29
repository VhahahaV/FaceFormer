#!/bin/bash
# FaceFormer Inference Script
# 输出顶点序列到评估脚本兼容的目录结构

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Data root directory (按照规范设置)
export DATA_ROOT="/home/caizhuoqiang/hdd/data"

# Run inference with evaluation-compatible output
cd /home/caizhuoqiang/hdd/code/audio_driven_baseline/FaceFormer && \
PYTHONPATH=/home/caizhuoqiang/hdd/code/audio_driven_baseline/FaceFormer conda run -n faceformer python main/inference.py \
  --config config/faceformer.yaml

echo "Inference completed!"
echo "Results saved to: ./results/metrics/faceformer_run/test/{exp_name}/{DATASET}/{STYLE_ID}/"
echo "Run evaluation with: scripts/eval_motion_guided.sh"

