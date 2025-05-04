#!/bin/bash

#BATCH --job-name=llmq
#SBATCH --account=yangzhijian
#SBATCH --partition=a100x4
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
source /home/huangmingzhe/project/miniconda3/bin/activate vlm

#python vllm_model.py

vllm serve ../model/Qwen2.5-1.5B-Instruct
#python -m vllm.entrypoints.openai.api_server \
#  --model ../model/QwQ-32B-AWQ \
#  --served-model-name QwQ-32B \
#  --max-model-len=4096


