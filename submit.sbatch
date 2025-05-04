#!/bin/bash

#BATCH --job-name=llmq
#SBATCH --account=yangzhijian
#SBATCH --partition=a100x4
#SBATCH --gres=gpu:4
#SBATCH --nodes=1

source /home/huangmingzhe/project/miniconda3/bin/activate llmq
python e.py
#python test.py

