#!/bin/sh

#export CUDA_HOME=/opt/cuda-9.0.176.1/
#source activate pytorch

EXECUTABLE_FILE=~/Repositories/ZeroShotKnowledgeTransfer/main.py
LOG_DIR=~/Repositories/ZeroShotKnowledgeTransfer/logs
DATASETS_DIR=~

python3 ${EXECUTABLE_FILE} \
--dataset "CIFAR-FS" \
--total_n_pseudo_batches 4e4 \
--n_generator_iter 1 \
--n_student_iter 10 \
--batch_size 64 \
--z_dim 100 \
--x_channels 3 \
--x_dim 32 \
--student_learning_rate 2e-3 \
--generator_learning_rate 1e-3 \
--teacher_architecture "Conv4" \
--student_architecture "Conv4" \
--KL_temperature 1 \
--AT_beta 0 \
--pretrained_models_path ~/Repositories/master_thesis_sg/results/CIFAR-FS/batch_size_4_meta_epochs_100_max_batches_100_max_batches_val_100_seed_123/'teacher_model_2020-07-03 14:25:21.562470.th' \
--datasets_path ${DATASETS_DIR} \
--log_directory_path ${LOG_DIR} \
--save_final_model True \
--save_n_checkpoints 0 \
--save_model_path ${LOG_DIR} \
--seeds 0 1 2 \
--workers 2 \
--device "cuda:2"
