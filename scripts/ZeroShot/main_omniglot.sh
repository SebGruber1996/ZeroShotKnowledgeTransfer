#!/bin/sh

#export CUDA_HOME=/opt/cuda-9.0.176.1/
#source activate pytorch

EXECUTABLE_FILE=~/Repositories/ZeroShotKnowledgeTransfer/main.py
LOG_DIR=~/Repositories/ZeroShotKnowledgeTransfer/logs
DATASETS_DIR=~

python3 ${EXECUTABLE_FILE} \
--dataset "Omniglot" \
--total_n_pseudo_batches 2e4 \
--n_generator_iter 1 \
--n_student_iter 10 \
--batch_size 64 \
--z_dim 100 \
--x_channels 1 \
--x_dim 28 \
--student_learning_rate 2e-3 \
--generator_learning_rate 1e-3 \
--teacher_architecture "Conv4" \
--student_architecture "Conv4" \
--KL_temperature 1 \
--AT_beta 0 \
--pretrained_models_path ~/Repositories/master_thesis_sg/results/Omniglot/batch_size_8_meta_epochs_100_max_batches_100_max_batches_val_100_seed_123/'teacher_model_2020-07-01 17:04:22.504888.th' \
--datasets_path ${DATASETS_DIR} \
--log_directory_path ${LOG_DIR} \
--save_final_model True \
--save_n_checkpoints 0 \
--save_model_path ${LOG_DIR} \
--seeds 3 4 5 \
--workers 2 \
--device "cuda:2"
