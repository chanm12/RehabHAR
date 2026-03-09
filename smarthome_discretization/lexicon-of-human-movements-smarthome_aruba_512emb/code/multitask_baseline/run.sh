#!/usr/bin/env bash
# Multi encoder
#CUDA_VISIBLE_DEVICES=1 python -u main.py --finetune False --gpu_id 0 --window 400 --overlap 0

#CUDA_VISIBLE_DEVICES=1 python -u main.py --finetune False --gpu_id 0 --window 400 --overlap 0

# ----------------------------------------------------------------------------------------------------------------------
# 10/11/2021
# ----------------------------------------------------------------------------------------------------------------------
#CUDA_VISIBLE_DEVICES=1 python evaluate_with_classifier.py --dataset motionsense_3 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-09-2021/multitask_baseline_capture_24_schedule_last_layer_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/motionsense/all_data/Oct-10-2021 \
#--data_file motionsense_3_sr_50_fold_0.joblib &


# ----------------------------------------------------------------------------------------------------------------------
# 10/12/2021
# ----------------------------------------------------------------------------------------------------------------------
# Classifying on PAMAP2
#CUDA_VISIBLE_DEVICES=3 python finetuner.py --dataset pamap2_3 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-09-2021/multitask_baseline_capture_24_schedule_last_layer_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/pamap2/all_data/Oct-12-2021 \
#--data_file pamap2_3_sr_50.joblib &

# Pre-training with differing settings of learning rate and weight decay
#CUDA_VISIBLE_DEVICES=1 python -u main.py --window 400 --overlap 0 --learning_rate 1e-4 --weight_decay 0 &
#CUDA_VISIBLE_DEVICES=2 python -u main.py --window 400 --overlap 0 --learning_rate 1e-4 --weight_decay 1e-4 &
#CUDA_VISIBLE_DEVICES=3 python -u main.py --window 400 --overlap 0 --learning_rate 1e-4 --weight_decay 5e-4 &

#CUDA_VISIBLE_DEVICES=4 python -u main.py --window 400 --overlap 0 --learning_rate 3e-4 --weight_decay 0 &
#CUDA_VISIBLE_DEVICES=3 python -u main.py --window 400 --overlap 0 --learning_rate 3e-4 --weight_decay 1e-4 &
#CUDA_VISIBLE_DEVICES=2 python -u main.py --window 400 --overlap 0 --learning_rate 3e-4 --weight_decay 5e-4 &
#
#CUDA_VISIBLE_DEVICES=3 python -u main.py --window 400 --overlap 0 --learning_rate 5e-4 --weight_decay 0 &
#CUDA_VISIBLE_DEVICES=4 python -u main.py --window 400 --overlap 0 --learning_rate 5e-4 --weight_decay 1e-4 &
#CUDA_VISIBLE_DEVICES=1 python -u main.py --window 400 --overlap 0 --learning_rate 5e-4 --weight_decay 5e-4 &


# ----------------------------------------------------------------------------------------------------------------------
# 10/13/2021
# ----------------------------------------------------------------------------------------------------------------------
# testing early stopping
#CUDA_VISIBLE_DEVICES=4 python -u main.py --window 400 --overlap 0 --learning_rate 1e-3 --weight_decay 0 \
#--root_dir /coc/pcba1/hharesamudram3/data_preparation/capture_24/all_data/Sep-08-2021 --data_file capture_24_debug_sr_50.joblib

# ----------------------------------------------------------------------------------------------------------------------
# 10/17/2021
# ----------------------------------------------------------------------------------------------------------------------
# Hyperparameter tuning for the fine-tuning
# Mobiactv2
# lr=1e-4
#CUDA_VISIBLE_DEVICES=1 python finetuner.py --dataset mobiactv2_6 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 1e-4 --weight_decay 0 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-13-2021/multitask_capture_24_lr_0.0001_wd_0.0__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/mobiactv2/all_data/Oct-10-2021 \
#--data_file mobiactv2_6_sr_50_fold_0.joblib &
#
#CUDA_VISIBLE_DEVICES=1 python finetuner.py --dataset mobiactv2_6 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 1e-4 --weight_decay 1e-4 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-13-2021/multitask_capture_24_lr_0.0001_wd_0.0001__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/mobiactv2/all_data/Oct-10-2021 \
#--data_file mobiactv2_6_sr_50_fold_0.joblib &
#
#CUDA_VISIBLE_DEVICES=1 python finetuner.py --dataset mobiactv2_6 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 1e-4 --weight_decay 5e-4 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-13-2021/multitask_capture_24_lr_0.0001_wd_0.0005__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/mobiactv2/all_data/Oct-10-2021 \
#--data_file mobiactv2_6_sr_50_fold_0.joblib &
#
## lr=3e-4
#CUDA_VISIBLE_DEVICES=2 python finetuner.py --dataset mobiactv2_6 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 3e-4 --weight_decay 0 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-13-2021/multitask_capture_24_lr_0.0003_wd_0.0__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/mobiactv2/all_data/Oct-10-2021 \
#--data_file mobiactv2_6_sr_50_fold_0.joblib &
#
#CUDA_VISIBLE_DEVICES=2 python finetuner.py --dataset mobiactv2_6 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 3e-4 --weight_decay 1e-4 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-13-2021/multitask_capture_24_lr_0.0003_wd_0.0001__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/mobiactv2/all_data/Oct-10-2021 \
#--data_file mobiactv2_6_sr_50_fold_0.joblib &
#
#CUDA_VISIBLE_DEVICES=2 python finetuner.py --dataset mobiactv2_6 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 3e-4 --weight_decay 5e-4 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-13-2021/multitask_capture_24_lr_0.0003_wd_0.0005__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/mobiactv2/all_data/Oct-10-2021 \
#--data_file mobiactv2_6_sr_50_fold_0.joblib &
#
## lr=5e-4
#CUDA_VISIBLE_DEVICES=3 python finetuner.py --dataset mobiactv2_6 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 5e-4 --weight_decay 0 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-14-2021/multitask_capture_24_lr_0.0005_wd_0.0__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/mobiactv2/all_data/Oct-10-2021 \
#--data_file mobiactv2_6_sr_50_fold_0.joblib &
#
#CUDA_VISIBLE_DEVICES=3 python finetuner.py --dataset mobiactv2_6 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 5e-4 --weight_decay 1e-4 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-14-2021/multitask_capture_24_lr_0.0005_wd_0.0001__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/mobiactv2/all_data/Oct-10-2021 \
#--data_file mobiactv2_6_sr_50_fold_0.joblib &
#
#CUDA_VISIBLE_DEVICES=3 python finetuner.py --dataset mobiactv2_6 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 5e-4 --weight_decay 5e-4 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-14-2021/multitask_capture_24_lr_0.0005_wd_0.0005__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/mobiactv2/all_data/Oct-10-2021 \
#--data_file mobiactv2_6_sr_50_fold_0.joblib &

# Motionsense
# lr=1e-4
#CUDA_VISIBLE_DEVICES=1 python finetuner.py --dataset motionsense_3 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 1e-4 --weight_decay 0 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-13-2021/multitask_capture_24_lr_0.0001_wd_0.0__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/motionsense/all_data/Oct-10-2021 \
#--data_file motionsense_3_sr_50_fold_0.joblib &
#
#CUDA_VISIBLE_DEVICES=1 python finetuner.py --dataset motionsense_3 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 1e-4 --weight_decay 1e-4 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-13-2021/multitask_capture_24_lr_0.0001_wd_0.0001__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/motionsense/all_data/Oct-10-2021 \
#--data_file motionsense_3_sr_50_fold_0.joblib &
#
#CUDA_VISIBLE_DEVICES=1 python finetuner.py --dataset motionsense_3 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 1e-4 --weight_decay 5e-4 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-13-2021/multitask_capture_24_lr_0.0001_wd_0.0005__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/motionsense/all_data/Oct-10-2021 \
#--data_file motionsense_3_sr_50_fold_0.joblib &
#
## lr=3e-4
#CUDA_VISIBLE_DEVICES=2 python finetuner.py --dataset motionsense_3 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 3e-4 --weight_decay 0 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-13-2021/multitask_capture_24_lr_0.0003_wd_0.0__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/motionsense/all_data/Oct-10-2021 \
#--data_file motionsense_3_sr_50_fold_0.joblib &
#
#CUDA_VISIBLE_DEVICES=2 python finetuner.py --dataset motionsense_3 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 3e-4 --weight_decay 1e-4 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-13-2021/multitask_capture_24_lr_0.0003_wd_0.0001__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/motionsense/all_data/Oct-10-2021 \
#--data_file motionsense_3_sr_50_fold_0.joblib &
#
#CUDA_VISIBLE_DEVICES=2 python finetuner.py --dataset motionsense_3 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 3e-4 --weight_decay 5e-4 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-13-2021/multitask_capture_24_lr_0.0003_wd_0.0005__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/motionsense/all_data/Oct-10-2021 \
#--data_file motionsense_3_sr_50_fold_0.joblib &
#
## lr=5e-4
#CUDA_VISIBLE_DEVICES=3 python finetuner.py --dataset motionsense_3 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 5e-4 --weight_decay 0 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-14-2021/multitask_capture_24_lr_0.0005_wd_0.0__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/motionsense/all_data/Oct-10-2021 \
#--data_file motionsense_3_sr_50_fold_0.joblib &
#
#CUDA_VISIBLE_DEVICES=3 python finetuner.py --dataset motionsense_3 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 5e-4 --weight_decay 1e-4 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-14-2021/multitask_capture_24_lr_0.0005_wd_0.0001__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/motionsense/all_data/Oct-10-2021 \
#--data_file motionsense_3_sr_50_fold_0.joblib &
#
#CUDA_VISIBLE_DEVICES=3 python finetuner.py --dataset motionsense_3 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 5e-4 --weight_decay 5e-4 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-14-2021/multitask_capture_24_lr_0.0005_wd_0.0005__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/motionsense/all_data/Oct-10-2021 \
#--data_file motionsense_3_sr_50_fold_0.joblib &


# PAMAP2
# lr=1e-4
#CUDA_VISIBLE_DEVICES=1 python finetuner.py --dataset pamap2_3 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 1e-4 --weight_decay 0 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-13-2021/multitask_capture_24_lr_0.0001_wd_0.0__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/pamap2/all_data/Oct-12-2021 \
#--data_file pamap2_3_sr_50.joblib &
#
#CUDA_VISIBLE_DEVICES=1 python finetuner.py --dataset pamap2_3 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 1e-4 --weight_decay 1e-4 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-13-2021/multitask_capture_24_lr_0.0001_wd_0.0001__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/pamap2/all_data/Oct-12-2021 \
#--data_file pamap2_3_sr_50.joblib &
#
#CUDA_VISIBLE_DEVICES=1 python finetuner.py --dataset pamap2_3 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 1e-4 --weight_decay 5e-4 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-13-2021/multitask_capture_24_lr_0.0001_wd_0.0005__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/pamap2/all_data/Oct-12-2021 \
#--data_file pamap2_3_sr_50.joblib &
#
## lr=3e-4
#CUDA_VISIBLE_DEVICES=2 python finetuner.py --dataset pamap2_3 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 3e-4 --weight_decay 0 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-13-2021/multitask_capture_24_lr_0.0003_wd_0.0__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/pamap2/all_data/Oct-12-2021 \
#--data_file pamap2_3_sr_50.joblib &
#
#CUDA_VISIBLE_DEVICES=2 python finetuner.py --dataset pamap2_3 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 3e-4 --weight_decay 1e-4 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-13-2021/multitask_capture_24_lr_0.0003_wd_0.0001__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/pamap2/all_data/Oct-12-2021 \
#--data_file pamap2_3_sr_50.joblib &
#
#CUDA_VISIBLE_DEVICES=2 python finetuner.py --dataset pamap2_3 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 3e-4 --weight_decay 5e-4 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-13-2021/multitask_capture_24_lr_0.0003_wd_0.0005__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/pamap2/all_data/Oct-12-2021 \
#--data_file pamap2_3_sr_50.joblib &
#
## lr=5e-4
#CUDA_VISIBLE_DEVICES=3 python finetuner.py --dataset pamap2_3 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 5e-4 --weight_decay 0 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-14-2021/multitask_capture_24_lr_0.0005_wd_0.0__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/pamap2/all_data/Oct-12-2021 \
#--data_file pamap2_3_sr_50.joblib &
#
#CUDA_VISIBLE_DEVICES=3 python finetuner.py --dataset pamap2_3 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 5e-4 --weight_decay 1e-4 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-14-2021/multitask_capture_24_lr_0.0005_wd_0.0001__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/pamap2/all_data/Oct-12-2021 \
#--data_file pamap2_3_sr_50.joblib &
#
#CUDA_VISIBLE_DEVICES=3 python finetuner.py --dataset pamap2_3 --window 400 --overlap 200 --classifier_lr 1e-4 \
#--learning_schedule last_layer --learning_rate 5e-4 --weight_decay 5e-4 \
#--saved_model /coc/pcba1/hharesamudram3/capture_24/code/multitask_baseline/models/Oct-14-2021/multitask_capture_24_lr_0.0005_wd_0.0005__bs_256_rs_42_multiple_runs_0.pkl \
#--root_dir /coc/pcba1/hharesamudram3/capture_24/code/data_preparation/pamap2/all_data/Oct-12-2021 \
#--data_file pamap2_3_sr_50.joblib &


# ----------------------------------------------------------------------------------------------------------------------
# 12/04/2021
# ----------------------------------------------------------------------------------------------------------------------
# Training models with Ray
#CUDA_VISIBLE_DEVICES=2,3,4 python classifier.py

# ----------------------------------------------------------------------------------------------------------------------
# 12/20/2021
# ----------------------------------------------------------------------------------------------------------------------
# Pre-training models with Ray and using a smaller window size
CUDA_VISIBLE_DEVICES=2,3,4 python main.py
