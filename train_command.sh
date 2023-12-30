#!/bin/bash
DATA_PATH="binary_jointnel_mainsplit_bs8_last3/"
SAVE_DIR='output/"
python train_dyn.py  \
--resume "none" \
--fine_tune_bert 0 \
--data_path ${DATA_PATH} \
--snapshots ${SAVE_DIR}/snapshots \
--path_results  ${SAVE_DIR}/results \
--path_inference ${SAVE_DIR}/inference \
--batch_size 1 \
--epochs 10 \
--valfreq 1 \
--type_acc 'pooler_output' \
--sim 'cos' \
--optim 'adamw' \
--model_type 'dyn-type' \
--learn_loss_weights 0 \
--schedule_loss_weight 1 \
--lr 0.0001 \
--beta 0.5 \
--iter_type 'binary_batch' \
--enc_layers 2 \
--dec_layers 2 \
--tbd $SAVE_DIR'/tensorboard_logs' \
--logdir 'logs' \
--expname "experiment_name" \
--use_wandb 0 \
--gpus 1

