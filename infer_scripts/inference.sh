#!/usr/bin/env bash
cd ..
echo $PWD
batch_size=1 ## not used
warmup=4000 ## not used
optim='adamw'
model_type='dyn-type'
tgt_emb=768
lr=0.0001 ## not used
learn_loss_weights=0 ## not used
schedule_loss_weight=1 ## not used
lambda1=0.9 ## not used
enc_layers=2
dec_layers=2
export DATA_PATH=$1
export EXP_NAME=$2
tbd='evaltensorboard'
n='layer'${enc_layers}${dec_layers}'_noft'
export SAVE_DIR='log_evaluations/'${EXP_NAME}'_'${n}
mkdir ${SAVE_DIR}
export model_path=$3

CUDA_VISIBLE_DEVICES=0 python inference_dyn.py  \
--resume ${model_path} \
--fine_tune_bert 0 \
--data_path ${DATA_PATH} \
--snapshots ${SAVE_DIR}/snapshots \
--path_results  ${SAVE_DIR}/results \
--path_inference ${SAVE_DIR}/inference \
--batch_size ${batch_size} \
--epochs 10 \
--warmup ${warmup} \
--valfreq 1 \
--type_acc 'pooler_output' \
--sim 'cos' \
--lambda1 ${lambda1} \
--optim ${optim} \
--model_type ${model_type} \
--learn_loss_weights ${learn_loss_weights} \
--schedule_loss_weight ${schedule_loss_weight} \
--lr ${lr} \
--beta 0.5 \
--iter_type 'binary_batch' \
--enc_layers ${enc_layers} \
--dec_layers ${dec_layers} \
--tbd ${tbd} \
--logdir 'logs' \
--expname ${EXP_NAME} \
--use_wandb 0 \
--gpus 1

