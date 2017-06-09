#!/bin/bash
#
# Copyright 2017    Ke Wang     Xiaomi
#
# Train LSTM model using TensorFlow for speech enhancement

set -euo pipefail

stage=3
data=data
fs=8k
config=lists
exp=exp_${fs}/lstm
rnn_size=256
num_layers=3
keep_prob=0.5
input_dim=129
output_dim=129
lr=0.0005
gpu_id=1
assign='opt'
decode=1
save_mask=False
data_dir=`pwd`/data/wsj0_separation/feats_separated/wsj0_lstm_${fs}_${assign}/
ori_wav_path=data/wsj0/create-speaker-mixtures/data/2speakers/wav${fs}/min/tt/mix
rec_wav_path=data/wsj0_separation/wav/rec_wav_lstm_${rnn_size}_${num_layers}_${fs}_${assign}/
 
if [ $decode -eq 1 ]; then
  batch_size=1
else
  batch_size=40
fi

# Prepare data
if [ $stage -le 0 ]; then
  python misc/get_train_val_scp.py
fi

# Prepare TFRecords format data
if [ $stage -le 1 ]; then
  [ ! -e "$data/tfrecords" ] && mkdir -p "$data/tfrecords"
  [ ! -e "$data/tfrecords/train" ] && mkdir -p "$data/tfrecords/train"
  [ ! -e "$data/tfrecords/val" ] && mkdir -p "$data/tfrecords/val"
  [ ! -e "$data/tfrecords/test" ] && mkdir -p "$data/tfrecords/test"
  python utils/convert_to_records_parallel.py \
    --data_dir=$data/raw/cmvn \
    --output_dir=$data/tfrecords \
    --config_dir=$config \
    --num_thread=12
fi

# Train LSTM model
if [ $stage -le 2 ]; then
  echo "Start train LSTM model."
  CUDA_VISIBLE_DEVICES=$gpu_id TF_CPP_MIN_LOG_LEVEL=1 python -u  run_lstm_8k.py \
    --config_dir=$config  --rnn_num_layers=$num_layers --rnn_size=$rnn_size  --batch_size=$batch_size --decode=$decode \
    --learning_rate=$lr --save_dir=$exp --data_dir=$data_dir --keep_prob=$keep_prob \
    --input_dim=$input_dim --output_dim=$output_dim --assign='opt' --save_mask=$save_mask
fi
if [ $stage -le 3 ]; then
  npy_list=tmp.lst
  echo "Reconstructing time domain separated speech signal \n"
  echo "Store the reconstructed wav to $rec_wav_path \n"
  mkdir -p $rec_wav_path
  find $data_dir -iname "*.npy" > $npy_list
  for line in `cat $npy_list`; do
    wavname=`basename -s .npy $line`
    w=`echo $wavname | awk -F '_' 'BEGIN{OFS="_"}{print $1,$2,$3,$4}'` 
    w=${w}.wav
    python ./utils/reconstruct_spectrogram.py $line ${ori_wav_path}/$w ${rec_wav_path}/${wavname} || exit 1
  done

rm  $npy_list
echo "Done OK!"
fi
