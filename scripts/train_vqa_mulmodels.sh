#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

subset=$1
image_path="data/images"
output_dir="data/"
train_path="data/train-subsets/train-subset-$subset.csv"
valid_path="data/valid-subsets/valid-subset-$subset.csv"
test_path="data/test-subsets/test-subset-$subset.csv"
ans_path="data/answers.json"
checkpoint_dir="vqa_checkpoints"
predictions_dir="data/multi-predictions"
train_batch_size=32
eval_batch_size=32
encoder_layers=6
encoder_attention_heads_layers=6
epoch=1

# Chạy training script với các tham số cụ thể
python main.py --log-level 'info' \
               --checkpoint-dir ${checkpoint_dir} \
               --sub-id ${subset} \
               --image-path ${image_path} \
               --train-path ${train_path} \
               --val-path ${valid_path} \
               --test-path ${test_path} \
               --ans-path ${ans_path} \
               --predictions-dir ${predictions_dir} \
               --train-batch-size ${train_batch_size} \
               --eval-batch-size ${eval_batch_size} \
               --encoder-layers ${encoder_layers} \
               --encoder-attention-heads-layers ${encoder_attention_heads_layers} \
               --epoch ${epoch} \
