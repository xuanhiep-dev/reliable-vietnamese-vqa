#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

subset=$1
use_selector=$2
lyp_mode=$3
train_path="data/subsets/train/train-$subset.csv"
valid_path="data/subsets/valid/valid-$subset.csv"
test_path="data/subsets/test/test-$subset.csv"
 
python main.py --set paths.subset_id=${subset} \
                     model.use_selector=${use_selector} \
                     training.lyp_mode=${lyp_mode} \
                     paths.train_path=${train_path} \
                     paths.valid_path=${valid_path} \
                     paths.test_path=${test_path}
