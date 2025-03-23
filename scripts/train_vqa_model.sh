#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

use_selector=$1
output_path=$2
train_path="data/full/train.csv"
valid_path="data/full/valid.csv"
test_path="data/full/test.csv"


python main.py --set model.use_selector=${use_selector} \
                     paths.output_path=${output_path} \
                     paths.train_path=${train_path} \
                     paths.valid_path=${valid_path} \
                     paths.test_path=${test_path}
