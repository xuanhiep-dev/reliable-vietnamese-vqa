#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

parse_args() {
  for arg in "$@"; do
    [[ $arg == *=* ]] && eval "${arg%%=*}='${arg#*=}'"
  done
}
parse_args "$@"

lyp_mode=${lyp_mode:-false}
train_path="data/full/train.csv"
valid_path="data/full/valid.csv"
test_path="data/full/test.csv"

python main.py --set training.lyp_mode="$lyp_mode" \
                     model.use_selector="$use_selector" \
                     paths.checkpoints.save_path="$save_checkpoint_path" \
                     paths.checkpoints.load_path="$load_checkpoint_path" \
                     paths.train_path="$train_path" \
                     paths.valid_path="$valid_path" \
                     paths.test_path="$test_path"
