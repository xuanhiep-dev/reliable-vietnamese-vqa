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

lyp_mode=${lyp_mode:-true}
use_selector=${use_selector:-false}
IFS=',' read -ra subset_array <<< "$subsets"


echo "Training from peers (selector: off, subsets: $subsets)"
for id in "${subset_array[@]}"; do
    id=$(echo "$id" | xargs)
    echo "Subset $id has been loaded successfully."
    bash scripts/train_vqa_mulmodels.sh $id $lyp_mode $use_selector
done


