#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

for id in 1 2 3 4 5 6 7 8 9 10
do
    bash scripts/train_vqa_mulmodels.sh $id
done
