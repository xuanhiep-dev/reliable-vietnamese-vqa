#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

for id in 1 2
do
    bash scripts/train_vqa_mulmodels.sh $id
done
