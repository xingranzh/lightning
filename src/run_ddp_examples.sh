#!/bin/bash
# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -e

# test that a user can manually launch individual processes
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
args="--trainer.gpus 2 --trainer.strategy ddp --trainer.max_epochs=1 --trainer.limit_train_batches=1 --trainer.limit_val_batches=1 --trainer.limit_test_batches=1"
MASTER_ADDR="localhost" MASTER_PORT=1234 LOCAL_RANK=1 python pl_examples/basic_examples/mnist_examples/image_classifier_5_lightning_datamodule.py ${args} &
MASTER_ADDR="localhost" MASTER_PORT=1234 LOCAL_RANK=0 python pl_examples/basic_examples/mnist_examples/image_classifier_5_lightning_datamodule.py ${args}
