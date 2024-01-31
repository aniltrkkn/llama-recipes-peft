# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class dataset_config:
    training_folder: str=None
    eval_folder: str=None
    split_training_no_eval: float=0.15
    json_file_input_column: str="inputs_pretokenized"
    json_file_target_column: str="targets_pretokenized"
    huggingface_file_input_column: str="input"
    huggingface_file_target_column: str="target"
    is_mixtral: bool=False
    is_mistral_instruct: bool=False
