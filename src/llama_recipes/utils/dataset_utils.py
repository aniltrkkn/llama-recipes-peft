# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import importlib
from functools import partial
from pathlib import Path

import torch
import json
import os

from datasets import Dataset


def read_files_to_dataset(folder, dataset_config, rank) -> Dataset:
    """
    This method reads all files in the given folder
    """
    # read data in folders one by one
    # loop through all files in the folder
    dataset_dict = {}
    input_column = dataset_config.huggingface_file_input_column
    target_column = dataset_config.huggingface_file_target_column
    for filename in os.listdir(folder):
        # check if directory or file
        if os.path.isdir(os.path.join(folder, filename)):
            continue
        # get the file path
        file_path = os.path.join(folder, filename)
        # check json
        data = {input_column: [], target_column: []}
        if "json" in file_path:
            # open the file
            with open(file_path, encoding="utf-8") as f:
                # read all text in the file
                json_data = json.load(f)
                # get the input and target
                for i in range(0, len(json_data)):
                    # check for unknown
                    if "resolution for the case is unknown" in json_data[i][dataset_config.json_file_target_column]:
                        continue
                    data[input_column].append(json_data[i][dataset_config.json_file_input_column])
                    data[target_column].append(json_data[i][dataset_config.json_file_target_column])
        else:
            dataset_single = Dataset.from_file(file_path)
            data[input_column] = dataset_single[input_column]
            data[target_column] = dataset_single[target_column]
        dataset_dict[filename] = data
    # print training data size
    if rank == 0:
        for key in dataset_dict:
            print(key, " Data Size: ", len(dataset_dict[key][input_column]))
    # combine to single dataset
    dataset = {input_column: [], target_column: []}
    for key in dataset_dict:
        dataset[input_column] += dataset_dict[key][input_column]
        dataset[target_column] += dataset_dict[key][target_column]
    dataset = Dataset.from_dict(dataset)
    # remove old tags
    def replace_old_tags(example):
        example[input_column] = (
            example[input_column]
            .replace("<|system|>", "<|system|>\n")
            .replace("<|customer|>", "<|user|>\n")
            .replace("<|agent|>", "<|assistant|>\n")
            .replace("<|endoftext|>", "<|end|>\n")
        )
        example[target_column] = example[target_column].replace("<|endoftext|>", "<|end|>")
        return example
    dataset = dataset.map(
        replace_old_tags,
        batched=False,
        num_proc=1,
        load_from_cache_file=False,
    )
    return dataset


# custom preprocess function to calculate the loss only on the labels
def preprocess_function_training(batch, tokenizer, dataset_config):
    batch_size = len(batch[dataset_config.huggingface_file_input_column])
    # create the model inputs
    inputs = batch[dataset_config.huggingface_file_input_column]
    targets = [str(x) for x in batch[dataset_config.huggingface_file_target_column]]
    # tokenize the inputs and labels
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    # keep track of truncated inputs
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        # add bos_token_id to the labels - model should stop predicting after this token
        label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
        # concatenate the labels to the end of the sample
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        # ignore everything before the label in loss calculations
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    for i in range(batch_size):
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i])
        model_inputs["attention_mask"][i] = torch.tensor(
            model_inputs["attention_mask"][i]
        )
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
