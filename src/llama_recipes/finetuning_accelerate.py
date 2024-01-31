# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
os.environ["HF_HOME"] = os.path.join("/code/", "huggingface")
from pkg_resources import packaging

import fire
import random
import torch
import datasets
from accelerate import Accelerator
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
import json
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_int8_training
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from transformers.data.data_collator import DataCollatorMixin
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
)

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.data.data_collator import default_data_collator
from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.configs import dataset_config as DATASET_CONFIG
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    get_dataloader_kwargs,
)
from llama_recipes.utils.dataset_utils import (
    read_files_to_dataset,
    preprocess_function_training,
)
from llama_recipes.utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies
)
from typing import Any, Dict, List, Tuple, Union


class BatchPaddingDataCollator(DataCollatorMixin):
    return_tensors: str = "pt"

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(
        self, features: List[Dict[str, Any]], return_tensors=None
    ) -> Dict[str, Any]:
        if return_tensors is None:
            return_tensors = self.return_tensors
        # get max length
        max_length = max([len(feature["input_ids"]) for feature in features])
        # pad to max length
        for i in range(len(features)):
            features[i]["input_ids"] = [self.tokenizer.pad_token_id] * (
                max_length - len(features[i]["input_ids"])
            ) + features[i]["input_ids"]
            features[i]["attention_mask"] = [0] * (
                max_length - len(features[i]["attention_mask"])
            ) + features[i]["attention_mask"]
            features[i]["labels"] = [-100] * (
                max_length - len(features[i]["labels"])
            ) + features[i]["labels"]
        return default_data_collator(features, return_tensors=return_tensors)


def main(**kwargs):
    # Update the configuration for the training and sharding process
    train_config, fsdp_config, dataset_config = TRAIN_CONFIG(), FSDP_CONFIG(), DATASET_CONFIG()
    update_config((train_config, fsdp_config, dataset_config), **kwargs)
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    additional_args = {}
   # additional_args["low_cpu_mem_usage"] = True
    if train_config.rope_scaling:
        additional_args["rope_scaling"] = json.loads(train_config.rope_scaling)
    # additional_args["use_cache"] = use_cache
    # load model
    model = AutoModelForCausalLM.from_pretrained(
                train_config.model_name,
                torch_dtype=torch.float16 if fsdp_config.use_fp16 else torch.bfloat16,
                **additional_args,
        ).to(local_rank)
    model.gradient_checkpointing_enable()
    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name, use_fast=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        print(f"PEFT Config: {peft_config}")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # read datasets
    training_dataset = read_files_to_dataset(dataset_config.training_folder, dataset_config)
    eval_dataset = None
    if dataset_config.eval_folder:
        eval_dataset = read_files_to_dataset(dataset_config.eval_folder, dataset_config)
    elif dataset_config.split_training_no_eval:
        train_eval_dataset = training_dataset.train_test_split(test_size=dataset_config.split_training_no_eval)
        training_dataset = train_eval_dataset["train"]
        eval_dataset = train_eval_dataset["test"]

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(training_dataset)}")

    if eval_dataset and (not train_config.enable_fsdp or rank == 0):
        print(f"--> Validation Set Length = {len(eval_dataset)}")
    
    # tokenize
    training_dataset = training_dataset.map(
        lambda batch: preprocess_function_training(batch, tokenizer, dataset_config),
        batched=True,
        num_proc=1,
        remove_columns=training_dataset.column_names,
        load_from_cache_file=False,
    )
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            lambda batch: preprocess_function_training(batch, tokenizer, dataset_config),
            batched=True,
            num_proc=1,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=False,
        )

    optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    # train
    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        num_train_epochs=train_config.num_epochs,
        evaluation_strategy="epoch",
        dataloader_drop_last=True,
        logging_strategy="steps",
        logging_steps=10,
        per_device_train_batch_size=train_config.batch_size_training,
        per_device_eval_batch_size=train_config.val_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        gradient_checkpointing=True,
        fp16=True,
        save_strategy="no",
        seed=train_config.seed,
        ddp_find_unused_parameters=False,
        logging_first_step=True,
        optim="adamw_anyprecision",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=eval_dataset,
       # optimizers=(optimizer, scheduler),
        data_collator=BatchPaddingDataCollator(tokenizer),
    )
    trainer.train()
    if accelerator.is_main_process:
        model.save_pretrained(train_config.output_dir)

if __name__ == "__main__":
    fire.Fire(main)
