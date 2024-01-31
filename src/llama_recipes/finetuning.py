# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
os.environ["TRANSFORMERS_CACHE"] = os.path.join("/code/", "huggingface")
os.environ["HF_HOME"] = os.path.join("/code/", "huggingface")
from pkg_resources import packaging

import fire
import random
import torch
import datasets
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.data.data_collator import default_data_collator

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.configs import dataset_config as DATASET_CONFIG
from llama_recipes.data.concatenator import ConcatDataset
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from llama_recipes.utils import fsdp_auto_wrap_policy
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

def pad_to_batch_max(batch, tokenizer):
    # get max length
    max_length = max([len(feature["input_ids"]) for feature in batch])
    # pad to max length
    for i in range(len(batch)):
        batch[i]["input_ids"] = [tokenizer.pad_token_id] * (
            max_length - len(batch[i]["input_ids"])
        ) + batch[i]["input_ids"]
        batch[i]["attention_mask"] = [0] * (
            max_length - len(batch[i]["attention_mask"])
        ) + batch[i]["attention_mask"]
        batch[i]["labels"] = [-100] * (
            max_length - len(batch[i]["labels"])
        ) + batch[i]["labels"]
    return  default_data_collator(batch)

def main(**kwargs):
    # Update the configuration for the training and sharding process
    train_config, fsdp_config, dataset_config = TRAIN_CONFIG(), FSDP_CONFIG(), DATASET_CONFIG()
    update_config((train_config, fsdp_config, dataset_config), **kwargs)
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    local_rank = 0
    rank = 0
    if train_config.enable_fsdp:
        setup()
        # torchrun specific
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
   # additional_args["low_cpu_mem_usage"] = True
    if train_config.rope_scaling:
        additional_args["rope_scaling"] = json.loads(train_config.rope_scaling)
    if dataset_config.is_mixtral:
        additional_args["use_flash_attention_2"] = True
        additional_args["torch_dtype"] = torch.bfloat16 if fsdp_config.pure_bf16 else torch.float16 if fsdp_config.use_fp16 else torch.float32
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        if rank == 0:
            model = AutoModelForCausalLM.from_pretrained(
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
                **additional_args,
            )
        else:
            llama_config = AutoModelForCausalLM.from_pretrained(train_config.model_name)
            llama_config.use_cache = use_cache
            with torch.device("meta"):
                model = AutoModelForCausalLM(llama_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization or dataset_config.is_mixtral else None,
            use_cache=use_cache,
            **additional_args,
        )
    if train_config.enable_fsdp and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name, use_fast=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if rank == 0:
        print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16 and not dataset_config.is_mixtral:
        model.to(torch.bfloat16)
    elif train_config.enable_fsdp and fsdp_config.use_fp16 and not dataset_config.is_mixtral:
        model.to(torch.float16)

    model.gradient_checkpointing_enable()

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        if rank == 0:
            print(f"--> Training Config: {train_config}")
            print(f"--> FSDP Config: {fsdp_config}")
            print(f"--> Dataset Config: {dataset_config}")
            print(f"--> PEFT Config: {peft_config}")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:
            freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)

        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=True)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp and not dataset_config.is_mixtral:
        model.to("cuda")

    # read datasets
    training_dataset = read_files_to_dataset(dataset_config.training_folder, dataset_config, rank)
    eval_dataset = None
    if dataset_config.eval_folder:
        eval_dataset = read_files_to_dataset(dataset_config.eval_folder, dataset_config, rank)
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
    
    # dataloaders
    train_dl_kwargs = get_dataloader_kwargs(train_config, training_dataset, tokenizer, "train")
    training_dataloader = DataLoader(training_dataset, collate_fn=lambda batch: pad_to_batch_max(batch, tokenizer), pin_memory=True, **train_dl_kwargs)
    eval_dataloder = None
    if eval_dataset:
        val_dl_kwargs = get_dataloader_kwargs(train_config, eval_dataset, tokenizer, "val")
        eval_dataloder = DataLoader(eval_dataset, collate_fn=lambda batch: pad_to_batch_max(batch, tokenizer), pin_memory=True, **val_dl_kwargs)


    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)


    # Start the training process
    results = train(
        model,
        training_dataloader,
        eval_dataloder,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
    )
    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

if __name__ == "__main__":
    fire.Fire(main)
