# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass, field
from typing import List

@dataclass
class pept_config:
     task_type: str="CAUSAL_LM"
     prompt_tuning_init: str="TEXT"
     prompt_tuning_init_text: str=None
     num_virtual_tokens: int=20
     tokenizer_name_or_path: str=None

@dataclass
class v1_config:
     task_type: str="CAUSAL_LM"
     encoder_reparameterization_type: str="LSTM"
     encoder_hidden_size: int=750
     encoder_num_layers: int=3
     encoder_dropout: float=0.1
     num_virtual_tokens: int=20

@dataclass
class lora_config:
     r: int=8
     lora_alpha: int=16
     target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
     bias= "none"
     task_type: str= "CAUSAL_LM"
     lora_dropout: float=0.05
     inference_mode: bool = False

@dataclass
class llama_adapter_config:
     adapter_len: int= 10
     adapter_layers: int= 30
     task_type: str= "CAUSAL_LM"

@dataclass
class prefix_config:
     num_virtual_tokens: int=30
     task_type: str= "CAUSAL_LM"    

