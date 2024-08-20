# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from llama_recipes.utils.config_utils import update_config
from llama_recipes.configs import quantization_config  as QUANT_CONFIG
from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig
from warnings import warn

# Function to load the main model for text generation
def load_model(model_name, quantization, use_fast_kernels, **kwargs):
    if type(quantization) == type(True):
            warn("Quantization (--quantization) is a boolean, please specify quantization as '4bit' or '8bit'. Defaulting to '8bit' but this might change in the future.", FutureWarning)
            quantization = "8bit"

    bnb_config = None
    if quantization:
        quant_config = QUANT_CONFIG()
        update_config(quant_config, **kwargs)
        bnb_config = quant_config.create_bnb_config(quantization)

    print(f"use_fast_kernels{use_fast_kernels}")

    kwargs = {}
    if bnb_config:
        kwargs["quantization_config"]=bnb_config
    kwargs["device_map"]="auto"
    kwargs["low_cpu_mem_usage"]=True
    kwargs["attn_implementation"]="sdpa" if use_fast_kernels else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        **kwargs,
    )
    return model


# Function to load the PeftModel for performance optimization
def load_peft_model(model, peft_model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model

# Loading the model from config to load FSDP checkpoints into that
def load_llama_from_config(config_path):
    model_config = LlamaConfig.from_pretrained(config_path) 
    model = LlamaForCausalLM(config=model_config)
    return model
    
    