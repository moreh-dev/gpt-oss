"""
generate_bin.py
---------------
Wraps a converted checkpoint (safetensors) and its config.json
into a single `.bin` file compatible with the GPT-OSS C++ runtime.

Usage
-----
python generate_bin.py \
  --config config.json \
  --input  model.safetensors \
  --output gpt-oss-7M.bin
"""

import argparse
from collections import OrderedDict
import json
import re
import struct

import numpy as np
from safetensors.torch import load_file
import torch

# Explicit list to ensure correct keys order
KEYS = [
    "vocab_size",
    "hidden_size",
    "num_experts",
    "experts_per_token",
    "intermediate_size",
    "num_hidden_layers",
    "head_dim",
    "num_attention_heads",
    "num_key_value_heads",
    "max_seq_len",
    # "initial_context_length",
    # "rope_theta",
    # "sliding_window",
    # "swiglu_limit"
]

CATEGORIES = [
    "embedding.weight",
    "unembedding.weight",
    "attn.norm.scale",
    "mlp.norm.scale",
    "norm.scale",
    # Attention
    "attn.qkv.weight",
    "attn.qkv.bias",
    "attn.out.weight",
    "attn.out.bias",
    "attn.sinks",
    # MoE
    "mlp.gate.weight",
    "mlp.gate.bias",
    "mlp.mlp1_weight",
    "mlp.mlp1_bias",
    "mlp.mlp2_weight",
    "mlp.mlp2_bias",
]


def binarize_config(cfg):
    config_bin = b""
    for key in KEYS:
        val = cfg[key]
        if isinstance(val, int):
            config_bin += struct.pack("<i", val)
        elif isinstance(val, float):
            config_bin += struct.pack("<f", val)
        else:
            raise TypeError(f"Unsupported type for key {key}: {type(val)}")
    return config_bin


def binarize_weights(state_dict, dtype="float32"):
    # reorder weights such that it is compatible with llama2.c loading
    buckets = {cat: [] for cat in CATEGORIES}
    others = []
    for key in state_dict:
        matched = False
        for cat in CATEGORIES:
            if key.endswith(cat):
                match = re.search(r"block\.(\d+)\.", key)
                block_idx = int(match.group(1)) if match else -1
                buckets[cat].append((block_idx, key))
                matched = True
                break
        if not matched:
            others.append(key)

    reordered = OrderedDict()
    for cat in CATEGORIES:
        for _, key in buckets[cat]:
            reordered[key] = state_dict[key]
    for key in sorted(others):
        reordered[key] = state_dict[key]

    # convert to binary
    np_dtype = np.float32 if dtype == "float32" else np.bfloat16
    torch_dtype = torch.float32 if dtype == "float32" else torch.bfloat16
    weights_bin = b""
    for name, tensor in reordered.items():
        np_tensor = tensor.detach().cpu().to(torch_dtype).numpy().astype(
            np_dtype)
        weights_bin += np_tensor.tobytes()

    return weights_bin


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate .bin from config + safetensors.")
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--input",
                        required=True,
                        help="Path to converted model.safetensors")
    parser.add_argument("--output",
                        required=True,
                        help="Path to write .bin file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Loading config: {args.config}")
    with open(args.config, "r") as f:
        config = json.load(f)

    config_bin = binarize_config(config)

    print(f"Loading weights: {args.input}")
    state_dict = load_file(args.input)
    weights_bin = binarize_weights(state_dict)

    print(f"Writing output binary: {args.output}")
    with open(args.output, "wb") as f:
        f.write(config_bin + weights_bin)

    print("Done.")
