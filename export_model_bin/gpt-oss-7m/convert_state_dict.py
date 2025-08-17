"""
convert_state_dict.py
---------------------
Convert Hugging Face GPT-OSS 7M safetensors checkpoint into a format compatible
with the GPT-OSS C++ runtime. This script renames tensor keys, fuses QKV, reshapes
MLP weights, and adds an unembedding layer.

Usage
-----
python convert_state_dict.py \
  --input  /path/to/original/model.safetensors \
  --output /path/to/converted/model.safetensors
"""

import argparse
from collections import OrderedDict

from safetensors.torch import load_file
from safetensors.torch import safe_open
from safetensors.torch import save_file
import torch


def convert_keys(state_dict_b):
    new_state_dict = OrderedDict()
    for k, v in state_dict_b.items():
        new_k = k
        # Attention
        new_k = new_k.replace(".input_layernorm.weight", ".attn.norm.scale")
        new_k = new_k.replace("model.layers", "block")
        new_k = new_k.replace(".self_attn", ".attn")
        new_k = new_k.replace(".o_proj", ".out")
        # MLP
        new_k = new_k.replace(".post_attention_layernorm.weight",
                              ".mlp.norm.scale")
        new_k = new_k.replace(".experts.gate_up_proj_bias", ".mlp1_bias")
        new_k = new_k.replace(".experts.gate_up_proj", ".mlp1_weight")
        new_k = new_k.replace(".experts.down_proj_bias", ".mlp2_bias")
        new_k = new_k.replace(".experts.down_proj", ".mlp2_weight")
        new_k = new_k.replace(".router", ".gate")
        # Other
        new_k = new_k.replace("model.norm.weight", "norm.scale")
        new_k = new_k.replace("model.embed_tokens", "embedding")

        new_state_dict[new_k] = v
    return new_state_dict


def concat_qkv(state_dict, block_id):
    q_w = state_dict[f"block.{block_id}.attn.q_proj.weight"]
    k_w = state_dict[f"block.{block_id}.attn.k_proj.weight"]
    v_w = state_dict[f"block.{block_id}.attn.v_proj.weight"]
    qkv_w = torch.cat([q_w, k_w, v_w], dim=0)

    q_b = state_dict[f"block.{block_id}.attn.q_proj.bias"]
    k_b = state_dict[f"block.{block_id}.attn.k_proj.bias"]
    v_b = state_dict[f"block.{block_id}.attn.v_proj.bias"]
    qkv_b = torch.cat([q_b, k_b, v_b], dim=0)

    new_state_dict = OrderedDict(state_dict)
    new_state_dict[f"block.{block_id}.attn.qkv.weight"] = qkv_w
    new_state_dict[f"block.{block_id}.attn.qkv.bias"] = qkv_b

    # Remove originals
    for proj in ["q_proj", "k_proj", "v_proj"]:
        del new_state_dict[f"block.{block_id}.attn.{proj}.weight"]
        del new_state_dict[f"block.{block_id}.attn.{proj}.bias"]

    return new_state_dict


def reshape_mlp(state_dict, block_id):
    mlp1_weight = state_dict[f"block.{block_id}.mlp.mlp1_weight"]
    mlp2_weight = state_dict[f"block.{block_id}.mlp.mlp2_weight"]

    new_state_dict = OrderedDict(state_dict)
    new_state_dict[
        f"block.{block_id}.mlp.mlp1_weight"] = mlp1_weight.transpose(
            1, 2).contiguous()
    new_state_dict[
        f"block.{block_id}.mlp.mlp2_weight"] = mlp2_weight.transpose(
            1, 2).contiguous()

    return new_state_dict


def add_unembedding(state_dict):
    embedding_weight = state_dict["embedding.weight"]
    unembedding_weight = embedding_weight.clone()
    new_state_dict = OrderedDict(state_dict)
    new_state_dict["unembedding.weight"] = unembedding_weight
    return new_state_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert GPT-OSS 7M state_dict for C++ runtime.")
    parser.add_argument("--input",
                        required=True,
                        help="Path to input safetensors checkpoint")
    parser.add_argument("--output",
                        required=True,
                        help="Path to save converted safetensors")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Loading input checkpoint: {args.input}")
    with safe_open(args.input, framework="pt", device="cpu") as f:
        print("Keys in input:")
        for key in f.keys():
            print(f"  {key} {tuple(f.get_tensor(key).shape)}")

    ckpt = load_file(args.input)
    ckpt = convert_keys(ckpt)

    # Hard-coded for 2 blocks in 7M — adjust if needed
    for bid in [0, 1]:
        ckpt = concat_qkv(ckpt, bid)
        ckpt = reshape_mlp(ckpt, bid)

    ckpt = add_unembedding(ckpt)

    print(f"Saving converted checkpoint: {args.output}")
    save_file(ckpt, args.output)

    with safe_open(args.output, framework="pt", device="cpu") as f:
        print("Keys in converted checkpoint:")
        for key in f.keys():
            print(f"  {key} {tuple(f.get_tensor(key).shape)}")
