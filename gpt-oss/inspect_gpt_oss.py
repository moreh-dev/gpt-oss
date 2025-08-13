#!/usr/bin/env python3
"""
Inspect a GPT-OSS model directory and dump a precise inventory:
- Reads config (original/ or root)
- Loads safetensors (original/ single file or root shards)
- Lists all keys with shape & dtype
- Detects likely embeddings, norms, attention proj/bias, sink, MoE tensors by SHAPE
- Lists MXFP4 base keys (have .blocks + .scales)
Outputs: JSON (default: model_inventory.json) and a small text summary.
"""
import argparse
import json
import pathlib
from typing import Any, Dict, List, Tuple

from safetensors.torch import load_file as torch_load_file
import torch


def read_json(p: pathlib.Path) -> dict:
    with p.open("r") as f:
        return json.load(f)


def load_any(root: pathlib.Path):
    """Prefer original/model.safetensors; else root shards; else root single file."""
    orig = root / "original"
    if (orig / "model.safetensors").exists():
        T = torch_load_file(orig / "model.safetensors")
        cfg = read_json(orig /
                        "config.json") if (orig /
                                           "config.json").exists() else {}
        return T, cfg, "original/"
    idx = root / "model.safetensors.index.json"
    if idx.exists():
        m = json.loads(idx.read_text())
        files = sorted(
            {root / shard
             for shard in m.get("weight_map", {}).values()})
        T: Dict[str, torch.Tensor] = {}
        for fp in files:
            T.update(torch_load_file(fp))
        cfg = read_json(root /
                        "config.json") if (root /
                                           "config.json").exists() else {}
        return T, cfg, "sharded/"
    one = root / "model.safetensors"
    if one.exists():
        T = torch_load_file(one)
        cfg = read_json(root /
                        "config.json") if (root /
                                           "config.json").exists() else {}
        return T, cfg, "root/"
    raise FileNotFoundError(f"No safetensors found under {root}")


def cfg_get(cfg: dict, *names, default=None, cast=lambda x: x):
    """Safe get with optional casting; if default is None, return None (don't cast)."""
    for n in names:
        if n in cfg and cfg[n] is not None:
            return cast(cfg[n])
    if default is None:
        return None
    return cast(default)


def tensor_info(t: torch.Tensor) -> dict:
    return {
        "shape": tuple(int(s) for s in t.shape),
        "dtype": str(t.dtype).replace("torch.", "")
    }


def main():
    ap = argparse.ArgumentParser(
        description=
        "Inspect GPT-OSS weights & config and dump an inventory JSON.")
    ap.add_argument("src", help="Path to model dir (e.g. ./gpt-oss-20b)")
    ap.add_argument("--out",
                    default="model_inventory.json",
                    help="Output JSON path")
    ap.add_argument("--txt",
                    default="model_inventory.txt",
                    help="Output text summary path")
    args = ap.parse_args()

    root = pathlib.Path(args.src)
    T, cfg, src = load_any(root)

    # Read config with both schemas (safe casting)
    dim = cfg_get(cfg, "dim", "hidden_size", cast=int)
    hidden_dim = cfg_get(cfg, "hidden_dim", "intermediate_size", cast=int)
    n_layers = cfg_get(cfg, "n_layers", "num_hidden_layers", cast=int)
    n_heads = cfg_get(cfg, "n_heads", "num_attention_heads", cast=int)
    n_kv_heads = cfg_get(cfg, "n_kv_heads", "num_key_value_heads", cast=int)
    n_experts = cfg_get(cfg,
                        "n_experts",
                        "num_local_experts",
                        default=0,
                        cast=int)
    top_k = cfg_get(cfg,
                    "top_k",
                    "experts_per_token",
                    "num_experts_per_tok",
                    default=0,
                    cast=int)
    vocab_size = cfg_get(cfg, "vocab_size", cast=int)
    seq_len = cfg_get(cfg,
                      "seq_len",
                      "max_position_embeddings",
                      "initial_context_length",
                      default=0,
                      cast=int)
    window = cfg_get(cfg, "window", "sliding_window", default=0, cast=int)
    rope_base = cfg_get(cfg,
                        "rope_base",
                        "rope_theta",
                        default=10000.0,
                        cast=float)
    rope_scale = cfg_get(cfg, "rope_scale", default=None, cast=float)
    if rope_scale is None:
        rope_scale = cfg.get("rope_scaling", {}).get("factor", 1.0)
    layer_types = cfg.get("layer_types", [])
    alt_banded = int(cfg.get("alt_banded", 1 if
                             (window or layer_types) else 0))
    kv_dim = None
    if dim and n_heads and n_kv_heads:
        kv_dim = dim * n_kv_heads // n_heads

    # Raw keys inventory
    keys = sorted(T.keys())
    info = {k: tensor_info(T[k]) for k in keys}

    # MXFP4 base detection
    mxfp4_bases: List[str] = []
    for k in keys:
        if k.endswith(".blocks"):
            base = k[:-7]
            if f"{base}.scales" in T:
                mxfp4_bases.append(base)

    # Heuristic detection by exact shapes (if config available)
    detected: dict[str, Any] = {}

    def pick2d(d0, d1, hints: List[str] = None):
        cand = []
        for k in keys:
            s = info[k]["shape"]
            if len(s) == 2 and s == (d0, d1):
                if not hints or any(h in k.lower() for h in hints):
                    cand.append(k)
        return cand

    def pick3d(d0, d1, d2, hints: List[str] = None):
        cand = []
        for k in keys:
            s = info[k]["shape"]
            if len(s) == 3 and s == (d0, d1, d2):
                if not hints or any(h in k.lower() for h in hints):
                    cand.append(k)
        return cand

    def pick4d(d0, d1, d2, d3, hints: List[str] = None):
        cand = []
        for k in keys:
            s = info[k]["shape"]
            if len(s) == 4 and s == (d0, d1, d2, d3):
                if not hints or any(h in k.lower() for h in hints):
                    cand.append(k)
        return cand

    if vocab_size and dim:
        detected["embeddings_2d_vocab_dim"] = pick2d(
            vocab_size, dim, hints=["embed", "tok", "wte", "embedding"])

    if n_layers and dim:
        detected["rms_att_w_L_dim"] = pick2d(n_layers,
                                             dim,
                                             hints=["rms", "att"])
        detected["rms_ffn_w_L_dim"] = pick2d(n_layers,
                                             dim,
                                             hints=["rms", "ffn", "mlp"])

    if n_layers and dim and kv_dim:
        detected["wq_L_dim_dim"] = pick3d(n_layers,
                                          dim,
                                          dim,
                                          hints=["wq", "query", "att"])
        detected["wk_L_dim_kv"] = pick3d(n_layers,
                                         dim,
                                         kv_dim,
                                         hints=["wk", "key", "att"])
        detected["wv_L_dim_kv"] = pick3d(n_layers,
                                         dim,
                                         kv_dim,
                                         hints=["wv", "value", "att"])
        detected["wo_L_dim_dim"] = pick3d(n_layers,
                                          dim,
                                          dim,
                                          hints=["wo", "proj", "att"])

        detected["bq_L_dim"] = pick2d(
            n_layers, dim, hints=["wq", "query", "bias"]) or pick2d(
                n_layers, dim, hints=["bq"])
        detected["bk_L_kv"] = pick2d(
            n_layers, kv_dim, hints=["wk", "key", "bias"]) or pick2d(
                n_layers, kv_dim, hints=["bk"])
        detected["bv_L_kv"] = pick2d(
            n_layers, kv_dim, hints=["wv", "value", "bias"]) or pick2d(
                n_layers, kv_dim, hints=["bv"])
        detected["bo_L_dim"] = pick2d(
            n_layers, dim, hints=["wo", "proj", "bias"]) or pick2d(
                n_layers, dim, hints=["bo"])

    if n_layers and n_heads:
        detected["attn_sink_L_heads"] = pick2d(n_layers,
                                               n_heads,
                                               hints=["sink", "att"])

    if dim:
        detected["rms_final_dim"] = [
            k for k in keys if info[k]["shape"] == (dim, ) and any(
                h in k.lower() for h in ["rms", "norm", "final"])
        ]

    if n_experts and n_layers and hidden_dim and dim:
        # router
        detected["wr_L_E_dim"] = pick3d(n_layers,
                                        n_experts,
                                        dim,
                                        hints=["router", "wr"])
        detected["br_L_E"] = pick2d(n_layers,
                                    n_experts,
                                    hints=["router", "bias", "br"])
        # MoE weights (MXFP4 likely)
        detected["w_up_L_E_hidden_dim"] = pick4d(n_layers,
                                                 n_experts,
                                                 hidden_dim,
                                                 dim,
                                                 hints=["w_up", "up", "fc1"])
        detected["w_gate_L_E_hidden_dim"] = pick4d(n_layers,
                                                   n_experts,
                                                   hidden_dim,
                                                   dim,
                                                   hints=["w_gate", "gate"])
        detected["w_down_L_E_dim_hidden"] = pick4d(
            n_layers,
            n_experts,
            dim,
            hidden_dim,
            hints=["w_down", "down", "fc2"])
        detected["b_up_L_E_hidden"] = pick3d(n_layers,
                                             n_experts,
                                             hidden_dim,
                                             hints=["b_up", "up", "bias"])
        detected["b_gate_L_E_hidden"] = pick3d(
            n_layers, n_experts, hidden_dim, hints=["b_gate", "gate", "bias"])
        detected["b_down_L_E_dim"] = pick3d(n_layers,
                                            n_experts,
                                            dim,
                                            hints=["b_down", "down", "bias"])

    # Summaries
    summary_txt = []
    summary_txt.append(f"Source: {src}")
    summary_txt.append("Config (mixed schema):")
    summary_txt.append(
        f"  dim={dim}, hidden_dim={hidden_dim}, n_layers={n_layers}, n_heads={n_heads}, n_kv_heads={n_kv_heads}"
    )
    summary_txt.append(
        f"  n_experts={n_experts}, top_k={top_k}, vocab={vocab_size}, seq_len={seq_len}, window={window}"
    )
    summary_txt.append(
        f"  rope_base={rope_base}, rope_scale={rope_scale}, alt_banded={alt_banded}"
    )
    if dim and n_heads and n_kv_heads:
        summary_txt.append(f"  kv_dim = {kv_dim}")

    summary_txt.append(f"\nTotal tensors: {len(keys)}")
    summary_txt.append("First 40 keys:")
    for k in keys[:40]:
        summary_txt.append(f"  {k:60s} {info[k]['shape']} {info[k]['dtype']}")

    summary_txt.append("\nMXFP4 base keys (have .blocks + .scales):")
    for b in mxfp4_bases[:80]:
        summary_txt.append(f"  {b}")

    summary_txt.append("\nDetected candidates by shape:")
    for name, cands in detected.items():
        summary_txt.append(f"  {name}:")
        for c in cands[:20]:
            summary_txt.append(f"    - {c}")

    # Write files
    out_json = {
        "source": src,
        "config": {
            "dim": dim,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "n_experts": n_experts,
            "top_k": top_k,
            "vocab_size": vocab_size,
            "seq_len": seq_len,
            "window": window,
            "rope_base": rope_base,
            "rope_scale": rope_scale,
            "alt_banded": alt_banded,
            "kv_dim": kv_dim,
        },
        "mxfp4_bases": mxfp4_bases,
        "all_tensors": info,  # {key: {shape, dtype}}
        "detected_by_shape": detected
    }
    pathlib.Path(args.out).write_text(json.dumps(out_json, indent=2))
    pathlib.Path(args.txt).write_text("\n".join(summary_txt))
    print(f"[OK] wrote {args.out} and {args.txt}")


if __name__ == "__main__":
    main()
