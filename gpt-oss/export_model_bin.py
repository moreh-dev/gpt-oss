# python export_model_bin.py ./gpt-oss-20b ./gpt-oss-20b.bin

import json
import pathlib
import struct
import sys
from typing import Dict, Optional, Tuple

import numpy as np
from safetensors.torch import load_file as torch_load_file
import torch

# =============== I/O ===============


def read_json(p: pathlib.Path) -> dict:
    with open(p, "r") as f:
        return json.load(f)


def load_tensors(root: pathlib.Path) -> Dict[str, torch.Tensor]:
    """Load safetensors dict (BF16 friendly). Supports HF shards or original/."""
    idx = root / "model.safetensors.index.json"
    if idx.exists():
        m = json.loads(idx.read_text())
        weight_map = m.get("weight_map") or {}
        T: Dict[str, torch.Tensor] = {}
        for shard in sorted(set(weight_map.values())):
            T.update(torch_load_file(root / shard))
        return T
    one = root / "model.safetensors"
    if one.exists():
        return torch_load_file(one)
    raise FileNotFoundError(f"No safetensors found in {root}")


# =============== utils ===============


def to_np32(t: torch.Tensor) -> np.ndarray:
    return t.to(torch.float32).cpu().numpy()


def write_mat(f, arr: np.ndarray):
    f.write(arr.astype(np.float32).ravel(order="C").tobytes())


def find_by_names(T: Dict[str, torch.Tensor], *names) -> Optional[str]:
    for n in names:
        if n in T: return n
    return None


def find_by_shape(
    T: Dict[str, torch.Tensor],
    shape,
    prefer=("embed", "tok", "wte", "attn", "mlp", "weight")
) -> Optional[str]:
    shape = tuple(shape)
    for k in [k for k in T.keys() if any(s in k for s in prefer)]:
        if tuple(T[k].shape) == shape: return k
    for k, v in T.items():
        if tuple(v.shape) == shape: return k
    return None


def fetch(T: Dict[str, torch.Tensor],
          *candidates,
          required=True,
          expect_shape=None,
          hints=None) -> Optional[np.ndarray]:
    k = find_by_names(T, *candidates)
    if k is None and expect_shape is not None:
        k = find_by_shape(T, expect_shape, prefer=tuple(hints or ()))
    if k is None:
        if required:
            sample = list(list(T.keys())[:60])
            raise KeyError(
                f"Missing any of {candidates} shape={expect_shape}; sample keys: {sample}"
            )
        return None
    return to_np32(T[k])


def nz(arr: Optional[np.ndarray], shape: Tuple[int, ...]) -> np.ndarray:
    return arr if arr is not None else np.zeros(shape, dtype=np.float32)


# =============== MXFP4 decode (experts) ===============


def decode_mxfp4(
        T: Dict[str, torch.Tensor],
        blocks_key: str,
        scales_key: str,
        decoded_last_dim: Optional[int] = None) -> Optional[np.ndarray]:
    """
    Decode per-block int4 with symmetric quantization: real = scale * (code - 8).
    Supports two storage layouts:

      A) blocks[..., Nb]                     scales[..., Nblk]
         -> decoded last dim D = Nb*2; block_size = D / Nblk (must be int)

      B) blocks[..., Nblk, packed_bytes]     scales[..., Nblk]
         -> block_size = packed_bytes*2; decoded last dim D = Nblk * block_size

    Returns float32 ndarray or None if keys not found.
    """
    bt = T.get(blocks_key, None)
    st = T.get(scales_key, None)
    if bt is None or st is None:
        return None

    blk = bt.cpu().numpy().astype(np.uint8)
    scl = st.cpu().numpy().astype(np.float32)

    # -------- Layout B: explicit Nblk axis --------
    if blk.ndim == scl.ndim + 1 and blk.shape[-2] == scl.shape[-1]:
        Nblk = blk.shape[-2]
        packed_bytes = blk.shape[-1]
        block_size = packed_bytes * 2  # 2 fp4 codes per byte
        D = Nblk * block_size  # decoded last dim

        # Optional validation against caller's expectation
        if decoded_last_dim is not None and D != decoded_last_dim:
            raise ValueError(
                f"{blocks_key}: decoded width {D} != expected {decoded_last_dim} "
                f"(Nblk={Nblk}, packed_bytes={packed_bytes}, block_size={block_size})"
            )

        flat_blk = blk.reshape(-1, Nblk, packed_bytes)  # [B, Nblk, packed]
        lo = (flat_blk & 0x0F).astype(np.int8) - 8
        hi = ((flat_blk >> 4) & 0x0F).astype(np.int8) - 8

        vals = np.empty((flat_blk.shape[0], Nblk, block_size), dtype=np.int8)
        vals[:, :, 0::2] = lo
        vals[:, :, 1::2] = hi

        flat_scl = scl.reshape(-1, Nblk, 1).astype(np.float32)
        out = (vals.astype(np.float32) * flat_scl)  # [B, Nblk, block_size]
        out = out.reshape(blk.shape[:-2] + (D, ))  # fold (Nblk,block_size)->D
        return out

    # -------- Layout A: packed-only last axis --------
    if blk.ndim == scl.ndim:
        Nb = blk.shape[-1]
        D = Nb * 2
        Nblk = scl.shape[-1]
        if D % Nblk != 0:
            raise ValueError(
                f"{blocks_key}: decoded {D} not divisible by Nblk {Nblk} "
                f"(Nb={Nb}, scl_last={Nblk})")
        block_size = D // Nblk

        if decoded_last_dim is not None and D != decoded_last_dim:
            raise ValueError(
                f"{blocks_key}: decoded width {D} != expected {decoded_last_dim} "
                f"(Nb={Nb}, block_size={block_size}, Nblk={Nblk})")

        flat_blk = blk.reshape(-1, Nb)  # [B, Nb]
        lo = (flat_blk & 0x0F).astype(np.int8) - 8
        hi = ((flat_blk >> 4) & 0x0F).astype(np.int8) - 8

        vals = np.empty((flat_blk.shape[0], D), dtype=np.int8)
        vals[:, 0::2] = lo
        vals[:, 1::2] = hi

        flat_scl = scl.reshape(-1, Nblk).astype(np.float32).repeat(
            block_size, axis=1)  # [B, D]
        out = (vals.astype(np.float32) * flat_scl).reshape(blk.shape[:-1] +
                                                           (D, ))
        return out

    # -------- Unknown layout --------
    raise ValueError(
        f"Unexpected MXFP4 layout for {blocks_key}: blocks {blk.shape} vs scales {scl.shape}"
    )


# =============== Attention helpers ===============


def infer_kv_from_fused_qkv_strict(T, i, dim, nH):
    key = f"block.{i}.attn.qkv.weight"
    if key not in T:  # sharded HF path may not have fused qkv
        return None
    out_rows = int(T[key].shape[0])
    if out_rows <= dim or (out_rows - dim) % 2 != 0:
        raise ValueError(
            f"layer {i}: unexpected fused qkv rows={out_rows}, dim={dim}")
    kv_dim_fused = (out_rows - dim) // 2
    head_dim = dim // nH
    if kv_dim_fused % head_dim != 0:
        raise ValueError(
            f"fused kv_dim={kv_dim_fused} not multiple of head_dim={head_dim} "
            f"(rows={out_rows}, dim={dim}, n_heads={nH})")
    nKV_infer = kv_dim_fused // head_dim
    return kv_dim_fused, nKV_infer


def split_fused_qkv(fused: np.ndarray, dim: int, kv_dim: int, i: int):
    out = fused.shape[0]
    if out != dim + 2 * kv_dim:
        raise ValueError(
            f"layer {i}: fused qkv rows {out} != {dim + 2*kv_dim}")
    q = fused[0:dim, :]
    k = fused[dim:dim + kv_dim, :]
    v = fused[dim + kv_dim:dim + 2 * kv_dim, :]
    return q, k, v


def fetch_qkv(T: Dict[str, torch.Tensor], i: int, dim: int, kv_dim: int):
    # Try separate q/k/v (HF style)
    wq = fetch(T,
               f"model.layers.{i}.self_attn.q_proj.weight",
               f"model.layers.{i}.attn.q_proj.weight",
               required=False,
               expect_shape=(dim, dim),
               hints=("q", "attn"))
    wk = fetch(T,
               f"model.layers.{i}.self_attn.k_proj.weight",
               f"model.layers.{i}.attn.k_proj.weight",
               required=False,
               expect_shape=(kv_dim, dim),
               hints=("k", "attn"))
    wv = fetch(T,
               f"model.layers.{i}.self_attn.v_proj.weight",
               f"model.layers.{i}.attn.v_proj.weight",
               required=False,
               expect_shape=(kv_dim, dim),
               hints=("v", "attn"))
    if wq is not None and wk is not None and wv is not None:
        return wq, wk, wv
    # Fused (original)
    fused = fetch(T,
                  f"block.{i}.attn.qkv.weight",
                  required=True,
                  hints=("qkv", "attn"))
    return split_fused_qkv(np.asarray(fused, np.float32), dim, kv_dim, i)


def fetch_qkv_bias(T: Dict[str, torch.Tensor], i: int, dim: int, kv_dim: int):
    bq = fetch(T,
               f"model.layers.{i}.self_attn.q_proj.bias",
               f"model.layers.{i}.attn.q_proj.bias",
               required=False,
               expect_shape=(dim, ),
               hints=("q", "attn"))
    bk = fetch(T,
               f"model.layers.{i}.self_attn.k_proj.bias",
               f"model.layers.{i}.attn.k_proj.bias",
               required=False,
               expect_shape=(kv_dim, ),
               hints=("k", "attn"))
    bv = fetch(T,
               f"model.layers.{i}.self_attn.v_proj.bias",
               f"model.layers.{i}.attn.v_proj.bias",
               required=False,
               expect_shape=(kv_dim, ),
               hints=("v", "attn"))
    if bq is not None and bk is not None and bv is not None:
        return bq, bk, bv
    fused = fetch(T, f"block.{i}.attn.qkv.bias", required=False)
    if fused is None:
        return (np.zeros((dim, ), np.float32), np.zeros(
            (kv_dim, ), np.float32), np.zeros((kv_dim, ), np.float32))
    fused = np.asarray(fused, np.float32)
    out = fused.shape[0]
    if out == dim + 2 * kv_dim:
        return fused[:dim], fused[dim:dim + kv_dim], fused[dim + kv_dim:]
    if out % 3 == 0:
        each = out // 3
        return fused[:each][:dim], fused[each:2 *
                                         each][:kv_dim], fused[2 * each:3 *
                                                               each][:kv_dim]
    return (np.zeros((dim, ), np.float32), np.zeros(
        (kv_dim, ), np.float32), np.zeros((kv_dim, ), np.float32))


# =============== MoE helpers ===============


def moe_available(T: Dict[str, torch.Tensor]) -> bool:
    # Original 20B exposes expert weights as MXFP4 blocks/scales:
    return ("block.0.mlp.mlp1_weight.blocks"
            in T) and ("block.0.mlp.mlp2_weight.blocks" in T)


def decode_mxfp4_arrays(blocks: np.ndarray, scales: np.ndarray,
                        decoded_last_dim: int) -> np.ndarray:
    """
    Same decode as decode_mxfp4(), but takes already-sliced NumPy arrays.
    Supports:
      A) blocks[..., Nb], scales[..., Nblk]
      B) blocks[..., Nblk, packed_bytes], scales[..., Nblk]
    Returns float32.
    """
    blk = blocks.astype(np.uint8)
    scl = scales.astype(np.float32)

    # Layout B
    if blk.ndim == scl.ndim + 1 and blk.shape[-2] == scl.shape[-1]:
        Nblk = blk.shape[-2]
        packed_bytes = blk.shape[-1]
        block_size = packed_bytes * 2
        D = Nblk * block_size
        if D != decoded_last_dim:
            raise ValueError(
                f"decoded width {D} != expected {decoded_last_dim}")
        flat_blk = blk.reshape(-1, Nblk, packed_bytes)
        lo = (flat_blk & 0x0F).astype(np.int8) - 8
        hi = ((flat_blk >> 4) & 0x0F).astype(np.int8) - 8
        vals = np.empty((flat_blk.shape[0], Nblk, block_size), dtype=np.int8)
        vals[:, :, 0::2] = lo
        vals[:, :, 1::2] = hi
        flat_scl = scl.reshape(-1, Nblk, 1)
        out = (vals.astype(np.float32) * flat_scl).reshape(blk.shape[:-2] +
                                                           (D, ))
        return out

    # Layout A
    if blk.ndim == scl.ndim:
        Nb = blk.shape[-1]
        D = Nb * 2
        Nblk = scl.shape[-1]
        if D % Nblk != 0:
            raise ValueError(f"decoded {D} not divisible by Nblk {Nblk}")
        if D != decoded_last_dim:
            raise ValueError(
                f"decoded width {D} != expected {decoded_last_dim}")
        block_size = D // Nblk
        flat_blk = blk.reshape(-1, Nb)
        lo = (flat_blk & 0x0F).astype(np.int8) - 8
        hi = ((flat_blk >> 4) & 0x0F).astype(np.int8) - 8
        vals = np.empty((flat_blk.shape[0], D), dtype=np.int8)
        vals[:, 0::2] = lo
        vals[:, 1::2] = hi
        flat_scl = scl.reshape(-1, Nblk).repeat(block_size, axis=1)
        out = (vals.astype(np.float32) * flat_scl).reshape(blk.shape[:-1] +
                                                           (D, ))
        return out

    raise ValueError(
        f"Unexpected MXFP4 layout: blocks {blk.shape} vs scales {scl.shape}")


def write_moe_from_mxfp4(f, T: Dict[str, torch.Tensor], nL: int, dim: int,
                         hidden: int, nExp: int):
    """
    Streamed writer: eight passes in the correct binary order.
      1) WR [L,E,dim]
      2) BR [L,E]
      3) WUP   [L,E,hidden,dim]      from mlp1 blocks/scales (first half)
      4) WGATE [L,E,hidden,dim]      from mlp1 blocks/scales (second half)
      5) WDOWN [L,E,dim,hidden]      from mlp2 blocks/scales
      6) BUP   [L,E,hidden]          from mlp1_bias (first half)
      7) BGATE [L,E,hidden]          from mlp1_bias (second half)
      8) BDOWN [L,E,dim]             from mlp2_bias
    """
    # --- 1) WR
    for i in range(nL):
        wr = fetch(T,
                   f"block.{i}.mlp.gate.weight",
                   required=True,
                   expect_shape=(nExp, dim),
                   hints=("gate", "router"))
        write_mat(f, wr)
        del wr

    # --- 2) BR
    for i in range(nL):
        br = fetch(T,
                   f"block.{i}.mlp.gate.bias",
                   required=True,
                   expect_shape=(nExp, ),
                   hints=("gate", "router"))
        write_mat(f, br)
        del br

    # helper to grab torch tensors once per layer
    def get_mlp1_tensors(i):
        blk_t = T[f"block.{i}.mlp.mlp1_weight.blocks"]
        scl_t = T[f"block.{i}.mlp.mlp1_weight.scales"]
        bias = fetch(T,
                     f"block.{i}.mlp.mlp1_bias",
                     required=True,
                     expect_shape=(nExp, 2 * hidden))
        return blk_t, scl_t, bias

    def get_mlp2_tensors(i):
        blk_t = T[f"block.{i}.mlp.mlp2_weight.blocks"]
        scl_t = T[f"block.{i}.mlp.mlp2_weight.scales"]
        bias = fetch(T,
                     f"block.{i}.mlp.mlp2_bias",
                     required=True,
                     expect_shape=(nExp, dim))
        return blk_t, scl_t, bias

    # --- 3) WUP: decode per expert slice
    for i in range(nL):
        print(f"[INFO] WUP L{i}")
        blk_t, scl_t, _ = get_mlp1_tensors(i)
        # slice each expert: [1, hidden, ...]  (first half of 2*hidden)
        for e in range(nExp):
            blk_np = blk_t[e:e + 1, :hidden, ...].cpu().numpy()
            scl_np = scl_t[e:e + 1, :hidden, ...].cpu().numpy()
            w_up_e = decode_mxfp4_arrays(
                blk_np, scl_np, decoded_last_dim=dim)  # [1, hidden, dim]
            write_mat(f, w_up_e[0])
            del blk_np, scl_np, w_up_e

    # --- 4) WGATE
    for i in range(nL):
        print(f"[INFO] WGATE L{i}")
        blk_t, scl_t, _ = get_mlp1_tensors(i)
        for e in range(nExp):
            blk_np = blk_t[e:e + 1, hidden:2 * hidden, ...].cpu().numpy()
            scl_np = scl_t[e:e + 1, hidden:2 * hidden, ...].cpu().numpy()
            w_gate_e = decode_mxfp4_arrays(
                blk_np, scl_np, decoded_last_dim=dim)  # [1, hidden, dim]
            write_mat(f, w_gate_e[0])
            del blk_np, scl_np, w_gate_e

    # --- 5) WDOWN
    for i in range(nL):
        print(f"[INFO] WDOWN L{i}")
        blk_t, scl_t, _ = get_mlp2_tensors(i)
        for e in range(nExp):
            blk_np = blk_t[e:e + 1, ...].cpu().numpy(
            )  # [1, dim, Nblk, packed] or [1, Nblk, packed] depending on layout
            scl_np = scl_t[e:e + 1, ...].cpu().numpy()
            w_down_e = decode_mxfp4_arrays(
                blk_np, scl_np, decoded_last_dim=hidden)  # [1, dim, hidden]
            write_mat(f, w_down_e[0])
            del blk_np, scl_np, w_down_e

    # --- 6) BUP
    for i in range(nL):
        _, _, gup_b = get_mlp1_tensors(i)
        b_up = gup_b[:, :hidden]  # [E, hidden]
        write_mat(f, b_up)
        del gup_b

    # --- 7) BGATE
    for i in range(nL):
        _, _, gup_b = get_mlp1_tensors(i)
        b_gate = gup_b[:, hidden:]  # [E, hidden]
        write_mat(f, b_gate)
        del gup_b

    # --- 8) BDOWN
    for i in range(nL):
        _, _, dwn_b = get_mlp2_tensors(i)  # [E, dim]
        write_mat(f, dwn_b)
        del dwn_b


# =============== main ===============


def main():
    if len(sys.argv) != 3:
        print("Usage: python export_model_bin.py <model_dir> <out.bin>",
              file=sys.stderr)
        sys.exit(2)

    # --- infer heads/kv_dim from tensors, override header if needed ---

    def pick_heads_from_k_shape(dim: int, kv_dim: int, nH_cfg: int,
                                nKV_cfg: int):
        """
        Try to infer (n_heads, n_kv_heads, head_dim) from kv_dim and dim.
        Prefer the provided config when it's consistent; otherwise search
        common n_kv_heads values.
        """
        # 1) If config is consistent, keep it
        if nKV_cfg > 0 and kv_dim % nKV_cfg == 0:
            head_dim = kv_dim // nKV_cfg
            if head_dim > 0 and dim % head_dim == 0:
                nH = dim // head_dim
                return nH, nKV_cfg, head_dim

        # 2) Otherwise, search typical KV head counts
        for nKV_try in (32, 16, 12, 10, 8, 6, 4, 2, 1):
            if kv_dim % nKV_try != 0:
                continue
            head_dim = kv_dim // nKV_try
            if head_dim > 0 and dim % head_dim == 0:
                nH = dim // head_dim
                return nH, nKV_try, head_dim

        raise ValueError(
            f"Cannot find consistent (n_heads, n_kv_heads) for dim={dim}, kv_dim={kv_dim}"
        )

    # ---- after reading cfg and tensors ----
    root = pathlib.Path(sys.argv[1])
    outp = sys.argv[2]

    print(f"[INFO] Loading config from {root / 'config.json'}")
    cfg = read_json(root / "config.json")
    print(f"[INFO] Loading tensors from {root}")
    T = load_tensors(root)

    dim = int(cfg["hidden_size"])
    nL = int(cfg["num_hidden_layers"])
    nH = int(cfg["num_attention_heads"])
    nKV = int(cfg.get("num_key_value_heads", cfg.get("n_kv_heads", 0)))
    vsz = int(cfg["vocab_size"])
    hidden = int(cfg["intermediate_size"])

    # context/rope
    seql = int(
        cfg.get(
            "max_position_embeddings",
            cfg.get(
                "initial_context_length",
                cfg.get("rope_scaling",
                        {}).get("original_max_position_embeddings", 0))))
    win = int(cfg.get("sliding_window", 0))
    alt = 1
    ropeB = float(
        cfg.get("rope_theta",
                cfg.get("rope_scaling", {}).get("base", 10000.0)))
    ropeS = float(
        cfg.get("rope_scaling_factor",
                cfg.get("rope_scaling", {}).get("factor", 1.0)))

    # experts
    nExp_cfg = int(cfg.get("num_local_experts", cfg.get("num_experts", 0)))
    topk = int(cfg.get("num_experts_per_tok", cfg.get("experts_per_token", 0)))

    # --- infer/override n_heads, n_kv_heads, head_dim, kv_dim ---
    # Try fused qkv (original/)
    res = infer_kv_from_fused_qkv_strict(T, 0, dim, nH)
    if res is not None:
        kv_dim_fused, nKV_infer = res
        head_dim = kv_dim_fused // nKV_infer
        nH_new = dim // head_dim
        if nH_new != nH or nKV_infer != nKV:
            print(
                f"[INFO] overriding heads to match fused QKV: n_heads {nH}->{nH_new}, n_kv_heads {nKV}->{nKV_infer}, head_dim={head_dim}"
            )
            nH = nH_new
            nKV = nKV_infer
        kv_dim = kv_dim_fused
    else:
        # HF separate K path: infer from K rows
        k0_name = find_by_names(T, "model.layers.0.self_attn.k_proj.weight",
                                "model.layers.0.attn.k_proj.weight")
        if not k0_name:
            # last resort: fall back to config math
            head_dim = dim // nH
            kv_dim = head_dim * nKV
        else:
            kv_dim_from_k = int(T[k0_name].shape[0])
            nH_new, nKV_new, head_dim_new = pick_heads_from_k_shape(
                dim, kv_dim_from_k, nH_cfg=nH, nKV_cfg=nKV)
            if nH_new != nH or nKV_new != nKV:
                print(
                    f"[INFO] overriding heads to match K: n_heads {nH}->{nH_new}, n_kv_heads {nKV}->{nKV_new}, head_dim={head_dim_new}"
                )
                nH, nKV = nH_new, nKV_new
            kv_dim = kv_dim_from_k
            head_dim = head_dim_new

    # Use these updated nH, nKV, kv_dim in the header and everywhere else

    # detect MoE (MXFP4 present)
    use_moe = nExp_cfg > 0 and moe_available(T)
    nExp = nExp_cfg if use_moe else 0
    if use_moe:
        print(f"[INFO] Exporting MoE path: E={nExp}, top_k={topk}")
    else:
        print("[INFO] Exporting DENSE path (no MXFP4 experts found)")

    # ---- header (matches gpt-oss.c) ----
    # int dim, hidden_dim, n_layers, n_heads, n_kv_heads, n_experts, top_k,
    #     vocab_size, seq_len, window, alt_banded; float rope_base, rope_scale
    vocab_field = vsz  # positive => separate lm_head at end
    hdr = struct.pack("<11i2f", dim, hidden, nL, nH, nKV, nExp, topk,
                      vocab_field, seql, win, alt, ropeB, ropeS)

    print(f"[INFO] Writing binary to {outp}")
    with open(outp, "wb") as f:
        f.write(hdr)

        print("[INFO] Writing embeddings...")
        emb = fetch(T,
                    "model.embed_tokens.weight",
                    "tok_embeddings.weight",
                    "model.tok_embeddings.weight",
                    "transformer.wte.weight",
                    "wte.weight",
                    expect_shape=(vsz, dim),
                    hints=("embed", "tok", "wte"))
        write_mat(f, emb)

        print("[INFO] Writing per-layer RMSNorm scales...")
        rms_att = [
            fetch(T,
                  f"model.layers.{i}.input_layernorm.weight",
                  f"model.layers.{i}.attn_norm.weight",
                  f"block.{i}.attn.norm.scale",
                  expect_shape=(dim, ),
                  hints=("norm", "attn", "scale")) for i in range(nL)
        ]
        write_mat(f, np.stack(rms_att, axis=0))
        rms_ffn = [
            fetch(T,
                  f"model.layers.{i}.post_attention_layernorm.weight",
                  f"model.layers.{i}.ffn_norm.weight",
                  f"block.{i}.mlp.norm.scale",
                  expect_shape=(dim, ),
                  hints=("norm", "mlp", "scale")) for i in range(nL)
        ]
        write_mat(f, np.stack(rms_ffn, axis=0))

        print("[INFO] Writing attention weights (Q,K,V,O)...")
        WQ, WK, WV, WO = [], [], [], []
        for i in range(nL):
            q, k, v = fetch_qkv(T, i, dim, kv_dim)
            WQ.append(q)
            WK.append(k)
            WV.append(v)
            wo = fetch(T,
                       f"model.layers.{i}.self_attn.o_proj.weight",
                       f"model.layers.{i}.attn.out_proj.weight",
                       f"block.{i}.attn.out.weight",
                       expect_shape=(dim, dim),
                       hints=("o", "out", "attn"))
            WO.append(wo)
        write_mat(f, np.stack(WQ, axis=0))
        write_mat(f, np.stack(WK, axis=0))
        write_mat(f, np.stack(WV, axis=0))
        write_mat(f, np.stack(WO, axis=0))

        print("[INFO] Writing attention biases...")
        BQ, BK, BV, BO = [], [], [], []
        for i in range(nL):
            bq, bk, bv = fetch_qkv_bias(T, i, dim, kv_dim)
            BQ.append(bq)
            BK.append(bk)
            BV.append(bv)
            bo = fetch(T,
                       f"model.layers.{i}.self_attn.o_proj.bias",
                       f"model.layers.{i}.attn.out_proj.bias",
                       f"block.{i}.attn.out.bias",
                       required=False,
                       expect_shape=(dim, ),
                       hints=("o", "out", "attn"))
            BO.append(nz(bo, (dim, )))
        write_mat(f, np.stack(BQ, axis=0))
        write_mat(f, np.stack(BK, axis=0))
        write_mat(f, np.stack(BV, axis=0))
        write_mat(f, np.stack(BO, axis=0))

        print("[INFO] Writing attention sinks...")
        sinks = []
        for i in range(nL):
            s = fetch(T,
                      f"model.layers.{i}.self_attn.sinks",
                      f"model.layers.{i}.attn.sinks",
                      f"block.{i}.attn.sinks",
                      required=False,
                      expect_shape=(nH, ),
                      hints=("sink", "attn"))
            sinks.append(nz(s, (nH, )))
        write_mat(f, np.stack(sinks, axis=0))

        if use_moe:
            print("[INFO] Writing MoE block...")
            write_moe_from_mxfp4(f, T, nL, dim, hidden, nExp)

        print("[INFO] Writing final norm...")
        norm = fetch(T,
                     "model.norm.weight",
                     "ln_f.weight",
                     "model.ln_f.weight",
                     "final_layernorm.weight",
                     "model.norm.scale",
                     "norm.scale",
                     expect_shape=(dim, ),
                     hints=("norm", "ln_f", "scale"))
        write_mat(f, norm)

        print("[INFO] Writing lm_head or tied embeddings...")
        lm = fetch(T,
                   "lm_head.weight",
                   "model.lm_head.weight",
                   "output.weight",
                   "transformer.lm_head.weight",
                   required=False,
                   expect_shape=(vsz, dim),
                   hints=("lm_head", "out", "cls"))
        if lm is None:
            print(
                "[INFO] No separate lm_head found — exporting with TIED embeddings"
            )
            f.seek(0)
            raw = f.read(struct.calcsize("<11i2f"))
            vals = list(struct.unpack("<11i2f", raw))
            vals[7] = -vsz  # negative vocab => tied in C
            f.seek(0)
            f.write(struct.pack("<11i2f", *vals))
        else:
            write_mat(f, lm)

    print("wrote", outp)


if __name__ == "__main__":
    main()
