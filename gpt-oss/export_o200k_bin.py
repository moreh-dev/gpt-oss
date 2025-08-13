import argparse
from pathlib import Path
import struct
import sys

import tiktoken


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o",
                    "--out",
                    required=True,
                    help="Output tokenizer.bin path")
    ap.add_argument("--encoding",
                    default="o200k_harmony",
                    help="tiktoken encoding name")
    args = ap.parse_args()

    enc = tiktoken.get_encoding(args.encoding)
    mergeable_ranks = enc._mergeable_ranks
    special_tokens = enc._special_tokens
    n_vocab = enc.n_vocab

    id_to_bytes = [None] * n_vocab
    id_to_text = [None] * n_vocab

    # Normal tokens
    for b, rank in mergeable_ranks.items():
        if rank < 0 or rank >= n_vocab:
            continue
        id_to_bytes[rank] = b

    # Special tokens
    for s, tid in special_tokens.items():
        if 0 <= tid < n_vocab:
            id_to_text[tid] = s

    # Fill any missing entries defensively using tiktoken decode
    for tid in range(n_vocab):
        if id_to_bytes[tid] is None and id_to_text[tid] is None:
            try:
                id_to_bytes[tid] = enc.decode_single_token_bytes(tid)
            except Exception:
                # If decode bytes fails, try to find matching special_tokens by id
                for s, sid in special_tokens.items():
                    if sid == tid:
                        id_to_text[tid] = s
                        break

    # Prepare output token bytes and scores
    token_bytes_out = []
    scores = [0.0] * n_vocab
    max_len = 0

    for tid in range(n_vocab):
        s_bytes = None
        if id_to_text[tid] is not None:
            s_bytes = id_to_text[tid].encode("utf-8", errors="strict")
            scores[tid] = -1e30
        else:
            b = id_to_bytes[tid]
            s_bytes = b
            scores[tid] = float(mergeable_ranks.get(b, tid))

        token_bytes_out.append(s_bytes)
        if len(s_bytes) > max_len:
            max_len = len(s_bytes)

    out_path = Path(args.out)
    with out_path.open("wb") as f:
        f.write(struct.pack("i", max_len))
        for tid, s_bytes in enumerate(token_bytes_out):
            f.write(struct.pack("f", scores[tid]))
            f.write(struct.pack("i", len(s_bytes)))
            f.write(s_bytes)


if __name__ == "__main__":
    main()
