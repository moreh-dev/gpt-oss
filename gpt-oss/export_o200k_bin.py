#!/usr/bin/env python3
import argparse
from pathlib import Path
import struct
import sys

try:
    import tiktoken
except Exception:
    print(
        "ERROR: this script requires the 'tiktoken' package.\n"
        "Install it with: pip install --upgrade tiktoken",
        file=sys.stderr,
    )
    raise


def is_printable_ascii_byte(b: int) -> bool:
    # printable or whitespace that won't break display
    return (32 <= b <= 126) or b in (9, 10, 13)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o",
                    "--out",
                    required=True,
                    help="Output tokenizer.bin path")
    ap.add_argument("--encoding",
                    default="o200k_harmony",
                    help="Tiktoken encoding name")
    ap.add_argument(
        "--check-vocab",
        type=int,
        default=None,
        help="Optional expected vocab size (e.g., 201088)",
    )
    args = ap.parse_args()

    enc = tiktoken.get_encoding(args.encoding)

    # Mergeable (normal) tokens: bytes -> rank (rank == token_id for non-specials)
    ranks = enc._mergeable_ranks  # {bytes: int}
    # Special tokens: str -> id
    special = enc._special_tokens  # {str: int}

    n_vocab = enc.n_vocab
    if args.check_vocab is not None and n_vocab != args.check_vocab:
        print(
            f"WARNING: encoding reports vocab_size={n_vocab}, but --check-vocab={args.check_vocab}",
            file=sys.stderr,
        )

    # Build id -> bytes/str
    id_to_bytes = [None] * n_vocab  # bytes for normal tokens
    id_to_text = [None] * n_vocab  # str for specials or <0xHH> substitutions

    # Normal tokens
    for b, rank in ranks.items():
        if rank < 0 or rank >= n_vocab:
            continue
        id_to_bytes[rank] = b

    # Special tokens
    for s, tid in special.items():
        if 0 <= tid < n_vocab:
            id_to_text[tid] = s

    # Fill any missing entries defensively using tiktoken decode
    for tid in range(n_vocab):
        if id_to_bytes[tid] is None and id_to_text[tid] is None:
            try:
                id_to_bytes[tid] = enc.decode_single_token_bytes(tid)
            except Exception:
                # If decode bytes fails, try to find matching special by id
                for s, sid in special.items():
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
            # special token: write its literal text as bytes
            s_bytes = id_to_text[tid].encode("utf-8", errors="strict")
            # keep score small so merges never prefer a special accidentally
            scores[tid] = -1e30
        else:
            b = id_to_bytes[tid]
            s_bytes = b
            scores[tid] = float(-tid)

        token_bytes_out.append(s_bytes)
        if len(s_bytes) > max_len:
            max_len = len(s_bytes)

    # Write binary file:
    # [int max_token_length]
    # then for each token id 0..n-1:
    #   [float score][int len][raw bytes]
    out_path = Path(args.out)
    with out_path.open("wb") as f:
        f.write(struct.pack("i", max_len))
        for tid, s_bytes in enumerate(token_bytes_out):
            f.write(struct.pack("f", scores[tid]))
            f.write(struct.pack("i", len(s_bytes)))
            f.write(s_bytes)

    print(
        f"Wrote tokenizer with {n_vocab} tokens to {out_path} (max_token_length={max_len})"
    )


if __name__ == "__main__":
    main()
