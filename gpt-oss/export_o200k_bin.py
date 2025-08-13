import argparse
import json
from pathlib import Path
import struct
import sys

try:
    import tiktoken
except Exception as e:
    print(
        "ERROR: this script requires the 'tiktoken' package.\n"
        "Install it with: pip install --upgrade tiktoken",
        file=sys.stderr)
    raise


def main():
    ap = argparse.ArgumentParser(
        description=
        "Export o200k_harmony tokenizer.bin for llama2.c-style C runtimes")
    ap.add_argument("-o",
                    "--out",
                    type=Path,
                    required=True,
                    help="Output tokenizer.bin")
    ap.add_argument("--encoding",
                    default="o200k_harmony",
                    help="Tiktoken encoding name (default: o200k_harmony)")
    ap.add_argument("--check-vocab",
                    type=int,
                    default=None,
                    help="Optional expected vocab size (e.g., 201088)")
    args = ap.parse_args()

    enc = tiktoken.get_encoding(args.encoding)

    n_vocab = enc.n_vocab
    if args.check_vocab is not None and n_vocab != args.check_vocab:
        print(
            f"WARNING: encoding reports vocab_size={n_vocab}, but --check-vocab={args.check_vocab}",
            file=sys.stderr)

    # Build per-token "score" based on merge ranks, like Karpathy's tokenizer.bin
    # Tiktoken exposes mergeable_ranks: dict[bytes, int rank], lower=more basic; we invert so higher score wins merges.
    mergeable_ranks = getattr(enc, "mergeable_ranks", None)
    if mergeable_ranks is None:
        # Newer tiktoken uses ._mergeable_ranks (private) but keep a fallback.
        mergeable_ranks = getattr(enc, "_mergeable_ranks", None)
    if mergeable_ranks is None:
        print(
            "ERROR: Could not access mergeable_ranks on tiktoken encoding; please upgrade tiktoken.",
            file=sys.stderr)
        sys.exit(1)

    # Precompute token byte strings via tiktoken; use decode_single_token_bytes to avoid decode errors.
    token_bytes = [enc.decode_single_token_bytes(i) for i in range(n_vocab)]

    # Highest rank should have highest score (for greedy merge). We'll use score = rank (float).
    # For tokens not present in mergeable_ranks (e.g., special tokens), set a very low score so they never win merges.
    MIN_SCORE = -1e10

    scores = [MIN_SCORE] * n_vocab

    # Build a reverse map: token_id -> rank if mergeable
    # mergeable_ranks maps byte sequences to rank; we can map those to ids by encoding back
    # Safer: iterate all ids, look up their byte sequence in mergeable_ranks
    for tid, b in enumerate(token_bytes):
        r = mergeable_ranks.get(b)
        if r is not None:
            # Higher score should be preferred; tiktoken ranks are 0..N-1 with lower meaning more basic.
            # We'll invert with a large constant so that later merges have higher scores.
            scores[tid] = float(r)

    # Prepare string forms for each token. For raw byte tokens (single byte), use "<0x%02X>"
    token_strs = []
    max_str_len = 0
    for tid, b in enumerate(token_bytes):
        if len(b) == 1:
            s = f"<0x{b[0]:02X}>"
        else:
            try:
                s = b.decode("utf-8")
            except UnicodeDecodeError:
                # Fallback: hex-escape all bytes if not valid UTF-8
                s = "".join(
                    chr(x) if 32 <= x <= 126 else f"\\x{x:02x}" for x in b)
        token_strs.append(s)
        if len(s) > max_str_len:
            max_str_len = len(s)

    with args.out.open("wb") as f:
        # write max_token_length
        f.write(struct.pack("i", max_str_len))
        # then for each token: float score, int len, bytes
        for tid, s in enumerate(token_strs):
            f.write(struct.pack("f", scores[tid]))
            s_bytes = s.encode("utf-8")
            f.write(struct.pack("i", len(s_bytes)))
            f.write(s_bytes)

    print(
        f"Wrote tokenizer with {n_vocab} tokens to {args.out} (max_token_length={max_str_len})"
    )


if __name__ == "__main__":
    main()
