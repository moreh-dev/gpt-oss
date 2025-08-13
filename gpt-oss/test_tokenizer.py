"""
test_tokenizer.py — sanity tests for tokenizer.bin against tiktoken o200k_harmony

Run:
  python test_tokenizer.py --bin ./token_test --tok ./tokenizer.bin

If you have not built the harness yet, compile it first:
  gcc -O3 -DTESTING -o token_test token_test.c -lm
"""

import argparse
import subprocess
import sys
from typing import List, Tuple

import tiktoken


def run_c_encoder(binary: str, tokbin: str, text: str) -> List[int]:
    cmd = [binary, "-t", tokbin, "-i", text]
    try:
        out = subprocess.check_output(cmd, text=True, encoding="utf-8").strip()
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] C harness failed with exit code {e.returncode}")
        if e.output:
            print("stdout:", e.output)
        if e.stderr:
            print("stderr:", e.stderr)
        sys.exit(2)
    return [int(x) for x in out.split()] if out else []


def first_diff(a: List[int], b: List[int]) -> Tuple[int, int, int]:
    """Return (index, a_val, b_val) of the first differing token; (-1, -1, -1) if equal."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i, a[i], b[i]
    if len(a) != len(b):
        # difference is in length
        return n, (a[n] if n < len(a) else -1), (b[n] if n < len(b) else -1)
    return -1, -1, -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="path to token_test binary")
    ap.add_argument("--tok", required=True, help="path to tokenizer.bin")
    ap.add_argument("--verbose",
                    action="store_true",
                    help="print all mismatches in detail")
    args = ap.parse_args()

    enc = tiktoken.get_encoding("o200k_harmony")
    if enc.name != "o200k_harmony":
        print(
            f"[ERROR] tiktoken encoding is {enc.name}, expected 'o200k_harmony'"
        )
        sys.exit(2)

    prompts = [
        "Hello",
        "Hello world",
        "Write a short haiku about the ocean.",
        "ฉันรักทะเล",  # Thai
        "naïve façade — déjà vu",
        "🍣 sushi and 🍜 ramen",
        "email: test@example.com",
        "newlines:\nline2\nline3",
        "tabs\tand\tspaces",
        'JSON: {"a": 1, "b": [2,3,4]}',
        "混ぜるな危険",
    ]

    ok = 0
    bad = 0

    for p in prompts:
        c_ids = run_c_encoder(args.bin, args.tok, p)

        # Use encode_ordinary to avoid special-token handling differences
        py_ids = enc.encode_ordinary(p)

        match = (c_ids == py_ids)
        status = "OK" if match else "MISMATCH"
        print(f"prompt = {p!r}")
        print(f"  [{status}] tokens: C={len(c_ids)} PY={len(py_ids)}")
        if not match:
            bad += 1
            i, av, bv = first_diff(c_ids, py_ids)
            if i >= 0:
                print(f"    first diff at idx {i}: C={av} PY={bv}")
            if args.verbose:
                print("    C  :", c_ids)
                print("    PY :", py_ids)
            else:
                print("    C  :", c_ids[:80])
                print("    PY :", py_ids[:80])
        else:
            ok += 1

    total = ok + bad
    print(f"\nSummary: {ok}/{total} matched.")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
