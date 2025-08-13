"""
test_tokenizer.py — Compare C and Python (tiktoken) tokenizer encode/decode

Usage:
    python test_tokenizer.py --bin ./token_test --tok ./tokenizer.bin [--verbose]
"""

import argparse
import subprocess
import sys
from typing import List, Tuple

import tiktoken


def run_c_encoder(binary: str, tokbin: str, text: str) -> List[int]:
    cmd = [binary, "-t", tokbin, "-i", text]
    try:
        out = subprocess.check_output(cmd, encoding="utf-8",
                                      errors="ignore").strip()
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] C encoder failed: {e}")
        sys.exit(2)
    return [int(x) for x in out.split()] if out else []


def run_c_decoder(binary: str, tokbin: str, text: str) -> str:
    cmd = [binary, "-t", tokbin, "-i", text, "-r"]
    try:
        out = subprocess.check_output(cmd, encoding="utf-8",
                                      errors="ignore").strip()
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] C decoder failed: {e}")
        sys.exit(2)
    lines = out.splitlines()
    return lines[-1] if lines else ""


def first_diff(a: List[int], b: List[int]) -> Tuple[int, int, int]:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i, a[i], b[i]
    if len(a) != len(b):
        return n, (a[n] if n < len(a) else -1), (b[n] if n < len(b) else -1)
    return -1, -1, -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="Path to token_test binary")
    ap.add_argument("--tok", required=True, help="Path to tokenizer.bin")
    ap.add_argument("--verbose", action="store_true", help="Show all details")
    args = ap.parse_args()

    enc = tiktoken.get_encoding("o200k_harmony")
    if enc.name != "o200k_harmony":
        print(
            f"[ERROR] tiktoken encoding is {enc.name}, expected 'o200k_harmony'"
        )
        sys.exit(2)

    with open("prompts.txt", encoding="utf-8") as f:
        prompts = [line.rstrip('\n') for line in f]

    encode_mismatches = 0
    decode_mismatches = 0

    for prompt in prompts:
        c_ids = run_c_encoder(args.bin, args.tok, prompt)
        py_ids = enc.encode_ordinary(prompt)
        c_decoded = run_c_decoder(args.bin, args.tok, prompt)
        py_decoded = enc.decode(py_ids)

        encode_match = (c_ids == py_ids)
        decode_match = (c_decoded == py_decoded)

        print(f"PROMPT: {prompt!r}")
        print(f"  C  encoded: {c_ids}")
        print(f"  PY encoded: {py_ids}")
        print(f"  C  decoded: {c_decoded!r}")
        print(f"  PY decoded: {py_decoded!r}")

        if encode_match:
            print("  [ENCODE MATCH]")
        else:
            print("  [ENCODE MISMATCH]")
            i, av, bv = first_diff(c_ids, py_ids)
            if i >= 0:
                print(f"    First diff at idx {i}: C={av} PY={bv}")
            encode_mismatches += 1

        if decode_match:
            print("  [DECODE MATCH]")
        else:
            print("  [DECODE MISMATCH]")
            decode_mismatches += 1

        if args.verbose or not (encode_match and decode_match):
            print("-" * 60)

    total = len(prompts)
    print(
        f"\nSummary: {total - encode_mismatches}/{total} encode matched, {total - decode_mismatches}/{total} decode matched."
    )
    if encode_mismatches or decode_mismatches:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
