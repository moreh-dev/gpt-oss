"""
test_tokenizer.py — sanity tests for tokenizer.bin against tiktoken o200k_harmony

This script:
  1) Builds/loads a small C harness (token_test) that uses the same encoder as gpt-oss.c.
  2) For a battery of prompts, compares the C encoder's token IDs to tiktoken's.
     IMPORTANT: the C encoder injects a leading space (like sentencepiece).
     So we compare C(prompt) == tiktoken.encode(" " + prompt).

Run:
  python test_tokenizer.py --bin ./token_test --tok ./tokenizer.bin

If you have not built the harness yet, compile it first:
  gcc -O3 -DTESTING -o token_test token_test.c -lm
"""

import argparse
import subprocess
import sys

import tiktoken


def run_c_encoder(binary, tokbin, text):
    cmd = [binary, "-t", tokbin, "-i", text]
    out = subprocess.check_output(cmd, text=True).strip()
    return [int(x) for x in out.split()] if out else []


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
        "JSON: {\"a\": 1, \"b\": [2,3,4]}",
        "混ぜるな危険",
    ]

    ok = 0
    bad = 0

    for p in prompts:
        c_ids = run_c_encoder(args.bin, args.tok, p)
        tiktoken_prompt = " " + p
        py_ids = enc.encode(tiktoken_prompt)

        match = (c_ids == py_ids)
        status = "OK" if match else "MISMATCH"
        print(f"prompt = {p!r}")
        print(f"  tiktoken prompt: {tiktoken_prompt!r}")
        print(f"  [{status}] tokens: C={len(c_ids)} PY={len(py_ids)}")
        if not match:
            bad += 1
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

    # Non-zero exit if any mismatch
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
