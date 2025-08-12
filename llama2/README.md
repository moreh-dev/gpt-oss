# Llama 2 C Inference

This folder contains a simple C implementation for running small Llama 2 models.

## Files

- `run.c` — Main inference code
- `test.c` — Basic unit tests
- `stories15M.bin`, `stories42M.bin` — Pretrained TinyStories models
- `tokenizer.bin` — Tokenizer data

## Quickstart

Download a model:

```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
```

Build:

```bash
make run
```

Run:

```bash
./run stories15M.bin
./run stories42M.bin
```

Test:

```bash
make test
```

## Notes

- The code uses `tokenizer.bin` for tokenization.
- The model file (`stories15M.bin`) is a small Llama 2 variant trained on TinyStories.
