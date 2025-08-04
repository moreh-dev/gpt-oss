# LLaMA 4 Project

This project is a pure C++ implementation of the LLaMA 4 model.
Its primary goal is educational—helping people understand the architecture and internals of LLaMA 4.

## Project Objectives

- Educational: Demystify the LLaMA 4 architecture through hands-on C++ implementation.
- Optimization Challenge: Compete in optimizing LLaMA 4 inference on a GPU node.
- Also read the [LLaMA 2 architecture explanation](https://github.com/moreh-dev/llama2.c/blob/master/LLAMA2.md).
- Reference: For background, see our previous work on [LLaMA 2 (LLAMA2.cpp)](https://github.com/moreh-dev/llama2.c/tree/350e04fe35433e6d2941dce5a1f53308f87058eb).

## TODOs

### Model C++ Implementation (Owner: Duc)

- [ ] Create `llama-4` repository (CPU baseline)
- [ ] Implement tokenizer
- [ ] Implement model loading
- [ ] Implement forward pass
- [ ] Add logging and debugging utilities

### Accuracy Evaluation (Owner: Tung)

- [ ] Decide evaluation metric (BLEU, MMLU, etc.)
- [ ] Build evaluation scripts
- [ ] Compare with baseline models

### Performance Evaluation (Owner: Long)

- [ ] Define threshold criteria
- [ ] Report throughput and latency metrics

### Server Configuration Requirements (Owner: Huy)

- [ ] SLURM setup and script templates
- [ ] Node requirement: 4 nodes minimum
- [ ] GPU type and configuration specs
- [ ] Software environment (CUDA, GCC, etc.)
