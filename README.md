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

#### Infrastucture technical requirement (Proposed by Long)

1. General
    - [ ] 29 accounts
    - [ ] A shared directory
    - [ ] Node requirement: 4 physical nodes, can be dynamically increased or decreased for other purposes
    - [ ] 3TB disk storage replicated across all physical nodes, used to store models
    - [ ] CPU/GPU info tools
    - [ ] Uniform software environment (ROCm, GCC, etc.) in login node and worker nodes
    - [ ] User guide document
1. Slurm jobs
    - [ ] Can be sent to worker nodes with `srun` command
    - [ ] Time limiation: 1 hour
    - [ ] Multi-nodes limitation: 2 nodes
    - [ ] GPUs limitation: 4 GPUs
    - [ ] 2 jobs can't use same GPU
    - [ ] 2 jobs from same user cannot be executed at the same time
    - [ ] Jobs from users with no jobs running have higher priority than jobs from users with jobs running. Slurm jobs queue has to loosen FIFO policy.
1. Slurm worker nodes
    1. Plan A
        - [ ] 4 physical nodes serve as 4 Slurm worker nodes
        - [ ] Worker nodes can process multiple jobs at the same time
        - [ ] User can define number of GPUs to be allocated (up to 4), default: 1
    2. Plan B (in case Plan A failed)
        - [ ] 8 virtual nodes serve as 8 Slurm worker nodes
        - [ ] Only one job can be processed by a worker node at a time
        - [ ] Each worker node has 4 inter-connected GPUs
        - [ ] No GPU can be accessed by more than 1 worker node

