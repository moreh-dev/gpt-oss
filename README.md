# LLaMA 4 Project

This project is a pure C++ implementation of the LLaMA 4 model.
Its primary goal is educational—helping people understand the architecture and internals of LLaMA 4.

## Project Objectives

- Educational: Demystify the LLaMA 4 architecture through hands-on C++ implementation.
- Optimization Challenge: Compete in optimizing LLaMA 4 inference on a GPU node.
- Also read the [LLaMA 2 architecture explanation](https://github.com/moreh-dev/llama2.c/blob/master/LLAMA2.md).
- Reference: For background, see our previous work on [LLaMA 2 (LLAMA2.cpp)](https://github.com/moreh-dev/llama2.c/tree/350e04fe35433e6d2941dce5a1f53308f87058eb).

## TODOs

Update Checkpoint: 11 Aug 2025

### Model C++ Implementation (Owner: Duc)

- [ ] Inference GPT-OSS using CPU only: on going
- [ ] Implement tokenizer: on going
- [x] Implement model loading
- [ ] Implement forward pass: on going

### Accuracy Evaluation (Owner: Tung)

- [x] Decide evaluation metric (BLEU, MMLU, etc.)
- [x] Build evaluation scripts
- [x] Compare with baseline models
- [ ] Make Slide for Output Norm: TODO
- [ ] Make Common Slide: TODO

### Performance Evaluation (Owner: Long)

- [ ] Define threshold criteria: on going
- [x] Report throughput and latency metrics
- [ ] Add logging and debugging utilities: optional
- [ ] Create reference outputs for gpt-oss-20b for 128 prompts: TODO
- [ ] Create reference outputs for gpt-oss-120b for 128 prompts: TODO

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
    - [ ] Time limitation: 1 hour
    - [ ] Multi-nodes limitation: 1 node
    - [ ] GPUs limitation: 8 GPUs
    - [ ] 2 jobs can't use same GPU
    - [ ] 2 jobs from same user cannot be executed at the same time
    - [ ] Jobs from users with no jobs running have higher priority than jobs from users with jobs running. Slurm jobs queue has to loosen FIFO policy: TODO until end project.
1. Slurm worker nodes
    1. Plan A
        - [ ] 4 physical nodes serve as 4 Slurm worker nodes
        - [ ] Worker nodes can process multiple jobs at the same time
        - [ ] User can define number of GPUs to be allocated (up to 8), default: 1
    2. Plan B (in case Plan A failed)
        - [ ] 11 virtual nodes serve as 11 Slurm worker nodes
            - [ ] 3 worker nodes have 8 inter-connected GPUs (type X worker node)
            - [ ] 8 worker nodes have 1 GPU (type Y worker node)
            - [ ] The number of type X and type Y worker node can be dynamically increased or decreased for workload balancing
        - [ ] Only one job can be processed by a worker node at a time
        - [ ] No GPU can be accessed by more than 1 worker node
