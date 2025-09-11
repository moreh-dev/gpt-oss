import argparse
import math
import os
import time
from typing import Any, List

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

# --------------------
# Threads / env
# --------------------
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
torch.set_num_interop_threads(1)

MODEL_PATH_120B = "/remote/vast0/share-mv/lmsys/gpt-oss-120b-bf16"
MODEL_PATH_20B = "/remote/vast0/share-mv/lmsys/gpt-oss-20b-bf16"
EOS_IDS = [199999, 200002]


# --------------------
# Dist wrapper (unifies single- and multi-process paths)
# --------------------
class Comm:

    def __init__(self, mp: int, all_devices: List[str]):
        self.using_dist = int(os.environ.get("WORLD_SIZE", "1")) > 1
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.local_world = 1

        if self.using_dist:
            import torch.distributed as dist
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.local_world = int(
                os.environ.get("LOCAL_WORLD_SIZE",
                               os.environ.get("WORLD_SIZE", "1")))
            # Validate GPU availability for MP groups
            if self.local_world * mp > len(all_devices):
                raise RuntimeError(
                    f"Need LOCAL_WORLD_SIZE*mp = {self.local_world*mp} GPUs on this node "
                    f"but only {len(all_devices)} visible: {all_devices}")
            # Mask this process's GPU group [local_rank*mp : local_rank*mp+mp)
            start = self.local_rank * mp
            group = all_devices[start:start + mp]
            if len(group) != mp:
                raise RuntimeError(
                    f"GPU group sizing failed for local_rank={self.local_rank}, mp={mp}, devices={all_devices}"
                )
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(group)
            torch.cuda.set_device(0)  # within masked view
            dist.init_process_group(backend="nccl")
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            # Single-process path still uses same interface
            if mp > len(all_devices):
                raise RuntimeError(
                    f"Need mp={mp} GPUs but only {len(all_devices)} visible: {all_devices}"
                )
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(all_devices[:mp])
            torch.cuda.set_device(0)

    def all_gather_object(self, obj: Any) -> List[Any]:
        if self.using_dist:
            import torch.distributed as dist
            gathered = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered, obj)
            return gathered
        else:
            return [obj]

    def barrier(self):
        if self.using_dist:
            import torch.distributed as dist
            dist.barrier()

    def destroy(self):
        if self.using_dist:
            import torch.distributed as dist
            dist.destroy_process_group()


# --------------------
# Helpers
# --------------------
def pick_dtype(dtype_arg: str):
    if dtype_arg == "bf16": return torch.bfloat16
    if dtype_arg == "fp16": return torch.float16
    if dtype_arg == "fp32": return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def parse_max_memory(s: str | None):
    if not s:
        return None
    mm = {}
    for item in s.split(","):
        k, v = item.split("=")
        k = k.strip()
        v = v.strip()
        if ":" in k:
            k = k.split(":")[-1]
        mm[int(k)] = v
    return mm


# --------------------
# Main
# --------------------
def main():
    parser = argparse.ArgumentParser(
        description=
        "DP×MP batched generation with STRICT input-order prints & writes (DP from torchrun)."
    )
    parser.add_argument(
        "input_file",
        help="UTF-8 text; first line is count header, then one prompt per line."
    )
    parser.add_argument("--model",
                        "-m",
                        choices=["120b", "20b"],
                        default="20b",
                        help="Bundled model path selector.")
    parser.add_argument("--model-path",
                        default=None,
                        help="Explicit HF model path (overrides --model).")
    parser.add_argument("--batch-size",
                        "-b",
                        type=int,
                        default=16,
                        help="Per-process batch size (per DP replica).")
    parser.add_argument("--max-length",
                        type=int,
                        default=1024,
                        help="Absolute cap (prompt + generated).")
    parser.add_argument("--dtype",
                        choices=["auto", "bf16", "fp16", "fp32"],
                        default="auto",
                        help="Computation dtype.")
    parser.add_argument(
        "--mp",
        type=int,
        choices=[1, 2, 4, 8],
        default=1,
        help="GPUs per model replica (model parallel; always device_map=auto)."
    )
    parser.add_argument(
        "--max-memory-local",
        default=None,
        help=
        'Per-LOCAL-GPU memory for device_map=auto, e.g. "0=180GiB,1=180GiB".')
    parser.add_argument(
        "--tps-include-prompt",
        action="store_true",
        help="Include prompt tokens in tokens/sec calculations.")
    args = parser.parse_args()

    # ---- Load input (skip header)
    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    prompts = lines[1:]
    n_total = len(prompts)

    # ---- Visible devices
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    all_devices = [d.strip() for d in vis.split(",")] if vis else [
        str(i) for i in range(torch.cuda.device_count())
    ]

    # ---- Comm (unified single/multi)
    comm = Comm(mp=args.mp, all_devices=all_devices)

    # ---- Resolve model path & tokenizer
    path_map = {"120b": MODEL_PATH_120B, "20b": MODEL_PATH_20B}
    model_path = args.model_path or path_map[args.model]

    if comm.rank == 0:
        print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ---- Load model (always device_map=auto on the masked local GPUs)
    dtype = pick_dtype(args.dtype)
    if comm.rank == 0:
        if args.mp == 1:
            print(
                f"[rank {comm.rank}] Loading model on single local GPU (dtype={dtype}) via device_map=auto..."
            )
        else:
            print(
                f"[rank {comm.rank}] Loading model sharded across {args.mp} local GPUs (dtype={dtype})..."
            )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        device_map="auto",
        max_memory=parse_max_memory(args.max_memory_local),
    ).eval()
    input_device = next(model.parameters()).device

    # ---- Split prompts contiguously by GLOBAL rank (works for 1 or many)
    chunk = math.ceil(n_total / comm.world_size)
    start_idx = comm.rank * chunk
    end_idx = min(n_total, start_idx + chunk)
    local_count = max(0, end_idx - start_idx)
    local_steps = math.ceil(local_count / max(1, args.batch_size))

    # ---- Step schedule agreement (same path for single/multi)
    steps_list = comm.all_gather_object(local_steps)
    max_steps = max(steps_list)

    # ---- Open output files (writer = rank 0)
    out_text_path = os.path.join("transformers_data",
                                 f"output_{args.model}_{args.dtype}_text.txt")
    out_ids_path = os.path.join(
        "transformers_data", f"output_{args.model}_{args.dtype}_token_ids.txt")
    if comm.rank == 0:
        os.makedirs(os.path.dirname(out_text_path), exist_ok=True)
        os.makedirs(os.path.dirname(out_ids_path), exist_ok=True)
        out_text_f = open(out_text_path, "w", encoding="utf-8")
        out_ids_f = open(out_ids_path, "w", encoding="utf-8")

    # ---- Strict ordered writing/printing state (writer only)
    next_to_write = 0
    pending = {}  # global_idx -> (prompt_text, gen_text, gen_token_ids)

    if comm.rank == 0:
        print(f"Using max_length={args.max_length}")

    # Measure total wall time across the generation phase
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    run_t0 = time.perf_counter()

    # ---- Main loop
    with torch.no_grad():
        for step in range(max_steps):
            s = start_idx + step * args.batch_size
            e = min(end_idx, s + args.batch_size)
            have_work = (s < e)

            if have_work:
                batch_prompts = prompts[s:e]
                enc = tokenizer(batch_prompts,
                                return_tensors="pt",
                                padding=True,
                                add_special_tokens=False)
                in_ids = enc.input_ids.to(input_device)
                attn = (enc.attention_mask if "attention_mask" in enc else
                        torch.ones_like(enc.input_ids)).to(input_device)
                B, Lin = in_ids.shape

                prompt_lens = attn.sum(dim=1)
                budgets = (args.max_length - prompt_lens).clamp_min(0)
                max_new = int(budgets.max().item())
                # Time the forward/generation for this batch
                if input_device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = model.generate(
                    input_ids=in_ids,
                    attention_mask=attn,
                    max_new_tokens=max_new,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    eos_token_id=EOS_IDS,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=False,
                )
                if input_device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                elapsed_forward = max(t1 - t0, 1e-9)

                local_batch_results = []
                tokens_generated_local = 0
                for i in range(B):
                    seq_full = out[i].detach().cpu()
                    gen_start = Lin
                    allowed_end = gen_start + int(budgets[i].item())
                    eos_cut = 10**9
                    for j in range(gen_start, min(allowed_end,
                                                  seq_full.size(0))):
                        if any(
                                int(seq_full[j]) == eos_id
                                for eos_id in EOS_IDS):
                            eos_cut = j + 1
                            break
                    cut_end = min(eos_cut, min(allowed_end, seq_full.size(0)))
                    gen_text = tokenizer.decode(
                        seq_full[gen_start:cut_end],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False)
                    global_idx = s + i
                    prompt_text = batch_prompts[i]
                    gen_token_ids = seq_full[gen_start:cut_end].tolist()
                    tokens_generated_local += len(gen_token_ids)
                    local_batch_results.append(
                        (global_idx, prompt_text, gen_text, gen_token_ids))
            else:
                local_batch_results = []
                tokens_generated_local = 0
                elapsed_forward = 0.0

            # Local per-rank tokens/sec after each forward; no synchronization
            if have_work and elapsed_forward > 0:
                prompt_tokens_local = int(prompt_lens.sum().item())
                tokens_for_tps = tokens_generated_local + (
                    prompt_tokens_local if args.tps_include_prompt else 0)
                local_tps = tokens_for_tps / elapsed_forward
                print(
                    f"[rank {comm.rank} step {step}] tokens={tokens_for_tps} (gen={tokens_generated_local}"
                    + (f", prompt={prompt_tokens_local})" if args.
                       tps_include_prompt else ")") +
                    f" in {elapsed_forward:.3f}s -> {local_tps:.2f} tok/s")

            # Gather results from all ranks (or just wrap local result in list) and strictly order at writer
            gathered = comm.all_gather_object(local_batch_results)

            if comm.rank == 0:
                for part in gathered:
                    if not part:
                        continue
                    for global_idx, prompt_text, gen_text, gen_token_ids in part:
                        pending[global_idx] = (prompt_text, gen_text,
                                               gen_token_ids)

                wrote_any = False
                while next_to_write in pending:
                    prompt_text, gen_text, gen_token_ids = pending.pop(
                        next_to_write)

                    # Ordered print on rank 0 only
                    print(f"[prompt {next_to_write}]")
                    print(gen_text)
                    print("\n[done]")

                    # Ordered file writes
                    out_text_f.write(
                        f'--- Prompt {next_to_write} = "{prompt_text}" ---\n')
                    out_text_f.write(f"{gen_text}\n")
                    out_ids_f.write(" ".join(
                        str(int(t)) for t in gen_token_ids) + "\n")

                    next_to_write += 1
                    wrote_any = True

                if wrote_any:
                    out_text_f.flush()
                    out_ids_f.flush()
                    try:
                        os.fsync(out_text_f.fileno())
                        os.fsync(out_ids_f.fileno())
                    except OSError:
                        pass

    # ---- Final drain and teardown
    if comm.rank == 0:
        while next_to_write in pending:
            prompt_text, gen_text, gen_token_ids = pending.pop(next_to_write)
            print(f"[prompt {next_to_write}]")
            print(gen_text)
            print("\n[done]")
            out_text_f.write(
                f'--- Prompt {next_to_write} = "{prompt_text}" ---\n')
            out_text_f.write(f"{gen_text}\n")
            out_ids_f.write(" ".join(str(int(t))
                                     for t in gen_token_ids) + "\n")
            next_to_write += 1
        out_text_f.flush()
        out_ids_f.flush()
        out_text_f.close()
        out_ids_f.close()

    # Final sync to ensure all ranks finish before final TPS
    comm.barrier()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    run_t1 = time.perf_counter()

    # Final overall tokens/sec computed from output file (writer only)
    if comm.rank == 0:
        total_generated_tokens = 0
        try:
            with open(out_ids_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Each line is space-separated token ids for one sample
                    total_generated_tokens += len(line.split())
        except FileNotFoundError:
            total_generated_tokens = 0

        total_prompt_tokens = 0
        if args.tps_include_prompt:
            # Count prompt tokens for all prompts
            for p in prompts:
                ids = tokenizer(p, add_special_tokens=False).input_ids
                total_prompt_tokens += len(ids)

        elapsed_total = max(run_t1 - run_t0, 1e-9)
        total_tokens = total_generated_tokens + (
            total_prompt_tokens if args.tps_include_prompt else 0)
        print(
            f"[final] tokens={total_tokens} (gen={total_generated_tokens}" +
            (f", prompt={total_prompt_tokens})" if args.
             tps_include_prompt else ")") +
            f" elapsed={elapsed_total:.3f}s -> {total_tokens/elapsed_total:.2f} tok/s"
        )

    comm.destroy()


if __name__ == "__main__":
    main()
