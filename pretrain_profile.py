from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import datetime
import random
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2_pytorch import AdamAtan2

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from torch.profiler import profile, ProfilerActivity, record_function
from compare_checkpoints import verify_checkpoint, assert_checkpoints_equal
from utils.functions import load_optim_model_class
load_models_class = load_optim_model_class


# ============== Configuration ==============
SEED = 33
WARMUP_STEPS = 10
PROFILE_STEPS = 5
# ===========================================


def set_deterministic(seed: int):
    """Set all random seeds and enable deterministic algorithms."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    arch: ArchConfig
    data_paths: List[str]
    data_paths_test: List[str] = []
    evaluators: List[EvaluatorConfig] = []

    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    project_name: Optional[str] = None
    run_name: Optional[str] = None
    name_suffix: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    seed: int = 33
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0
    eval_save_outputs: List[str] = []

    ema: bool = False
    ema_rate: float = 0.999
    freeze_weights: bool = False

    # Checkpoint verification
    # todo change this!!!
    # mlp
    reference_checkpoint: Optional[str] = '/home/simon/palic/TinyRecursiveModels/checkpoints/profile_20251211_045622_baseline_eager/step_15.pt'  # Path to reference checkpoint for verification
    # attn
    # reference_checkpoint: Optional[str] = '/home/simon/palic/TinyRecursiveModels/checkpoints/profile_20251211_160751_baseline_eager_attn/step_15.pt'
    verify_atol: float = 1e-6  # Absolute tolerance for verification
    verify_rtol: float = 1e-5  # Relative tolerance for verification


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any
    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths,
        rank=0,
        num_replicas=1,
        **kwargs
    ), split="train")
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,  # Deterministic: no multiprocessing
        pin_memory=True,
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, metadata: PuzzleDatasetMetadata):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False
    )

    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)
        if "MAX_AUTOTUNE" in os.environ:
            print("Compiling with max autotune")
            model = torch.compile(model, mode="max-autotune")
        elif "CUDA_GRAPHS" in os.environ:
            print("Compiling with CUDA graphs")
            model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        elif "DISABLE_COMPILE" not in os.environ:
            print("Compiling with torch.compile default")
            model = torch.compile(model)
        else:
            print("Not compiling")

        if config.load_checkpoint:
            state_dict = torch.load(config.load_checkpoint, map_location="cuda")
            model.load_state_dict(state_dict, assign=True)

    # Optimizers
    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [AdamAtan2(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(config.beta1, config.beta2))]
        optimizer_lrs = [config.lr]
    elif config.freeze_weights:
        optimizers = [CastedSparseEmbeddingSignSGD_Distributed(model.model.puzzle_emb.buffers(), lr=0, weight_decay=config.puzzle_emb_weight_decay, world_size=1)]
        optimizer_lrs = [config.puzzle_emb_lr]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(model.model.puzzle_emb.buffers(), lr=0, weight_decay=config.puzzle_emb_weight_decay, world_size=1),
            AdamAtan2(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(config.beta1, config.beta2))
        ]
        optimizer_lrs = [config.puzzle_emb_lr, config.lr]

    return model, optimizers, optimizer_lrs


def cosine_schedule_with_warmup(step: int, base_lr: float, warmup_steps: int, total_steps: int, min_ratio: float = 0.0):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def train_step(config: PretrainConfig, state: TrainState, batch: dict, batch_size: int):
    state.step += 1

    with record_function("data_to_device"):
        batch = {k: v.cuda() for k, v in batch.items()}

    with record_function("init_carry"):
        if state.carry is None:
            with torch.device("cuda"):
                state.carry = state.model.initial_carry(batch)

    with record_function("forward"):
        state.carry, loss, metrics, _, _ = state.model(carry=state.carry, batch=batch, return_keys=[])

    with record_function("backward"):
        (loss / batch_size).backward()

    with record_function("optimizer_step"):
        for optim, base_lr in zip(state.optimizers, state.optimizer_lrs):
            lr = cosine_schedule_with_warmup(state.step, base_lr, config.lr_warmup_steps, state.total_steps, config.lr_min_ratio)
            for pg in optim.param_groups:
                pg['lr'] = lr
            optim.step()
            optim.zero_grad()

    return {k: v.item() for k, v in metrics.items()}


def save_checkpoint(config: PretrainConfig, state: TrainState):
    if config.checkpoint_path:
        os.makedirs(config.checkpoint_path, exist_ok=True)
        path = os.path.join(config.checkpoint_path, f"step_{state.step}.pt")
        torch.save(state.model.state_dict(), path)
        print(f"Checkpoint saved: {path}")


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def main(hydra_config: DictConfig):
    # Determinism
    set_deterministic(SEED)

    # Config
    config = PretrainConfig(**hydra_config)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config.run_name = f"profile_{timestamp}_{config.name_suffix}"
    config.checkpoint_path = config.checkpoint_path or os.path.join("checkpoints", config.run_name)

    print(f"\n{'='*50}")
    print(f"PROFILING: {WARMUP_STEPS} warmup + {PROFILE_STEPS} profiled steps")
    print(f"Checkpoint: {config.checkpoint_path}")
    print(f"{'='*50}\n")

    # Data & model
    dataloader, metadata = create_dataloader(config, epochs_per_iter=config.epochs, global_batch_size=config.global_batch_size, test_set_mode=False)
    model, optimizers, optimizer_lrs = create_model(config, metadata)
    
    total_steps = int(config.epochs * metadata.total_groups * metadata.mean_puzzle_examples / config.global_batch_size)
    state = TrainState(model=model, optimizers=optimizers, optimizer_lrs=optimizer_lrs, carry=None, step=0, total_steps=total_steps)

    state.model.train()
    batch_iter = iter(dataloader)

    # Warmup
    print(f"Warmup ({WARMUP_STEPS} steps)...")
    for i in range(WARMUP_STEPS):
        _, batch, batch_size = next(batch_iter)
        train_step(config, state, batch, batch_size)
        print(f"  step {i+1}/{WARMUP_STEPS}")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    print(f"\nProfiling ({PROFILE_STEPS} steps)...")

    # Create trace directory
    trace_dir = os.path.join(config.checkpoint_path, "traces")
    os.makedirs(trace_dir, exist_ok=True)

    # Start memory snapshot recording
    torch.cuda.memory._record_memory_history(max_entries=100000)

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for i in range(PROFILE_STEPS):
            _, batch, batch_size = next(batch_iter)
            with record_function(f"step_{i}"):
                train_step(config, state, batch, batch_size)
            print(f"  step {i+1}/{PROFILE_STEPS}")

    torch.cuda.synchronize()

    # Dump memory snapshot
    snapshot_path = os.path.join(trace_dir, "memory_snapshot.pickle")
    torch.cuda.memory._dump_snapshot(snapshot_path)
    torch.cuda.memory._record_memory_history(enabled=None)

    # Save chrome trace
    trace_path = os.path.join(trace_dir, "trace.json")
    prof.export_chrome_trace(trace_path)

    # Print memory stats
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)

    print(f"\n{'='*50}")
    print(f"Peak CUDA memory allocated: {peak_memory_gb:.2f} GB")
    print(f"Peak CUDA memory reserved:  {peak_reserved_gb:.2f} GB")
    print(f"{'='*50}")

    # Save summary
    summary_path = os.path.join(trace_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Peak CUDA memory allocated: {peak_memory_gb:.2f} GB\n")
        f.write(f"Peak CUDA memory reserved:  {peak_reserved_gb:.2f} GB\n\n")
        f.write("=== CUDA Time ===\n")
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
        f.write("\n\n=== CPU Time ===\n")
        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
        f.write("\n\n=== Memory ===\n")
        f.write(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=50))

    prof.export_stacks(os.path.join(trace_dir, "stacks.txt"), "self_cuda_time_total")

    print(f"\nTrace: {trace_path}")
    print(f"Summary: {summary_path}")
    print(f"Memory snapshot: {snapshot_path}")
    print(f"  -> Visualize at: https://pytorch.org/memory_viz")

    # Checkpoint
    save_checkpoint(config, state)

    # Verify checkpoint against reference if provided
    if config.reference_checkpoint:
        checkpoint_path = os.path.join(config.checkpoint_path, f"step_{state.step}.pt")
        is_valid = verify_checkpoint(
            checkpoint_path,
            config.reference_checkpoint,
            atol=config.verify_atol,
            rtol=config.verify_rtol,
        )
        if not is_valid:
            print("WARNING: Checkpoint verification FAILED!")
    elif os.environ.get("REFERENCE_CHECKPOINT"):
        # Also support via environment variable
        checkpoint_path = os.path.join(config.checkpoint_path, f"step_{state.step}.pt")
        is_valid = verify_checkpoint(
            checkpoint_path,
            os.environ["REFERENCE_CHECKPOINT"],
            atol=config.verify_atol,
            rtol=config.verify_rtol,
        )
        if not is_valid:
            print("WARNING: Checkpoint verification FAILED!")

    print(f"\n{'='*50}")
    print("DONE")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()