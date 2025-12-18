"""
PyTorch DDP training entrypoint.
Standard DistributedDataParallel (Simpler than FSDP, avoids Checkpointing conflicts).
"""

import dataclasses
import gc
import logging
import os
import platform
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # === [修改] 增加 skip_norm_stats 支持，适配 AgiBotDataset ===
    skip_norm_stats = str(os.getenv("skip_norm_stats", "false")).lower() in ["1", "true", "yes", "on"]
    logging.info(f"skip_norm_stats: {skip_norm_stats}")
    
    # 传入 skip_norm_stats
    data_loader = _data.create_data_loader(
        config, 
        framework="pytorch", 
        shuffle=True, 
        skip_norm_stats=skip_norm_stats
    )
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors", metadata={"global_step": str(global_step)})

        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()

        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        
        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            raise RuntimeError(
                "Out of memory while loading checkpoint."
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    
    if dist.is_initialized():
        if dist.get_rank() == 0:
            logging.info(
                f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB"
            )


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    resuming = False
    if config.resume:
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            resuming = True
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                 logging.info(f"Resuming from: {exp_checkpoint_dir} at step {latest_step}")
            else:
                 raise FileNotFoundError("No valid checkpoint to resume")
        else:
            raise FileNotFoundError(f"Checkpoint dir {exp_checkpoint_dir} not found")
    elif config.overwrite and config.checkpoint_dir.exists():
        if is_main:
            shutil.rmtree(config.checkpoint_dir)
            logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")
        if use_ddp: torch.distributed.barrier()

    if not resuming and is_main:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if use_ddp: torch.distributed.barrier()

    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    if is_main:
        logging.info(f"Using batch size per GPU: {effective_batch_size} (World Size: {world_size})")

    loader, data_config = build_datasets(config)

    # --- Build Model ---
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)

    # DDP 模式下，这里使用 model.to(dtype) 是安全的
    # 如果你想用 bfloat16 训练，这里可以转，或者用 AMP
    # 推荐：保持 float32 权重，在 forward 中用 autocast，或者直接转 bfloat16
    # 既然之前的脚本是 float32 policy，这里我们先用 float32 保证稳健
    model = model.to(dtype=torch.float32)

    # DDP 中，这个自带的 gradient_checkpointing 是安全的，可以开启！
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing (DDP Safe)")
    else:
        logging.info("Gradient checkpointing not supported/found")

    if is_main:
        log_memory_usage(device, 0, "after_model_creation")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False, # Pi0 通常不需要这个，设为 False 更快
            gradient_as_bucket_view=True,
        )

    if config.pytorch_weight_path is not None:
        if is_main: logging.info(f"Loading weights from: {config.pytorch_weight_path}")
        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        # DDP 需要 load 到 module
        model_to_load = model.module if use_ddp else model
        safetensors.torch.load_model(model_to_load, model_path)
        if is_main: logging.info("Weights loaded.")

    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        if is_main: logging.info(f"Resumed from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []

    if is_main:
        logging.info(f"Start Training loop...")

    pbar = tqdm.tqdm(total=config.num_train_steps, initial=global_step, disable=not is_main) if is_main else None

    while global_step < config.num_train_steps:
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            if global_step >= config.num_train_steps:
                break

            observation = jax.tree.map(lambda x: x.to(device), observation)
            actions = actions.to(torch.float32).to(device)

            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # === [修改] 开启混合精度 (BF16) ===
            # 这会大幅减少显存占用，并加速训练
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                losses = model(observation, actions)
                
                if isinstance(losses, list | tuple):
                    losses = torch.stack(losses)
                elif not isinstance(losses, torch.Tensor):
                    losses = torch.tensor(losses, device=device, dtype=torch.float32)

                loss = losses.mean()
            # ================================

            # Backward
            # 如果用 float16 需要 Scaler，但在 BF16 下通常不需要，直接 backward 即可
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

            optim.step()
            optim.zero_grad(set_to_none=True)

            if is_main:
                infos.append({
                    "loss": loss.item(),
                    "learning_rate": optim.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                })

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)
                avg_grad_norm = sum(info["grad_norm"] for info in infos) / len(infos)
                
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad={avg_grad_norm:.2f} time={elapsed:.1f}s"
                )

                if config.wandb_enabled:
                     wandb.log({
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "grad_norm": avg_grad_norm,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval
                    }, step=global_step)

                start_time = time.time()
                infos = []

            global_step += 1
            save_checkpoint(model, optim, global_step, config, is_main, data_config)

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    if pbar is not None:
        pbar.close()

    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()