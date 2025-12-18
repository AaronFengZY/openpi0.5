import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb


import os
from PIL import Image
import time

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

import json
import pathlib


def init_logging():
    """Custom logging format for better readability."""
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
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    # Rank 0 负责创建目录，其他节点等待 (通常 initialize_checkpoint_dir 已经做了，这里只是保险)
    
    run_id = None
    group_name = config.exp_name  # 使用实验名作为 Group

    # 1. 确定 Run ID
    if resuming:
        try:
            # 尝试读取 Run ID
            run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        except Exception as e:
            logging.warning(f"⚠️ [Rank {jax.process_index()}] Failed to read wandb_id: {e}")
            # 如果读不到，就当新训练处理? 或者抛出异常? 视需求而定。
            # 这里如果不抛异常，后面 wandb.init(id=None) 会创建新的 run，这可能不是你想要的。
            # 但对于 Rank > 0，如果读不到，其实问题不大，因为它们只是附属进程。
            pass
    
    # 2. 确定 Run Name (如果是新训练)
    if not run_id:
        # 如果没有环境变量，使用默认逻辑
        env_run_name = os.getenv("WANDB_RUN_NAME")
        if env_run_name:
            run_name = env_run_name
        else:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            run_name = f"{config.exp_name}_{timestamp}"
    else:
        run_name = None # Resume 模式下 name 通常被忽略或由 server 决定

    # 3. 统一初始化
    # 每个节点都执行，确保都能上传 System Metrics
    wandb.init(
        id=run_id,
        name=run_name,
        resume="must" if run_id else "allow",
        config=dataclasses.asdict(config),
        project=config.project_name,
        group=group_name, # 关键：让它们在网页上聚在一起
    )

    # 4. [Rank 0 独占] 保存 ID (仅当是新训练时)
    if not resuming and jax.process_index() == 0:
        try:
            (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)
            logging.info(f"✅ [Rank 0] Saved wandb_id to {ckpt_dir}")
        except Exception as e:
            logging.warning(f"⚠️ [Rank 0] Failed to save wandb_id: {e}")

    # 5. [Rank 0 独占] Log Code
    if log_code and jax.process_index() == 0:
        wandb.run.log_code(epath.Path(__file__).parent.parent)

def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    # grads = jax.tree.map(lambda g: jax.lax.pmean(g, axis_name="dp"), grads)
    # loss = jax.lax.pmean(loss, axis_name="dp")  # 可选，但我建议加上，日志更一致

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()

    # =================================================================
    # [兼容性修改] JAX 分布式初始化 (自动适配 Single/Multi Node)
    # =================================================================
    # 逻辑：只有当环境变量中存在 RANK (由 AMLT/Torchrun 注入) 时才初始化分布式
    # 否则默认单机运行，不报错。
    
    os.environ["JAX_COORDINATION_SERVICE_TIMEOUT_SEC"] = "3600"
    logging.info(f"⏰ [Python] Set JAX timeout to 3600s to survive slow startup.")

    if os.environ.get('RANK'):
        try:
            # 1. 获取分布式参数
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            master_addr = os.environ.get('MASTER_ADDR', 'localhost')
            master_port = os.environ.get('MASTER_PORT', '12355')
            coordinator_address = f"{master_addr}:{master_port}"

            logging.info(f"🌍 [Dist] Mode Detected. Rank: {rank}/{world_size}, Master: {coordinator_address}")

            # 2. 初始化 JAX 分布式
            jax.distributed.initialize(
                coordinator_address=coordinator_address,
                num_processes=world_size,
                process_id=rank,
                local_device_ids=None, # 让 JAX 自动检测本地可见的 GPU
                initialization_timeout=3600  # 设置为 1 小时
            )
            logging.info(f"✅ [Dist] Initialized! Global Device Count: {jax.device_count()}")
            
        except Exception as e:
            logging.error(f"❌ [Dist] Initialization failed: {e}")
            raise e
    else:
        logging.info("ℹ️ [Dist] No RANK found. Running in SINGLE-NODE mode.")

    logging.info(f"Running on node: {platform.node()}")


    # logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    # --- build 2D mesh: dp (nodes) x fsdp (local gpus) ---
    num_nodes = jax.process_count()              # 4
    local_gpus = jax.local_device_count()        # 8
    assert jax.device_count() == num_nodes * local_gpus, (jax.device_count(), num_nodes, local_gpus)

    devices_2d = np.array(jax.devices()).reshape((num_nodes, local_gpus))
    mesh = jax.sharding.Mesh(devices_2d, axis_names=("dp", "fsdp"))

    # shard batch over BOTH dp and fsdp so global batch is distributed across 32 devices
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("dp", "fsdp"),))

    # replicated scalar / rng etc
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # =================================================================
    # [关键修复] 分布式安全清理：只有 Rank 0 负责 Overwrite 删除
    # =================================================================
    # 目的：防止多个节点同时执行 rmtree 导致 FileNotFoundError
    if config.overwrite and jax.process_index() == 0:
        ckpt_path = epath.Path(config.checkpoint_dir)
        if ckpt_path.exists():
            logging.info(f"🧹 [Rank 0] Overwrite flag is set. Cleaning up {ckpt_path}...")
            import shutil
            try:
                # 使用 shutil 强力删除，不用 etils 防止 backend 兼容问题
                if ckpt_path.is_dir():
                    shutil.rmtree(str(ckpt_path))
                else:
                    ckpt_path.unlink()
                logging.info("✅ [Rank 0] Cleanup done.")
            except Exception as e:
                logging.warning(f"⚠️ [Rank 0] Cleanup failed (might be deleted already): {e}")

    # 必须让其他节点等待 Rank 0 删完，否则它们可能会试图创建一个正在被删除的目录
    if config.overwrite:
        # 使用 JAX 的 barrier (或者简单的 sleep)
        logging.info(f"⏳ [Rank {jax.process_index()}] Waiting for Rank 0 to cleanup...")
        # 如果没有很好的 barrier 机制，简单的 sleep 也能解决大部分问题，或者依赖后面的 initialize_checkpoint_dir 自动重建
        time.sleep(5) 

    # =================================================================

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # =================================================================
    # 🚨 [Auto-Config v3] 终极补丁：覆盖 Stats + 移除 ResizeImages
    # =================================================================
    
    # 1. 准备 Stats 数据
    env_stats_path = os.getenv("NORM_STATS_FILE")
    real_stats = None
    
    if env_stats_path:
        logging.info(f"🔄 [Auto-Config] Loading stats from: {env_stats_path}")
        with open(env_stats_path, 'r') as f:
            loaded = json.load(f)
            real_stats = loaded.get("norm_stats", loaded)
    else:
        # 如果没环境变量，造个假的防止报错 (Dataset内部已处理归一化)
        logging.info(f"⚠️ [Auto-Config] No Env Stats, using dummy stats.")
        dummy = {"mean": np.zeros(32), "std": np.ones(32), "q01": np.zeros(32), "q99": np.ones(32)}
        real_stats = {"state": dummy, "actions": dummy}

# =================================================================
    # 🚨 [Auto-Config v4] 终极补丁：Stats + NoResize + Repack修正
    # =================================================================
    
    # 1. 准备 Stats
    env_stats_path = os.getenv("NORM_STATS_FILE")
    real_stats = None
    
    if env_stats_path:
        logging.info(f"🔄 [Auto-Config] Loading stats from: {env_stats_path}")
        with open(env_stats_path, 'r') as f:
            loaded = json.load(f)
            real_stats = loaded.get("norm_stats", loaded)
    else:
        logging.info(f"⚠️ [Auto-Config] No Env Stats, using dummy stats.")
        dummy = {"mean": np.zeros(32), "std": np.ones(32), "q01": np.zeros(32), "q99": np.ones(32)}
        real_stats = {"state": dummy, "actions": dummy}

    # 2. 定义补丁函数
    original_create = config.data.create
    import openpi.transforms as _transforms 

    def patched_create(assets_dirs, model_config):
        # A. 原始逻辑
        data_cfg = original_create(assets_dirs, model_config)
        
        # B. 【模型变换】只保留 Tokenize 和 Padding，剔除 Resize
        existing_inputs = data_cfg.model_transforms.inputs
        filtered_inputs = []
        for t in existing_inputs:
            t_name = t.__class__.__name__
            if "ResizeImages" in t_name or "PadStatesAndActions" in t_name:
                logging.info(f"✂️ [Patch] Removing Model Transform: {t_name}")
                continue
            filtered_inputs.append(t)
        new_model_transforms = _transforms.Group(inputs=filtered_inputs, outputs=[])

        # C. 【数据变换】置空 (移除 Libero)
        empty_data_transforms = _transforms.Group(inputs=[], outputs=[])

        # D. 【关键新增】修正 RepackTransform (搬家)
        # 目的：去掉 'observation/' 前缀，满足 Model.from_dict 的需求
        # 同时构造 image 字典结构
        new_repack_transforms = _transforms.Group(
            inputs=[_transforms.RepackTransform({
                # 1. 图像 (保持不变)
                "image/base_0_rgb": "head",              
                "image/left_wrist_0_rgb": "left_gripper",
                "image/right_wrist_0_rgb": "right_gripper",
                
                # 2. [新增] 图像 Mask (必须一一对应)
                # 模型会在 data["image_mask"] 下寻找对应的键
                "image_mask/base_0_rgb": "head_mask",
                "image_mask/left_wrist_0_rgb": "left_gripper_mask",
                "image_mask/right_wrist_0_rgb": "right_gripper_mask",
                
                # 3. 状态 (保持不变)
                "state": "states",                       
                "actions": "actions",
                "prompt": "prompt",
            })]
        )

        # E. 组装所有修改
        logging.info("💉 [Patch] Applying FULL Fixes (Stats, Resize, Repack)...")
        
        data_cfg = dataclasses.replace(
            data_cfg, 
            norm_stats=real_stats,          
            use_quantile_norm=False,        
            asset_id="agibot_full",         
            model_transforms=new_model_transforms, 
            data_transforms=empty_data_transforms,
            repack_transforms=new_repack_transforms # <--- 注入新的 Repack
        )
        return data_cfg
    
    # 3. 挂载补丁
    object.__setattr__(config.data, "create", patched_create)
    
    logging.info("✅ [Auto-Config] Patch applied successfully!")

    # =================================================================
    # 3. [关键修改] 修改 Seed 防止多机数据重复
    # =================================================================
    # 必须在 create_data_loader 之前执行
    current_process_id = jax.process_index() 
    if current_process_id > 0:
        logging.info(f"🎲 [Dist] Shifting data seed for Rank {current_process_id} to avoid duplication.")
        
        # 【修正】使用 replace 创建一个新的 config 对象
        config = dataclasses.replace(config, seed=config.seed + current_process_id)
        
    logging.info(f"📉 Creating Data Loader with Seed: {config.seed}")

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")


    # --------------------------------------------------------------------------
    # [可视化修复] 打印调试信息并修复 WandB 图片格式 (Channel First -> Last)
    # --------------------------------------------------------------------------
    def get_local_numpy(jax_array):
        # 如果是普通 numpy 数组直接返回
        if isinstance(jax_array, (np.ndarray, jnp.ndarray)) and not hasattr(jax_array, 'addressable_shards'):
            return np.array(jax_array)
            
        try:
            # 获取本地可见的所有分片 (Addressable Shards)
            # 每个 shard.data 是存储在本地 GPU 上的 jax.Array
            local_shards = jax_array.addressable_shards
            if not local_shards:
                return np.array([]) # 理论上不应发生
            
            # 将所有本地分片拼接起来，形成当前节点的 Local Batch
            # 例如: 全局 512，本地就是 256
            local_data = np.concatenate([np.array(s.data) for s in local_shards], axis=0)
            return local_data
        except Exception as e:
            logging.warning(f"⚠️ Failed to gather local shards: {e}")
            return np.array([])

    obs, act = batch
    logging.info("=== DEBUG: Checking Batch Data Structure (Local Only) ===")
    # 1. 打印基础统计信息 (使用 get_local_numpy)
    for k in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
        if k in obs.images:
            # 【关键修改】使用 get_local_numpy 替代 np.array()
            img_tensor = get_local_numpy(obs.images[k])
            mask_tensor = get_local_numpy(obs.image_masks[k])
            
            if img_tensor.size > 0:
                logging.info(f"[{k}] LocalShape: {img_tensor.shape}, Mean: {img_tensor.mean():.4f}, Valid: {mask_tensor.sum()}/{mask_tensor.size}")

    # 2. 生成 WandB 可视化图片
    # 只在 Rank 0 上做，且只画本地的图片
    if jax.process_index() == 0: 
        try:
            images_to_log = []
            first_key = next(iter(obs.images.keys()), None)
            
            if first_key:
                # 获取该 Key 的本地数据来确定长度
                local_sample_img = get_local_numpy(obs.images[first_key])
                batch_size_vis = len(local_sample_img)
                
                # 最多显示 5 张
                for i in range(min(5, batch_size_vis)):
                    imgs_list = []
                    for img_key, img_jax_array in obs.images.items():
                        # 【关键修改】获取本地数据
                        local_batch = get_local_numpy(img_jax_array)
                        if len(local_batch) <= i: continue

                        arr = local_batch[i] # 取出第 i 张图
                        
                        # 归一化处理
                        if arr.min() < 0:
                            arr = ((arr + 1.0) / 2.0 * 255).astype(np.uint8)
                        elif arr.max() <= 1.0:
                            arr = (arr * 255).astype(np.uint8)
                        else:
                            arr = arr.astype(np.uint8)

                        imgs_list.append(arr)
                    
                    if imgs_list:
                        concat_img = np.concatenate(imgs_list, axis=1)
                        images_to_log.append(wandb.Image(concat_img, caption=f"Sample {i}"))

                if images_to_log:
                    wandb.log({"camera_views": images_to_log}, step=0)
                    logging.info("✅ [Viz] Camera views logged to WandB successfully.")
            else:
                logging.warning("⚠️ [Viz] No images found in batch to log.")
                
        except Exception as e:
            logging.warning(f"⚠️ [Viz] Failed to log images to WandB: {e}")
            # 打印详细报错方便调试，但不中断训练
            import traceback
            traceback.print_exc()

    # # Log images from first batch to sanity check.
    # images_to_log = [
    #     wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
    #     for i in range(min(5, len(next(iter(batch[0].images.values())))))
    # ]
    # wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
