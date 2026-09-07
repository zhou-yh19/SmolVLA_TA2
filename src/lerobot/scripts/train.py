#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time
from contextlib import nullcontext
from functools import partial
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer
import os
from datetime import timedelta


from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.constants import ACTION
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.datasets.utils_must import multidataset_collate_fn, true_action_dim
# from lerobot.envs.factory import make_env  # Removed - not needed for SmolVLA2 pretraining
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla2.modeling_smolvla2 import resolved_action_dim
from lerobot.policies.utils import get_device_from_parameters
# from lerobot.scripts.eval import eval_policy  # Removed - not needed for SmolVLA2 pretraining
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger
from lerobot.utils.trackio_utils import TrackIOLogger

def is_launched_with_accelerate() -> bool:
    return "ACCELERATE_MIXED_PRECISION" in os.environ


def amp_dtype(device_type: str) -> torch.dtype | None:
    """Autocast dtype for this device, or None to keep torch's own default.

    `torch.autocast` defaults to float16 on CUDA, but the VLM is loaded in
    bfloat16 (`smolvlm_with_expert2.py`) and the shipped accelerate configs set
    `mixed_precision: 'no'`, so `accelerator.backward` installs no loss scaler
    and the `GradScaler` built in `train()` is wired into the single-process
    branch only. fp16 activations with no scaling anywhere is exactly the
    combination where small gradients underflow to zero. bfloat16 carries
    fp32's exponent range, so it needs no scaling and matches the weights
    already in memory.
    """
    if device_type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return None


def amp_autocast(device_type: str, use_amp: bool):
    """AMP context whose dtype matches the weights already in memory."""
    if not use_amp:
        return nullcontext()
    dtype = amp_dtype(device_type)
    if dtype is None:
        return torch.autocast(device_type=device_type)
    return torch.autocast(device_type=device_type, dtype=dtype)


def amp_needs_grad_scaler(device_type: str) -> bool:
    """Only fp16 autocast needs loss scaling; bfloat16 does not."""
    return amp_dtype(device_type) is torch.float16


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
    accelerator=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    
    grad_norm = 0.0  # Initialize grad_norm to avoid undefined variable

    if accelerator:
        with accelerator.accumulate(policy):
            with amp_autocast(device.type, use_amp):
                loss, output_dict = policy.forward(batch)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    grad_clip_norm,
                    error_if_nonfinite=False,
                )
            optimizer.step()
            optimizer.zero_grad()
    else:
        # Standard training loop without accelerate
        with amp_autocast(device.type, use_amp):
            loss, output_dict = policy.forward(batch)
        
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            grad_clip_norm,
            error_if_nonfinite=False,
        )
        grad_scaler.step(optimizer)
        grad_scaler.update()
        optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        if accelerator:
            accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()
        else:
            policy.update()
  
    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def run_validation(
    policy: PreTrainedPolicy,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int,
    action_batches: int,
    use_amp: bool,
    seed: int,
    action_dim: int | None = None,
) -> dict[str, float]:
    """Score the current weights on the held-out episodes.

    Two different questions are answered here, because the cheap metric alone is
    not enough to act on:

    * `val_loss` is the flow-matching denoising loss on unseen episodes. It is
      the same quantity as the training loss, so the gap between the two is what
      says whether the policy is memorising the training episodes.
    * `val_action_mae*` is the open-loop error of a full 10-step denoising
      rollout, in physical units, per action dimension. This is the diagnostic
      that actually maps onto the robot: it says which joint is wrong and by how
      many radians, and it exposes a gripper dimension that has collapsed to its
      mean while the averaged loss still looks healthy.

    Neither number predicts closed-loop task success -- behaviour cloning suffers
    from covariate shift, so a policy can improve on both and still fail on the
    robot. They bound the problem from one side only: a rising `val_loss` or a
    stuck `val_action_mae` is real bad news, while good values are necessary
    rather than sufficient.
    """
    was_training = policy.training
    policy.eval()

    def amp_ctx():
        return amp_autocast(device.type, use_amp)

    # The policy reports the *padded* action width and the collate function pads
    # the ground truth to match, so the trailing columns are zero on one side and
    # unconstrained model output on the other. Averaging over them would roughly
    # halve every number reported here, hence the caller-supplied true width.
    if action_dim is None:
        action_dim = resolved_action_dim(policy.config)
    max_action_dim = policy.config.max_action_dim
    chunk_size = policy.config.chunk_size

    loss_sum = 0.0
    loss_count = 0
    abs_err_sum = torch.zeros(action_dim, dtype=torch.float64, device=device)
    abs_err_count = 0.0
    num_loss_batches = 0
    num_action_batches = 0

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break

                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device, non_blocking=True)

                gt_actions = batch[ACTION]
                bsize, horizon = gt_actions.shape[0], gt_actions.shape[1]

                # `SmolVLA2Policy.forward` draws fresh noise and a fresh time on
                # every call, so an unseeded validation loss moves more between
                # two calls on identical weights than it does between
                # checkpoints. Both are drawn here instead, from a CPU generator
                # seeded by the batch index, so every checkpoint is scored on
                # exactly the same denoising problem.
                # `forward` needs one noise vector per ground-truth timestep,
                # while `sample_actions` needs exactly `chunk_size` of them; the
                # two agree in this config, but drawing the longer of the two and
                # slicing keeps them from ever disagreeing silently.
                generator = torch.Generator().manual_seed(seed + batch_idx)
                noise = torch.randn(
                    (bsize, max(horizon, chunk_size), max_action_dim),
                    generator=generator,
                    dtype=torch.float32,
                ).to(device)
                # One flow-matching time per sample, spread evenly over (0, 1] so
                # a single batch still covers the whole trajectory rather than a
                # single noise level. The loader is unshuffled, so `bsize` -- and
                # therefore this grid -- is identical at every checkpoint.
                flow_time = ((torch.arange(bsize, dtype=torch.float32) + 0.5) / bsize).to(device)

                with amp_ctx():
                    loss, _ = policy.forward(batch, noise=noise[:, :horizon], time=flow_time)
                loss_sum += loss.item() * bsize
                loss_count += bsize
                num_loss_batches += 1

                if batch_idx < action_batches:
                    with amp_ctx():
                        pred = policy.select_action_chunk(batch, noise=noise[:, :chunk_size])
                    # `select_action_chunk` unnormalizes its output and
                    # `batch[ACTION]` was never normalized, so this difference is
                    # already in radians / gripper units.
                    steps = min(pred.shape[1], horizon)
                    err = (
                        pred[:, :steps, :action_dim].float()
                        - gt_actions[:, :steps, :action_dim].float()
                    ).abs()
                    is_pad = batch.get(f"{ACTION}_is_pad")
                    if is_pad is None:
                        valid = torch.ones((bsize, steps, 1), dtype=torch.float32, device=err.device)
                    else:
                        valid = (~is_pad[:, :steps].to(torch.bool)).unsqueeze(-1).to(torch.float32)
                    abs_err_sum += (err * valid).sum(dim=(0, 1)).double()
                    abs_err_count += valid.sum().item()
                    num_action_batches += 1
    finally:
        if was_training:
            policy.train()

    metrics: dict[str, float] = {}
    if loss_count > 0:
        metrics["val_loss"] = loss_sum / loss_count
        metrics["val_samples"] = float(loss_count)
        metrics["val_batches"] = float(num_loss_batches)
    if abs_err_count > 0:
        per_dim = (abs_err_sum / abs_err_count).cpu()
        metrics["val_action_mae"] = per_dim.mean().item()
        metrics["val_action_batches"] = float(num_action_batches)
        for d in range(action_dim):
            metrics[f"val_action_mae_dim{d:02d}"] = per_dim[d].item()
    return metrics


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    
    accelerator = None  # Initialize accelerator variable
    
    if is_launched_with_accelerate():
        import accelerate

        # For example pi0 has unused params (last llm block)
        from accelerate import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # accelerator = accelerate.Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs])
        from accelerate import InitProcessGroupKwargs
        # Set NCCL timeout (default 30 minutes = 1800 seconds)
        nccl_timeout = getattr(cfg, 'nccl_timeout', 1800)
        ddp_init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=nccl_timeout))
        # Set gradient accumulation steps (default 1)
        gradient_accumulation_steps = getattr(cfg, 'gradient_accumulation_steps', 1)
        accelerator = accelerate.Accelerator(step_scheduler_with_optimizer=False, gradient_accumulation_steps=gradient_accumulation_steps, kwargs_handlers=[ddp_init_kwargs, ddp_kwargs])
        if accelerator is not None and not accelerator.is_main_process:
            # Disable duplicate logging on non-main processes
            logging.info(f"Setting logging level on non-main process {accelerator.process_index} to WARNING.")
            logging.getLogger().setLevel(logging.WARNING)

    logging.info(pformat(cfg.to_dict()))

    if accelerator and not accelerator.is_main_process:
            # Disable logging on non-main processes.
            cfg.wandb.enable = False
            cfg.trackio.enable = False

    # Initialize loggers
    wandb_logger = None
    trackio_logger = None
    
    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    elif cfg.trackio.enable and cfg.trackio.project:
        trackio_logger = TrackIOLogger(cfg)
    else:
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    # Environment evaluation disabled for SmolVLA2 pretraining
    # if cfg.eval_freq > 0 and cfg.env is not None:
    #     logging.info("Creating env")
    #     eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    # The dataset pads every action to `max_action_dim` and rewrites the action
    # feature shape to match, so the policy config alone cannot tell a real
    # dimension from padding. Recover the real width here, while the dataset is
    # in scope, so `forward` averages the loss over real dimensions only instead
    # of diluting it with the always-zero padded slots.
    train_action_dim = true_action_dim(dataset, policy.config.action_feature.shape[0])
    policy.config.true_action_dim = train_action_dim
    if train_action_dim != policy.config.action_feature.shape[0]:
        logging.info(
            f"Action feature is padded to {policy.config.action_feature.shape[0]}; "
            f"training loss is averaged over the first {train_action_dim} real dimensions."
        )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    # A scaler is only meaningful for fp16 autocast; under bfloat16 it would scale
    # and unscale gradients that were never at risk of underflowing. It stays
    # unused on the accelerate branch either way.
    grad_scaler = GradScaler(
        device.type, enabled=cfg.policy.use_amp and amp_needs_grad_scaler(device.type)
    )
    if cfg.policy.use_amp:
        logging.info(
            f"AMP enabled: autocast dtype {amp_dtype(device.type) or 'torch default'}, "
            f"grad scaler enabled={grad_scaler.is_enabled()}"
        )

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None
    
    collate_fn = None
    if isinstance(dataset, MultiLeRobotDataset):
        keys_to_max_dim = {
            key: (max_dim,)
            for key, max_dim in dataset.meta.keys_to_max_dim.items()
            if max_dim is not None and key in ["action", "observation.state", "observation.environment_state"]
        }
        collate_fn = partial(multidataset_collate_fn, keys_to_max_dim=keys_to_max_dim)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    # Validation runs on the main process only, over whole episodes held out of
    # training, and is deliberately kept out of `accelerator.prepare`: a sharded
    # loader would make the score depend on the process count, and running a
    # forward on the DDP wrapper outside the training loop risks a collective
    # mismatch between ranks. Only the main process builds the dataset at all.
    validation_enabled = cfg.dataset.val_episodes_per_dataset > 0
    val_dataloader = None
    val_action_dim = None
    if validation_enabled and (accelerator is None or accelerator.is_main_process):
        val_dataset = make_dataset(cfg, split="val")
        if val_dataset is None:
            logging.warning(
                "val_episodes_per_dataset > 0 but no episodes could be held out; "
                "validation is disabled for this run."
            )
        else:
            val_action_dim = true_action_dim(val_dataset, train_action_dim)
            val_collate_fn = None
            if isinstance(val_dataset, MultiLeRobotDataset):
                val_keys_to_max_dim = {
                    key: (max_dim,)
                    for key, max_dim in val_dataset.meta.keys_to_max_dim.items()
                    if max_dim is not None
                    and key in ["action", "observation.state", "observation.environment_state"]
                }
                val_collate_fn = partial(multidataset_collate_fn, keys_to_max_dim=val_keys_to_max_dim)
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                collate_fn=val_collate_fn,
                num_workers=min(cfg.num_workers, 4),
                batch_size=cfg.batch_size,
                shuffle=False,
                pin_memory=device.type != "cpu",
                drop_last=False,
            )
            logging.info(
                f"Validation set: {val_dataset.num_frames} frames, "
                f"{val_dataset.num_episodes} episodes"
            )

    if accelerator:
        policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler
        )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size * (accelerator.num_processes if accelerator else 1),
        dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
            accelerator=accelerator,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            # Log metrics to enabled logger
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            elif trackio_logger:
                trackio_log_dict = train_tracker.to_dict()
                if output_dict:
                    trackio_log_dict.update(output_dict)
                trackio_logger.log_dict(trackio_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            # Unwrap policy from accelerate if needed
            unwrapped_policy = accelerator.unwrap_model(policy) if accelerator else policy
            save_checkpoint(checkpoint_dir, step, cfg, unwrapped_policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            # Log checkpoint to enabled logger
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)
            elif trackio_logger:
                trackio_logger.log_policy(checkpoint_dir)

            if validation_enabled:
                # The barriers keep the other ranks parked while rank 0 evaluates,
                # so nobody runs ahead into a collective its peers have not
                # reached yet.
                if accelerator:
                    accelerator.wait_for_everyone()
                if val_dataloader is not None:
                    val_start = time.perf_counter()
                    val_metrics = run_validation(
                        unwrapped_policy,
                        val_dataloader,
                        device,
                        max_batches=cfg.val_max_batches,
                        action_batches=cfg.val_action_batches,
                        use_amp=cfg.policy.use_amp,
                        seed=cfg.seed if cfg.seed is not None else 0,
                        action_dim=val_action_dim,
                    )
                    val_metrics["val_time_s"] = time.perf_counter() - val_start
                    logging.info(
                        f"Validation at step {step}: "
                        + " ".join(
                            f"{k}={v:.4f}"
                            for k, v in val_metrics.items()
                            if not k.startswith("val_action_mae_dim")
                        )
                    )
                    if wandb_logger:
                        wandb_logger.log_dict(val_metrics, step, mode="eval")
                    elif trackio_logger:
                        trackio_logger.log_dict(val_metrics, step, mode="eval")
                if accelerator:
                    accelerator.wait_for_everyone()
                # `run_validation` put the policy back in train mode, but only on
                # rank 0; restore it everywhere so the modes cannot drift apart.
                policy.train()

        # Environment evaluation disabled for SmolVLA2 pretraining
        # if cfg.env and is_eval_step:
        #     step_id = get_step_identifier(step, cfg.steps)
        #     logging.info(f"Eval policy at step {step}")
        #     with (
        #         torch.no_grad(),
        #         torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
        #     ):
        #         # Unwrap policy from accelerate if needed for evaluation
        #         unwrapped_policy = accelerator.unwrap_model(policy) if accelerator else policy
        #         eval_info = eval_policy(
        #             eval_env,
        #             unwrapped_policy,
        #             cfg.eval.n_episodes,
        #             videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
        #             max_episodes_rendered=4,
        #             start_seed=cfg.seed,
        #         )

        #         )

        #         eval_metrics = {
        #             "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
        #             "pc_success": AverageMeter("success", ":.1f"),
        #             "eval_s": AverageMeter("eval_s", ":.3f"),
        #         }
        #         eval_tracker = MetricsTracker(
        #             cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
        #         )
        #         eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
        #         eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
        #         eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
        #         logging.info(eval_tracker)
        #         if wandb_logger:
        #             wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
        #             wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
        #             wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    # Environment evaluation disabled for SmolVLA2 pretraining
    # if eval_env:
    #     eval_env.close()
    logging.info("End of training")

    if cfg.policy.push_to_hub:
        # Unwrap policy from accelerate if needed
        unwrapped_policy = accelerator.unwrap_model(policy) if accelerator else policy
        unwrapped_policy.push_model_to_hub(cfg)


if __name__ == "__main__":
    init_logging()
    train()
