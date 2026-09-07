#!/usr/bin/env python
"""Compare a trained checkpoint's predicted action chunks against the dataset ground truth.

This is an *open-loop* check: at a given frame the policy sees exactly the
observation the dataset recorded, predicts one chunk of `chunk_size` actions,
and that chunk is lined up against the actions the teleoperator actually
produced from that same frame onward. Nothing is fed back into the robot, so a
small error here does not prove the policy would succeed on hardware -- errors
compound once the policy starts driving the state distribution. It bounds the
problem from one side only: a *large* error here does prove something is wrong.

The default sampling is three whole episodes, several start frames each. The
per-chunk-position breakdown is the point of it: a policy that has learned to
copy `observation.state` instead of looking at the images gets the first few
steps nearly free and falls apart later in the chunk, which shows up as a steep
early-to-late gradient in the reported table.

Example:

    python scripts/inspect_checkpoint_actions.py \
        --checkpoint outputs/smolvla_ta2_multi_run06/checkpoints/last/pretrained_model \
        --episodes 3 --starts-per-episode 3
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from lerobot.configs.train import TrainPipelineConfig
from lerobot.constants import ACTION
from lerobot.datasets.factory import make_dataset, split_train_val_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from lerobot.datasets.utils_must import (
    EPISODES_DATASET_MAPPING,
    multidataset_collate_fn,
    true_action_dim,
)
from lerobot.policies.factory import make_policy
from lerobot.utils.utils import get_safe_torch_device, init_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to a `pretrained_model` directory (config.json + model.safetensors + train_config.json).",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="How many episodes to sample. Default 3.",
    )
    p.add_argument(
        "--starts-per-episode",
        type=int,
        default=3,
        help="How many chunk start frames to sample inside each episode. Default 3.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Seed for the episode/start draw and the flow noise.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where the per-sample CSVs go. Defaults to `<checkpoint>/action_inspect`.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Overrides the device recorded in train_config.",
    )
    p.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Overrides `dataset.root` from train_config, for when the checkpoint was trained elsewhere.",
    )
    p.add_argument(
        "--dataset-repo-ids",
        type=str,
        default=None,
        help="Comma-separated, overrides `dataset.repo_id` from train_config.",
    )
    p.add_argument(
        "--no-csv", action="store_true", help="Print the tables but do not write CSVs."
    )
    return p.parse_args(argv)


def build_configs(args: argparse.Namespace) -> tuple[TrainPipelineConfig, int]:
    """Load the checkpoint's own training config and point it back at the checkpoint.

    Returns the config plus the `val_episodes_per_dataset` it was trained with,
    which is needed later to report whether a sampled episode was actually seen.
    """
    cfg = TrainPipelineConfig.from_pretrained(args.checkpoint)
    from teleavatar_v2.smolvla_deploy.contracts import validate_state_layout

    validate_state_layout(getattr(cfg.policy, "teleavatar_state_layout", None))
    trained_val_episodes = cfg.dataset.val_episodes_per_dataset

    if args.dataset_root is not None:
        cfg.dataset.root = args.dataset_root
    if args.dataset_repo_ids is not None:
        cfg.dataset.repo_id = args.dataset_repo_ids
    if args.device is not None:
        cfg.policy.device = args.device

    # `make_dataset(split="train")` applies the random train-time augmentations.
    # Nothing is being trained here and the comparison has to be reproducible, so
    # they are switched off for both datasets built below.
    cfg.dataset.image_transforms.enable = False

    # The weights come from the checkpoint, not from the VLM hub snapshot; the
    # backbone only needs its architecture here.
    cfg.policy.pretrained_path = str(args.checkpoint)
    cfg.policy.load_vlm_weights = False
    return cfg, trained_val_episodes


def sub_datasets(dataset) -> list[tuple[str, LeRobotDataset, int]]:
    """Flatten to `(repo_id, sub_dataset, index_offset)` so both dataset classes look alike."""
    if isinstance(dataset, MultiLeRobotDataset):
        return [
            (repo_id, ds, int(dataset.cumulative_sizes[i]))
            for i, (repo_id, ds) in enumerate(
                zip(dataset.repo_ids, dataset._datasets, strict=True)
            )
        ]
    return [(dataset.repo_id, dataset, 0)]


def episode_index_ranges(ds: LeRobotDataset) -> dict[int, tuple[int, int]]:
    """Map each episode index to the half-open range of `ds[i]` indices that belong to it.

    `episode_data_index` is keyed by *position* within `ds.episodes`, not by the
    episode's own index, and `discard_first_n_frames` / `discard_first_idle_frames`
    insert a further remapping through `subset_frame_ids`. Both are resolved here
    so callers can work purely in `__getitem__` space.
    """
    ep_indices = (
        ds.episodes if ds.episodes is not None else list(ds.meta.episodes.keys())
    )
    raw_ranges = {
        ep: (
            int(ds.episode_data_index["from"][pos]),
            int(ds.episode_data_index["to"][pos]),
        )
        for pos, ep in enumerate(ep_indices)
    }

    subset = getattr(ds, "subset_frame_ids", None)
    if subset is None:
        return raw_ranges

    # Frames were dropped, so `__getitem__` indices are no longer contiguous per
    # episode in raw space. Rebuild the ranges in getitem space.
    raw_to_ep = {}
    for ep, (lo, hi) in raw_ranges.items():
        for raw in range(lo, hi):
            raw_to_ep[raw] = ep
    ranges: dict[int, tuple[int, int]] = {}
    for getitem_idx, raw in enumerate(subset):
        ep = raw_to_ep.get(int(raw))
        if ep is None:
            continue
        lo, hi = ranges.get(ep, (getitem_idx, getitem_idx))
        ranges[ep] = (min(lo, getitem_idx), max(hi, getitem_idx + 1))
    return ranges


def draw_samples(
    dataset, num_episodes: int, starts_per_episode: int, chunk_size: int, seed: int
):
    """Pick episodes uniformly across every sub-dataset, then start frames inside each.

    Start frames are spread over the episode rather than drawn independently, so
    three samples from one episode cannot all land in the same second of motion.
    """
    rng = random.Random(seed)

    candidates = []
    for repo_id, ds, offset in sub_datasets(dataset):
        for ep, (lo, hi) in episode_index_ranges(ds).items():
            if hi - lo < 2:
                continue
            candidates.append((repo_id, ep, offset + lo, offset + hi))

    if not candidates:
        raise RuntimeError("No episodes long enough to sample from.")
    if num_episodes > len(candidates):
        logging.warning(
            f"Asked for {num_episodes} episodes but only {len(candidates)} are available; using all of them."
        )
        num_episodes = len(candidates)

    samples = []
    for repo_id, ep, lo, hi in sorted(
        rng.sample(candidates, num_episodes), key=lambda c: (c[0], c[1])
    ):
        length = hi - lo
        # Leave the chunk room to run past the start where the episode allows it;
        # on a short episode fall back to the whole span and let `action_is_pad`
        # mask whatever runs off the end.
        last_start = max(1, length - chunk_size)
        n = min(starts_per_episode, last_start)
        # One start per equal-width bin rather than n independent draws, which
        # would happily put all three inside the same second of motion.
        edges = [(i * last_start) // n for i in range(n + 1)]
        offsets = [rng.randrange(edges[i], edges[i + 1]) for i in range(n)]
        for off in offsets:
            samples.append(
                {
                    "repo_id": repo_id,
                    "episode": ep,
                    "start": off,
                    "length": length,
                    "index": lo + off,
                }
            )
    return samples


# `meta.keys_to_max_dim` maps a key to a plain int, but `pad_tensor_to_shape`
# treats the value as a shape and reverses it. train.py wraps each one in a
# 1-tuple and keeps only the vector features; both dataloaders there do it, and
# this has to match or the padding is inconsistent with training.
PADDED_KEYS = ["action", "observation.state", "observation.environment_state"]


def collate(dataset, items: list[dict]) -> dict:
    if isinstance(dataset, MultiLeRobotDataset):
        keys_to_max_dim = {
            key: (max_dim,)
            for key, max_dim in dataset.meta.keys_to_max_dim.items()
            if max_dim is not None and key in PADDED_KEYS
        }
        return multidataset_collate_fn(items, keys_to_max_dim=keys_to_max_dim)
    return default_collate(items)


def action_names(dataset, action_dim: int) -> list[str]:
    """Joint names for the table, from whichever sub-dataset still carries them."""
    for _, ds, _ in sub_datasets(dataset):
        try:
            names = ds.meta.features[ACTION].get("names")
        except Exception:
            continue
        if names and len(names) >= action_dim:
            return [str(n) for n in names[:action_dim]]
    return [f"dim{d:02d}" for d in range(action_dim)]


def print_dim_table(
    names: list[str], mae: np.ndarray, max_err: np.ndarray, indent: str = "    "
) -> None:
    width = max(len(n) for n in names)
    print(f"{indent}{'dim':>3}  {'name':<{width}}  {'MAE':>10}  {'max':>10}")
    for d, name in enumerate(names):
        print(f"{indent}{d:>3}  {name:<{width}}  {mae[d]:>10.4f}  {max_err[d]:>10.4f}")


def print_position_breakdown(
    err: np.ndarray, valid: np.ndarray, bucket: int = 10, indent: str = "    "
) -> None:
    """`err` is (T, D) absolute error, `valid` is (T,) 1.0 for real timesteps."""
    horizon = err.shape[0]
    parts = []
    for lo in range(0, horizon, bucket):
        hi = min(lo + bucket, horizon)
        w = valid[lo:hi].sum()
        if w <= 0:
            continue
        parts.append(
            f"step{lo}-{hi - 1} {err[lo:hi].mean(axis=1) @ valid[lo:hi] / w:.4f}"
        )
    print(f"{indent}by chunk position: " + " | ".join(parts))


def main(argv: list[str] | None = None) -> int:
    init_logging()
    args = parse_args(argv)

    if not (args.checkpoint / "model.safetensors").is_file():
        logging.error(
            f"{args.checkpoint} does not look like a pretrained_model dir (no model.safetensors)."
        )
        return 1

    cfg, trained_val_episodes = build_configs(args)
    device = get_safe_torch_device(cfg.policy.device, log=True)

    # The policy's normalization buffers are rebuilt from dataset statistics at
    # construction time rather than restored from the checkpoint, so they have to
    # come from the split the checkpoint was trained on. Sampling, on the other
    # hand, was asked to cover every episode. When the checkpoint had no holdout
    # these are the same dataset and it is only built once.
    logging.info("Loading dataset (training split, for normalization statistics)")
    stats_dataset = make_dataset(cfg, split="train")

    if trained_val_episodes > 0:
        logging.info("Loading dataset (all episodes, for sampling)")
        cfg.dataset.val_episodes_per_dataset = 0
        sample_dataset = make_dataset(cfg, split="train")
        cfg.dataset.val_episodes_per_dataset = trained_val_episodes
    else:
        sample_dataset = stats_dataset

    logging.info("Loading policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=stats_dataset.meta)
    policy.eval()

    chunk_size = policy.config.chunk_size
    max_action_dim = policy.config.max_action_dim
    padded_action_dim = policy.config.action_feature.shape[0]
    action_dim = true_action_dim(sample_dataset, padded_action_dim)
    names = action_names(sample_dataset, action_dim)

    samples = draw_samples(
        sample_dataset, args.episodes, args.starts_per_episode, chunk_size, args.seed
    )

    # Which of the drawn episodes the checkpoint actually trained on. With no
    # holdout the answer is "all of them", and it is worth saying so out loud.
    seen: dict[tuple[str, int], bool] = {}
    is_multi = isinstance(sample_dataset, MultiLeRobotDataset)
    for repo_id, ds, _ in sub_datasets(sample_dataset):
        # `make_dataset` only consults the per-dataset episode mapping on the
        # multi-dataset path, so mirror that here or the reported split is wrong.
        base = (
            EPISODES_DATASET_MAPPING.get(repo_id, cfg.dataset.episodes)
            if is_multi
            else cfg.dataset.episodes
        )
        _, val_eps = split_train_val_episodes(
            base,
            ds.meta.total_episodes,
            trained_val_episodes,
            cfg.dataset.val_split_seed,
            repo_id,
        )
        held_out = set(val_eps)
        for ep in episode_index_ranges(ds):
            seen[(repo_id, ep)] = ep not in held_out

    out_dir = args.output_dir or (args.checkpoint / "action_inspect")
    if not args.no_csv:
        out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 78)
    print(f"checkpoint : {args.checkpoint}")
    print(
        f"chunk_size : {chunk_size}   action_dim: {action_dim}"
        f" (padded to {padded_action_dim})   device: {device}"
    )
    print(
        f"samples    : {args.episodes} episode(s) x {args.starts_per_episode} start(s), seed {args.seed}"
    )
    if trained_val_episodes > 0:
        print(
            f"holdout    : this checkpoint held out {trained_val_episodes} episode(s) per dataset"
        )
    else:
        print(
            "holdout    : none -- every sampled episode was in this checkpoint's training set,"
        )
        print("             so the numbers below measure fit, not generalization")
    print("=" * 78)

    all_err = []
    all_valid = []

    for s in samples:
        item = sample_dataset[s["index"]]
        batch = collate(sample_dataset, [item])
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        # Cloned because `select_action_chunk` normalizes `batch` in place; the
        # action key is an output feature and so is left alone today, but the
        # ground truth is the one thing here that must not be touched.
        gt = batch[ACTION].clone()  # (1, T, max_action_dim), never normalized
        horizon = gt.shape[1]

        # Fixed noise, so re-running this script on the same checkpoint reproduces
        # the same numbers. Flow matching integrates from noise, and an unseeded
        # draw moves the result more than a few thousand training steps do.
        generator = torch.Generator().manual_seed(args.seed + s["index"])
        noise = torch.randn(
            (1, chunk_size, max_action_dim), generator=generator, dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            pred = policy.select_action_chunk(
                batch, noise=noise
            )  # (1, chunk_size, action_dim), physical units

        steps = min(pred.shape[1], horizon)
        pred_np = pred[0, :steps, :action_dim].float().cpu().numpy()
        gt_np = gt[0, :steps, :action_dim].float().cpu().numpy()
        err = np.abs(pred_np - gt_np)

        is_pad = batch.get(f"{ACTION}_is_pad")
        if is_pad is None:
            valid = np.ones(steps, dtype=np.float32)
        else:
            valid = (~is_pad[0, :steps].to(torch.bool)).float().cpu().numpy()

        denom = max(valid.sum(), 1.0)
        mae = (err * valid[:, None]).sum(axis=0) / denom
        max_err = (err * valid[:, None]).max(axis=0)

        all_err.append(err * valid[:, None])
        all_valid.append(valid)

        tag = "train" if seen.get((s["repo_id"], s["episode"]), True) else "HELD OUT"
        print()
        print(
            f"[{s['repo_id']}] episode {s['episode']} (len={s['length']}, {tag})  "
            f"start={s['start']}  steps={steps}  valid={int(valid.sum())}"
        )
        print(f"    overall MAE: {mae.mean():.4f}")
        print_dim_table(names, mae, max_err)
        print_position_breakdown(err, valid)

        if not args.no_csv:
            csv_path = (
                out_dir
                / f"{s['repo_id']}_ep{s['episode']:04d}_start{s['start']:05d}.csv"
            )
            with csv_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    ["t", "is_valid"]
                    + [f"pred_{n}" for n in names]
                    + [f"gt_{n}" for n in names]
                    + [f"abserr_{n}" for n in names]
                )
                for t in range(steps):
                    w.writerow(
                        [t, int(valid[t])]
                        + [f"{v:.6f}" for v in pred_np[t]]
                        + [f"{v:.6f}" for v in gt_np[t]]
                        + [f"{v:.6f}" for v in err[t]]
                    )
            print(f"    -> {csv_path}")

    if all_err and len({e.shape for e in all_err}) > 1:
        logging.warning(
            "Chunk shapes differ across samples; skipping the aggregate table."
        )
    elif all_err:
        err_sum = np.stack(all_err).sum(axis=0)
        valid_sum = np.stack(all_valid).sum(axis=0)
        total = valid_sum.sum()
        mae = err_sum.sum(axis=0) / max(total, 1.0)
        max_err = np.stack(all_err).max(axis=(0, 1))
        print()
        print("=" * 78)
        print(f"AGGREGATE over {len(all_err)} chunk(s), {int(total)} valid timestep(s)")
        print(f"    overall MAE: {mae.mean():.4f}")
        print_dim_table(names, mae, max_err)
        print_position_breakdown(
            err_sum / np.maximum(valid_sum, 1.0)[:, None], np.minimum(valid_sum, 1.0)
        )
        print("=" * 78)

    return 0


if __name__ == "__main__":
    sys.exit(main())
