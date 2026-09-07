#!/usr/bin/env python3

"""Compute SmolVLA normalization statistics for a raw TeleAvatar V2 dataset.

This scans only Parquet state/action columns. It does not decode videos and it
does not modify the source LeRobot metadata. The generated sidecar is required
by the TeleAvatar V2 adapter because gripper effort-to-trigger conversion is
piecewise and cannot be reconstructed from the raw mean/std alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.datasets.adapters.teleavatar import (  # noqa: E402
    ACTION,
    OBS_STATE,
    TELEAVATAR_ACTION_INDICES,
    TELEAVATAR_IMAGE_KEY_MAPPING,
    TELEAVATAR_STATE_INDICES,
    TELEAVATAR_V2_STATS_PATH,
    TeleavatarV2Adapter,
    adapt_teleavatar_action,
    adapt_teleavatar_state,
)


class RunningFeatureStats:
    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.count = 0
        self.total = np.zeros(feature_dim, dtype=np.float64)
        self.total_squared = np.zeros(feature_dim, dtype=np.float64)
        self.minimum = np.full(feature_dim, np.inf, dtype=np.float64)
        self.maximum = np.full(feature_dim, -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[None]
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected values with shape [N, {self.feature_dim}], got {values.shape}."
            )
        if values.shape[0] == 0:
            return
        if not np.isfinite(values).all():
            raise ValueError(
                "TeleAvatar V2 state/action data contains NaN or infinite values."
            )
        self.count += values.shape[0]
        self.total += values.sum(axis=0)
        self.total_squared += np.square(values).sum(axis=0)
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

    def finalize(self) -> dict[str, np.ndarray]:
        if self.count == 0:
            raise ValueError("Cannot finalize empty statistics.")
        mean = self.total / self.count
        variance = np.maximum(self.total_squared / self.count - np.square(mean), 0.0)
        return {
            "min": self.minimum.astype(np.float32),
            "max": self.maximum.astype(np.float32),
            "mean": mean.astype(np.float32),
            "std": np.sqrt(variance).astype(np.float32),
            "count": np.asarray([self.count], dtype=np.int64),
        }


def _new_episode_accumulators() -> dict[str, RunningFeatureStats]:
    return {
        OBS_STATE: RunningFeatureStats(len(TELEAVATAR_STATE_INDICES)),
        ACTION: RunningFeatureStats(len(TELEAVATAR_ACTION_INDICES)),
    }


def _serialize_stats(stats: dict[str, dict[str, np.ndarray]]) -> dict:
    return {
        feature: {name: value.tolist() for name, value in feature_stats.items()}
        for feature, feature_stats in stats.items()
    }


def _finalize(accumulators: dict[str, RunningFeatureStats]) -> dict:
    return {
        feature: accumulator.finalize() for feature, accumulator in accumulators.items()
    }


def compute_stats(dataset_root: Path, batch_size: int) -> dict:
    info_path = dataset_root / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    info = json.loads(info_path.read_text())
    if info.get("robot_type") != TeleavatarV2Adapter.robot_type:
        raise ValueError(
            f"Expected robot_type={TeleavatarV2Adapter.robot_type!r}, got {info.get('robot_type')!r}."
        )
    TeleavatarV2Adapter().validate_source_features(info["features"])

    parquet_files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found below {dataset_root / 'data'}."
        )

    global_accumulators = _new_episode_accumulators()
    episode_accumulators: dict[int, dict[str, RunningFeatureStats]] = {}
    required_columns = [OBS_STATE, ACTION, "episode_index"]

    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        missing = [
            column
            for column in required_columns
            if column not in parquet_file.schema_arrow.names
        ]
        if missing:
            raise KeyError(f"{parquet_path} is missing required columns {missing}.")

        for batch in parquet_file.iter_batches(
            batch_size=batch_size, columns=required_columns
        ):
            data = batch.to_pydict()
            raw_state = np.asarray(data[OBS_STATE], dtype=np.float32)
            raw_action = np.asarray(data[ACTION], dtype=np.float32)
            episode_indices = np.asarray(data["episode_index"], dtype=np.int64).reshape(
                -1
            )
            state = adapt_teleavatar_state(raw_state)
            action = adapt_teleavatar_action(raw_action)

            global_accumulators[OBS_STATE].update(state)
            global_accumulators[ACTION].update(action)

            for episode_index in np.unique(episode_indices):
                mask = episode_indices == episode_index
                accumulators = episode_accumulators.setdefault(
                    int(episode_index), _new_episode_accumulators()
                )
                accumulators[OBS_STATE].update(state[mask])
                accumulators[ACTION].update(action[mask])

    global_stats = _finalize(global_accumulators)
    episodes = {
        str(episode_index): _serialize_stats(_finalize(accumulators))
        for episode_index, accumulators in sorted(episode_accumulators.items())
    }
    return {
        "version": 1,
        "camera_eye": "left",
        "state_indices": list(TELEAVATAR_STATE_INDICES),
        "action_indices": list(TELEAVATAR_ACTION_INDICES),
        "image_key_mapping": TELEAVATAR_IMAGE_KEY_MAPPING,
        "stats": _serialize_stats(global_stats),
        "episodes": episodes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Root of the raw TA2 LeRobot dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output JSON path (default: <dataset>/{TELEAVATAR_V2_STATS_PATH}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8192, help="Parquet rows processed per batch."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset.expanduser().resolve()
    output_path = args.output or dataset_root / TELEAVATAR_V2_STATS_PATH
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    payload = compute_stats(dataset_root, args.batch_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n")
    temporary_path.replace(output_path)

    count = payload["stats"][OBS_STATE]["count"][0]
    print(f"Wrote adapted TA2 statistics for {count} frames to {output_path}")
    print("state shape: 16; action shape: 16; cameras: head/left/right left eye")


if __name__ == "__main__":
    main()
