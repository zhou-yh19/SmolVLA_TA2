# SmolVLA on TeleAvatar V2

This directory is a single-process deployment project for checkpoints trained
by `SmolVLA_TA2`. It contains only the SmolVLA runtime, TeleAvatar V2 hardware
interface, bring-up scripts, configuration, and offline contract tests.

## Repository layout

```text
teleavatar_v2/
├── README.md
├── LICENSE
├── environment.yml
├── arm_config.yml
├── smolvla_deploy/
│   ├── contracts.py
│   ├── policy_runtime.py
│   ├── robot_interface.py
│   ├── rtp_video_interface.py
│   └── runner.py
├── scripts/
│   ├── inspect_observation.py
│   ├── run_smolvla.py
│   ├── test_video.py
│   └── zero.py
└── tests/
    └── test_contracts.py
```

## Runtime contract

- State: 16 values: left arm 7, right arm 7, then measured left/right gripper
  positions.
- Cameras: head left eye, left-wrist left eye, right-wrist left eye, as cropped
  from the 1280x2720 RTP composite (head 960x960, wrists 400x640). Training
  frames are downscaled to these same sizes, and the runtime warns once per
  camera if a crop arrives at a different size than the checkpoint was trained at.
- Action: 16 absolute controls: left arm 7, left gripper trigger, right arm 7,
  right gripper trigger.
- Gripper outputs are continuous trigger values in `[0, 1]` after checkpoint
  unnormalization. They are clipped and published directly. There is no
  trigger/effort round trip.
- The policy returns a complete action chunk. The runner executes only the
  first `--execution-horizon` actions and then replans from a new observation.

## Data flow

```text
RTP/H.265 cameras + ROS2 joint states
                |
                v
TeleavatarSmolVLAInterface -- 14-D state + three RGB images
                |
                v
SmolVLARuntime -- complete [T,16] direct-trigger chunk
                |
                v
runner -- execute first H actions at the requested control frequency
                |
                v
/api/{left,right}_arm/joint_cmd + /api/{left,right}_gripper/cmd
```

No WebSocket server, OpenPI `ActionChunkBroker`, or SmolVLA internal action
queue is used.

## Prerequisites

1. Put the robot in API mode with both arms enabled in joint-control mode.
2. Point the robot RTP stream at this host (default UDP port 8890).
3. Start the zenoh bridge and use the same ROS domain in every terminal:

   ```bash
   export ROS_DOMAIN_ID=29
   export ROS_DISTRO=humble
   zenoh-bridge-ros2dds -e tcp/<ROBOT_IP>:9000
   ```

4. Use an environment containing ROS2 Humble, PyGObject/GStreamer, PyTorch,
   Transformers, Safetensors, NumPy, and PyYAML. The existing `environment.yml`
   is the combined robot/inference environment. It is a human-maintained list
   of direct dependencies, not a machine-specific `conda env export`.
5. Use the `pretrained_model` directory of a checkpoint. It must contain
   `config.json` and `model.safetensors`.
6. The configuration and tokenizer/processor named by `vlm_model_name` must
   already be cached on the deployment host. For an offline host, copy them
   beforehand and pass their directory through `--vlm-model-path`. Deployment
   does not preload the base VLM weight shards: the complete trained policy is
   restored from the checkpoint's `model.safetensors`.

Create the combined environment on the Linux deployment host with:

```bash
conda env create -n teleavatar-smolvla -f environment.yml
conda activate teleavatar-smolvla
```

The default PyTorch wheels use CUDA 12.8 because RTX 50-series GPUs require
`sm_120`. After creating or updating the environment, verify that Pip did not
silently choose another CUDA build:

```bash
python -m pip check
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA architectures:", torch.cuda.get_arch_list())
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA smoke test:", torch.randn(1, device="cuda"))
PY
```

The expected PyTorch version ends in `+cu128`, and RTX 50-series hosts must
show `sm_120`. To update an existing environment after editing the file:

```bash
conda env update -n teleavatar-smolvla -f environment.yml --prune
```

Do not regenerate `environment.yml` with `conda env export`. If an exact
machine snapshot is needed for debugging, write it to a separate ignored file,
for example `conda env export > environment.lock.local.yml`.

## Bring-up order

Measure inference latency before touching the robot. This needs no ROS, no
cameras and no arms: it loads the real checkpoint and feeds it random frames of
the shapes the RTP splitter would deliver, so only the timings mean anything —
the actions come from noise.

```bash
python scripts/bench_inference.py \
  --checkpoint /path/to/checkpoint/pretrained_model \
  --smolvla-repo /path/to/SmolVLA_TA2 \
  --vlm-model-path /path/to/SmolVLM2-500M-Video-Instruct \
  --device cuda \
  --iters 20
```

It reports the latency distribution and how much of the chunk budget
(`execution_horizon / control_frequency`) the p90 consumes. Add
`--profile-inference` to see where the time goes; on this policy essentially
all of it is `sample_actions`, which costs a fixed prefix pass plus
`--num-steps` denoising steps.

Check ROS topics:

```bash
ros2 topic echo /left_arm/joint_states --once
ros2 topic echo /right_arm/joint_states --once
```

Check RTP decoding and save all six crops:

```bash
python scripts/test_video.py --split-output-dir /tmp/teleavatar_cameras --duration-s 5
```

Inspect the exact observation sent to SmolVLA:

```bash
python scripts/inspect_observation.py
```

Zero both arms before policy control:

```bash
python scripts/zero.py
```

Run one inference chunk without publishing any command:

```bash
python scripts/run_smolvla.py \
  --checkpoint /path/to/checkpoint/pretrained_model \
  --smolvla-repo /path/to/SmolVLA_TA2 \
  --vlm-model-path /path/to/SmolVLM2-500M-Video-Instruct \
  --device cuda \
  --task "stack the three blocks"
```

To diagnose latency, run a dry pass with per-stage timing:

```bash
python scripts/run_smolvla.py \
  --checkpoint /path/to/checkpoint/pretrained_model \
  --smolvla-repo /path/to/SmolVLA_TA2 \
  --device cuda \
  --task "stack the three blocks" \
  --profile-inference
```

For speed/quality A/B tests, reduce the flow denoising steps from the
checkpoint default, usually 10:

```bash
python scripts/run_smolvla.py \
  --checkpoint /path/to/checkpoint/pretrained_model \
  --smolvla-repo /path/to/SmolVLA_TA2 \
  --device cuda \
  --task "stack the three blocks" \
  --num-steps 6
```

`bench_inference.py`, `run_smolvla.py` and every stage of
`scripts/deploy_small_ta2_robot.sh` share the same speed switches:

| flag | env var in the deploy script | default | effect |
| --- | --- | --- | --- |
| `--precision {fp32,bf16,fp16}` | `PRECISION` | `bf16` on CUDA | dtype the backbone is held in. The checkpoint is a silent fp32/bf16 mix; `fp32` leaves that mix in place, anything else makes it uniform. Largest single win. |
| `--attn-implementation {sdpa,eager}` | `ATTN_IMPLEMENTATION` | `sdpa` | `sdpa` uses the fused kernel; `eager` materializes the full score matrix and matches training exactly. |
| `--num-steps N` | `NUM_STEPS` | checkpoint value (10) | flow denoising steps. Time is roughly linear in this; quality is not — validate on hardware. |
| `--compile` | `COMPILE_MODEL=1` | off | `torch.compile` the denoise step. The first chunk pays compilation. |
| `--profile-inference` | `PROFILE_INFERENCE=1` | off | per-stage timing on every chunk. |
| `--iters N` | `BENCH_ITERS` | 20 | timed iterations, `bench` only. |

Only after validating the logged state, cameras, action shape, and trigger
range, enable execution explicitly:

```bash
python scripts/run_smolvla.py \
  --checkpoint /path/to/checkpoint/pretrained_model \
  --smolvla-repo /path/to/SmolVLA_TA2 \
  --device cuda \
  --task "stack the three blocks" \
  --control-frequency 20 \
  --execution-horizon 16 \
  --max-joint-step-rad 0.10 \
  --execute
```

example:
```bash
python scripts/run_smolvla.py \
  --checkpoint /home/new/checkpoint/smol_floor2_test/pretrained_model \
  --smolvla-repo /home/new/SmolVLA_TA2 \
  --vlm-model-path /home/new/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  --device cuda \
  --task "Stack the second layer of blocks." \
  --control-frequency 20 \
  --execution-horizon 16 \
  --max-joint-step-rad 0.05 \
  --execute
```

`--execute` is deliberately required for command publication. Stop with
Ctrl+C. Do not run `zero.py`, replay tools, OpenPI `main.py`, or another arm
controller at the same time, because they publish to the same command topics.

## Checkpoint checks

Checkpoints fine-tuned from `lerobot/smolvla_base` or
`lerobot/smolvla_robotwin` retain `type: smolvla`; checkpoints initialized
with this repository's scratch mode use `type: smolvla2`. The deployment
runtime accepts both types through the same policy implementation.

Startup fails before commanding the robot unless the checkpoint declares:

- state shape `(14,)`;
- action shape `(16,)`;
- all three canonical TeleAvatar camera keys;
- `n_obs_steps == 1`;
- absolute actions (`predict_relative_actions == false`);
- `adapt_to_pi_aloha == false`.

Deployment restores state/action normalization statistics from the same
`model.safetensors`. Startup fails before connecting to the robot if any
required statistic is missing, infinite, or NaN. During fine-tuning, finite
statistics supplied by the target dataset are preserved instead.

The runtime intentionally implements full-chunk inference itself. It does not
call the repository's currently broken `predict_action_chunk()` path and does
not use `select_action()`, whose internal queue would conflict with replanning
after the chosen execution horizon.

## Offline contract tests

From this directory in an environment with NumPy:

```bash
python -m unittest discover -s tests -v
```

These tests verify state order, all-left-eye mapping, direct-trigger semantics,
trigger clipping, and execution-horizon slicing. They do not connect to ROS2 or
the robot.
