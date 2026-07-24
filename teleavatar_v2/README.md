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

- State: 14 absolute arm positions, left 7 followed by right 7.
- Cameras: head left eye, left-wrist left eye, right-wrist left eye.
- Action: 16 absolute controls: left arm 7, left gripper trigger, right arm 7,
  right gripper trigger.
- Gripper outputs are already trigger values in `[0, 1]` after checkpoint
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
   is the combined robot/inference environment.
5. Use the `pretrained_model` directory of a checkpoint. It must contain
   `config.json` and `model.safetensors`.
6. The tokenizer/processor named by `vlm_model_name` must already be cached on
   the deployment host. For an offline host, download it beforehand and pass
   its directory through `--vlm-model-path`.

Create the combined environment on the Linux deployment host with:

```bash
conda env create -n teleavatar-smolvla -f environment.yml
conda activate teleavatar-smolvla
```

## Bring-up order

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

`--execute` is deliberately required for command publication. Stop with
Ctrl+C. Do not run `zero.py`, replay tools, OpenPI `main.py`, or another arm
controller at the same time, because they publish to the same command topics.

## Checkpoint checks

Startup fails before commanding the robot unless the checkpoint declares:

- state shape `(14,)`;
- action shape `(16,)`;
- all three canonical TeleAvatar camera keys;
- `n_obs_steps == 1`;
- absolute actions (`predict_relative_actions == false`);
- `adapt_to_pi_aloha == false`.

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
