"""Single-process SmolVLA deployment for TeleAvatar V2."""

from .contracts import (
    ACTION_DIM,
    STATE_DIM,
    SMOLVLA_CAMERA_MAPPING,
    build_state,
    select_execution_actions,
    split_trigger_action,
)

__all__ = [
    "ACTION_DIM",
    "STATE_DIM",
    "SMOLVLA_CAMERA_MAPPING",
    "build_state",
    "select_execution_actions",
    "split_trigger_action",
]
