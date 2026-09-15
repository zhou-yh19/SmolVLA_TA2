"""Parallel-in-time sampler used by StreamTP inference.

This module is deliberately independent of the SmolVLA network.  The caller
provides a batched velocity function; keeping the numerical iteration here
makes it possible to test exact fallback and warm-start semantics without a
checkpoint or a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class StreamTPConfig:
    """Runtime-only controls for the paper's StreamTP sampler."""

    tolerance: float = 0.02
    max_sweeps: int = 10
    anderson_depth: int = 3
    anderson_regularization: float = 1e-4
    warm_start: bool = True
    shift_steps: int = 1

    def validate(self) -> None:
        if self.tolerance <= 0:
            raise ValueError("StreamTP tolerance must be positive")
        if self.max_sweeps <= 0:
            raise ValueError("StreamTP max_sweeps must be positive")
        if self.anderson_depth < 0:
            raise ValueError("StreamTP anderson_depth cannot be negative")
        if self.anderson_regularization < 0:
            raise ValueError("StreamTP anderson_regularization cannot be negative")
        if self.shift_steps < 0:
            raise ValueError("StreamTP shift_steps cannot be negative")


def shift_action_chunk(actions: Tensor, steps: int) -> Tensor:
    """Drop executed actions and repeat the final action to restore the horizon."""
    if actions.ndim < 2:
        raise ValueError(f"Expected [batch, horizon, ...] actions, got {tuple(actions.shape)}")
    if steps <= 0:
        return actions.clone()
    horizon = actions.shape[1]
    if horizon == 0:
        raise ValueError("Cannot shift an empty action chunk")
    kept = actions[:, min(steps, horizon) :]
    tail = actions[:, -1:].expand(-1, min(steps, horizon), *([-1] * (actions.ndim - 2)))
    if steps >= horizon:
        return actions[:, -1:].expand_as(actions).clone()
    return torch.cat((kept, tail), dim=1)


def _rms_per_batch(value: Tensor) -> Tensor:
    return value.float().square().flatten(start_dim=1).mean(dim=1).sqrt()


def _synchronize(value: Tensor, enabled: bool) -> None:
    if enabled and value.device.type == "cuda":
        torch.cuda.synchronize(value.device)


def _stock_euler_times(num_steps: int, device: torch.device) -> Tensor:
    """Build the grid with the same float32 accumulation as stock SmolVLA."""
    dt = float(torch.tensor(-1.0 / num_steps, dtype=torch.float32))
    current = torch.tensor(1.0, dtype=torch.float32)
    values = []
    for _ in range(num_steps):
        values.append(float(current))
        current += dt
    return torch.tensor(values, dtype=torch.float32, device=device)


def _anderson_type2(
    gs: list[Tensor], residuals: list[Tensor], depth: int, regularization: float
) -> Tensor:
    """Return a depth-limited Type-II Anderson proposal for the latest iterate."""
    columns = min(depth, len(residuals) - 1)
    if columns <= 0:
        return gs[-1]

    recent_f = residuals[-(columns + 1) :]
    recent_g = gs[-(columns + 1) :]
    delta_f = torch.stack(
        [(recent_f[index + 1] - recent_f[index]).flatten(start_dim=1) for index in range(columns)],
        dim=2,
    )
    delta_g = torch.stack(
        [(recent_g[index + 1] - recent_g[index]).flatten(start_dim=1) for index in range(columns)],
        dim=2,
    )
    current_f = residuals[-1].flatten(start_dim=1).unsqueeze(2)
    gram = delta_f.transpose(1, 2) @ delta_f
    if regularization:
        identity = torch.eye(columns, dtype=gram.dtype, device=gram.device).expand(gram.shape[0], -1, -1)
        gram = gram + regularization * identity
    rhs = delta_f.transpose(1, 2) @ current_f
    try:
        coefficients = torch.linalg.solve(gram, rhs)
    except RuntimeError:
        # A singular history should never take down control; a plain Picard
        # proposal retains the finite-propagation property.
        return gs[-1]
    correction = (delta_g @ coefficients).squeeze(2).reshape_as(gs[-1])
    return gs[-1] - correction


def streamtp_solve(
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
    noise: Tensor,
    *,
    num_steps: int,
    config: StreamTPConfig,
    previous_actions: Tensor | None = None,
    collect_timing: bool = False,
    residual_dim: int | None = None,
    fallback_fn: Callable[[Tensor], Tensor] | None = None,
) -> tuple[Tensor, dict[str, object]]:
    """Solve the deployed Euler grid with residual-governed parallel sweeps.

    ``velocity_fn`` receives the K trajectory points folded into its batch
    dimension and the matching time vector.  Times follow SmolVLA's convention:
    source noise is at t=1 and actions are at t=0.
    """
    config.validate()
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if noise.ndim < 2:
        raise ValueError(f"Expected batched noise, got {tuple(noise.shape)}")
    if previous_actions is not None and previous_actions.shape != noise.shape:
        raise ValueError(
            f"Previous action shape {tuple(previous_actions.shape)} does not match noise {tuple(noise.shape)}"
        )
    if residual_dim is not None and not 0 < residual_dim <= noise.shape[-1]:
        raise ValueError(f"residual_dim must be in [1, {noise.shape[-1]}], got {residual_dim}")

    batch_size = noise.shape[0]
    dt = float(torch.tensor(-1.0 / num_steps, dtype=torch.float32))
    # X stores x_1..x_K.  x_0 is always the fresh noise and is prepended only
    # when evaluating the field.
    warm = None
    if config.warm_start and previous_actions is not None:
        warm = shift_action_chunk(previous_actions, config.shift_steps)
        alpha = torch.arange(1, num_steps + 1, dtype=torch.float32, device=noise.device)
        alpha = (alpha / num_steps).reshape(1, num_steps, *([1] * (noise.ndim - 1)))
        trajectory = (1.0 - alpha) * noise[:, None] + alpha * warm[:, None]
        initialization = "warm"
    else:
        trajectory = noise[:, None].expand(-1, num_steps, *([-1] * (noise.ndim - 1))).clone()
        initialization = "cold"

    times = _stock_euler_times(num_steps, noise.device)
    times = times[None, :].expand(batch_size, -1).reshape(-1)
    gs: list[Tensor] = []
    residual_history: list[Tensor] = []
    previous_full_residual: float | None = None
    sweep_times_ms: list[float] = []
    anderson_ms = 0.0
    final_endpoint_residual = float("inf")

    for sweep in range(1, config.max_sweeps + 1):
        evaluation_points = torch.cat((noise[:, None], trajectory[:, :-1]), dim=1)
        _synchronize(noise, collect_timing)
        started = time.perf_counter()
        velocities = velocity_fn(
            evaluation_points.reshape(batch_size * num_steps, *noise.shape[1:]), times
        ).reshape(batch_size, num_steps, *noise.shape[1:])
        proposal = noise[:, None] + dt * torch.cumsum(velocities.float(), dim=1)
        _synchronize(noise, collect_timing)
        if collect_timing:
            sweep_times_ms.append((time.perf_counter() - started) * 1000.0)

        residual = proposal - trajectory
        scored_residual = residual if residual_dim is None else residual[..., :residual_dim]
        endpoint_residuals = _rms_per_batch(scored_residual[:, -1])
        full_residuals = _rms_per_batch(scored_residual)
        # The scalar check is the intended synchronization point between
        # sequential sweeps.  A batch is accepted only when every item passes.
        final_endpoint_residual = float(endpoint_residuals.max().item())
        full_residual = float(full_residuals.max().item())
        if final_endpoint_residual < config.tolerance:
            endpoint = proposal[:, -1]
            plan_shift = None
            if warm is not None:
                difference = endpoint - warm
                if residual_dim is not None:
                    difference = difference[..., :residual_dim]
                plan_shift = float(_rms_per_batch(difference).max().item())
            return endpoint, {
                "accepted": True,
                "fallback": False,
                "initialization": initialization,
                "sweeps": sweep,
                "critical_nfe": sweep,
                "expert_evaluations": sweep * num_steps,
                "final_residual_rms": final_endpoint_residual,
                "full_residual_rms": full_residual,
                "sweep_times_ms": sweep_times_ms,
                "anderson_ms": anderson_ms,
                "fallback_ms": 0.0,
                "warm_plan_shift_rms": plan_shift,
            }

        grew = previous_full_residual is not None and full_residual > previous_full_residual
        if grew:
            gs.clear()
            residual_history.clear()
        gs.append(proposal)
        residual_history.append(scored_residual)
        history_limit = config.anderson_depth + 1
        if len(gs) > history_limit:
            del gs[:-history_limit]
            del residual_history[:-history_limit]
        if config.anderson_depth and not grew:
            _synchronize(noise, collect_timing)
            started = time.perf_counter()
            trajectory = _anderson_type2(
                gs, residual_history, config.anderson_depth, config.anderson_regularization
            )
            _synchronize(noise, collect_timing)
            if collect_timing:
                anderson_ms += (time.perf_counter() - started) * 1000.0
        else:
            trajectory = proposal
        previous_full_residual = full_residual

    # Exact fallback: discard the accelerated iterate and reproduce the stock
    # sequential Euler sampler from the same fresh noise and time grid.
    _synchronize(noise, collect_timing)
    fallback_started = time.perf_counter()
    if fallback_fn is not None:
        endpoint = fallback_fn(noise.clone())
    else:
        endpoint = noise.clone()
        single_times = _stock_euler_times(num_steps, noise.device)
        for timestep in single_times:
            expanded_time = timestep.expand(batch_size)
            endpoint = endpoint + dt * velocity_fn(endpoint, expanded_time)
    _synchronize(noise, collect_timing)
    fallback_ms = (time.perf_counter() - fallback_started) * 1000.0 if collect_timing else 0.0
    plan_shift = None
    if warm is not None:
        difference = endpoint - warm
        if residual_dim is not None:
            difference = difference[..., :residual_dim]
        plan_shift = float(_rms_per_batch(difference).max().item())
    return endpoint, {
        "accepted": False,
        "fallback": True,
        "initialization": initialization,
        "sweeps": config.max_sweeps,
        "critical_nfe": config.max_sweeps + num_steps,
        "expert_evaluations": config.max_sweeps * num_steps + num_steps,
        "final_residual_rms": final_endpoint_residual,
        "full_residual_rms": previous_full_residual,
        "sweep_times_ms": sweep_times_ms,
        "anderson_ms": anderson_ms,
        "fallback_ms": fallback_ms,
        "warm_plan_shift_rms": plan_shift,
    }
