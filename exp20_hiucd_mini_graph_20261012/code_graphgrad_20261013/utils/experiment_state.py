from dataclasses import dataclass
import random
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class ResumeState:
    epoch: int
    optimizer_step: int
    class_precision: torch.Tensor
    best_metrics: Dict[str, float]
    run_config: Dict[str, Any]


def _scheduler_state(scheduler):
    if scheduler is None:
        return None
    if hasattr(scheduler, "state_dict"):
        return scheduler.state_dict()
    return {
        key: value
        for key, value in scheduler.__dict__.items()
        if key != "optimizer"
    }


def _load_scheduler_state(scheduler, state):
    if scheduler is None or state is None:
        return
    if hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(state)
        return
    for key, value in state.items():
        setattr(scheduler, key, value)


def build_training_state(
    student,
    teacher,
    optimizer,
    scheduler,
    epoch,
    optimizer_step,
    class_precision,
    best_metrics,
    run_config,
):
    config = dict(run_config)
    return {
        "format_version": 1,
        "student": student.state_dict(),
        "teacher": teacher.state_dict() if teacher is not None else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": _scheduler_state(scheduler),
        "epoch": int(epoch),
        "optimizer_step": int(optimizer_step),
        "class_precision": class_precision.detach().cpu().clone(),
        "best_metrics": dict(best_metrics),
        "run_config": config,
        "dataset": config.get("dataset"),
        "seed": config.get("seed"),
        "mode": config.get("mode"),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def _validate_compatibility(payload, active_run_config):
    checkpoint_config = payload.get("run_config", {})
    if active_run_config is None:
        return
    for key in ("dataset", "num_classes"):
        checkpoint_value = checkpoint_config.get(key)
        active_value = active_run_config.get(key)
        if checkpoint_value is not None and active_value is not None:
            if str(checkpoint_value).lower() != str(active_value).lower():
                raise ValueError(
                    "Checkpoint {}={!r} does not match active {}={!r}".format(
                        key, checkpoint_value, key, active_value
                    )
                )


def _validate_finite_state_dict(state_dict, owner):
    for name, value in state_dict.items():
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
            if not torch.isfinite(value).all():
                raise FloatingPointError(
                    "Non-finite tensor in {} state_dict: {}".format(owner, name)
                )


def restore_training_state(
    payload,
    student,
    teacher,
    optimizer,
    scheduler,
    active_run_config: Optional[Mapping[str, Any]] = None,
):
    _validate_compatibility(payload, active_run_config)
    if "student" not in payload:
        raise ValueError("Checkpoint does not contain complete training state")
    if payload.get("teacher") is not None and teacher is None:
        raise ValueError("Checkpoint contains an EMA teacher but the active run does not")

    _validate_finite_state_dict(payload["student"], "student")
    if payload.get("teacher") is not None:
        _validate_finite_state_dict(payload["teacher"], "teacher")

    student.load_state_dict(payload["student"], strict=True)
    if teacher is not None:
        teacher_state = (
            payload["teacher"]
            if payload.get("teacher") is not None
            else payload["student"]
        )
        teacher.load_state_dict(teacher_state, strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    _load_scheduler_state(scheduler, payload.get("scheduler"))

    rng_state = payload.get("rng_state", {})
    if rng_state.get("python") is not None:
        random.setstate(rng_state["python"])
    if rng_state.get("numpy") is not None:
        np.random.set_state(rng_state["numpy"])
    if rng_state.get("torch") is not None:
        torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        visible_cuda_states = rng_state["cuda"][:torch.cuda.device_count()]
        torch.cuda.set_rng_state_all(visible_cuda_states)

    device = next(student.parameters()).device
    return ResumeState(
        epoch=int(payload["epoch"]),
        optimizer_step=int(payload["optimizer_step"]),
        class_precision=payload["class_precision"].to(device=device),
        best_metrics=dict(payload.get("best_metrics", {})),
        run_config=dict(payload.get("run_config", {})),
    )
