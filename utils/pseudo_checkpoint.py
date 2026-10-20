from collections import OrderedDict
from collections.abc import Mapping

import torch


def _is_state_dict(value):
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(key, str) for key in value)
        and all(torch.is_tensor(tensor) for tensor in value.values())
    )


def _strip_module_prefix(state_dict):
    return OrderedDict(
        (key[7:] if key.startswith("module.") else key, value)
        for key, value in state_dict.items()
    )


def select_model_state(checkpoint, owner="teacher"):
    """Select a model state from structured or legacy checkpoints."""
    if owner not in ("teacher", "student"):
        raise ValueError("checkpoint owner must be 'teacher' or 'student'")

    if isinstance(checkpoint, Mapping) and "student" in checkpoint:
        selected = checkpoint.get(owner)
        if owner == "teacher" and selected is None:
            raise ValueError(
                "Checkpoint does not contain an EMA teacher; "
                "use --checkpoint-model student.")
        if not _is_state_dict(selected):
            raise ValueError(
                "Checkpoint {!r} entry is not a model state_dict.".format(owner))
        return _strip_module_prefix(selected), owner

    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model", "net"):
            if key in checkpoint:
                selected = checkpoint[key]
                if not _is_state_dict(selected):
                    raise ValueError(
                        "Checkpoint {!r} entry is not a model state_dict.".format(key))
                return _strip_module_prefix(selected), key

    if _is_state_dict(checkpoint):
        return _strip_module_prefix(checkpoint), "legacy"

    raise ValueError(
        "Unsupported checkpoint format: expected training state or model state_dict.")


def infer_semantic_num_classes(state_dict):
    for key in ("head.head1.2.weight", "head.head2.2.weight"):
        value = state_dict.get(key)
        if torch.is_tensor(value) and value.ndim == 4:
            return int(value.shape[0])
    return None


def validate_checkpoint_num_classes(state_dict, expected, dataset):
    actual = infer_semantic_num_classes(state_dict)
    if actual is None:
        raise ValueError(
            "Cannot infer semantic class count from checkpoint head weights.")
    if actual != int(expected):
        raise ValueError(
            "Checkpoint has {} semantic classes, but {} expects {}.".format(
                actual, dataset, expected))
