import os
from pathlib import Path

import torch


def updated_best_metrics(previous, epoch, metrics, train_acc):
    """Return one coherent best-epoch record on strict Fscd improvement."""
    if float(metrics["Fscd"]) <= float(previous.get("Fscd", float("-inf"))):
        return None
    return {
        "epoch": int(epoch),
        "Fscd": float(metrics["Fscd"]),
        "mIoU": float(metrics["mIoU"]),
        "Sek": float(metrics["Sek"]),
        "accuracy": float(metrics["accuracy"]),
        "loss": float(metrics["loss"]),
        "train_acc": float(train_acc),
    }


def _float_token(value):
    return format(float(value), ".10g")


def _safe_token(value):
    token = str(value)
    if not token or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-" for char in token):
        raise ValueError("Unsafe checkpoint token: {!r}".format(token))
    return token


def configuration_id(run_config):
    dataset = _safe_token(run_config["dataset"])
    mode = _safe_token(run_config["mode"])
    seed = int(run_config["seed"])
    if mode != "pseudo":
        return "{}_{}_seed{}".format(dataset, mode, seed)
    return (
        "{dataset}_{mode}_lam{lam}_a{alpha_start}-{alpha_end}_min{minimum}_"
        "thr{threshold}_start{start}_ramp{ramp}_ema{ema}_seed{seed}"
    ).format(
        dataset=dataset, mode=mode,
        lam=_float_token(run_config["lambda_sem_u"]),
        alpha_start=_float_token(run_config["pseudo_alpha_start"]),
        alpha_end=_float_token(run_config["pseudo_alpha_end"]),
        minimum=_float_token(run_config["pseudo_min_ratio"]),
        threshold=_float_token(run_config["pseudo_change_thr"]),
        start=int(run_config["pseudo_start_epoch"]),
        ramp=int(run_config["pseudo_rampup_epoch"]),
        ema=_float_token(run_config["ema_decay"]), seed=seed,
    )


def checkpoint_filename(run_config, epoch, metrics):
    return (
        "{identity}_e{epoch:03d}_mIoU{miou:.2f}_Sek{sek:.2f}_"
        "Fscd{fscd:.2f}_OA{oa:.2f}.pth"
    ).format(
        identity=configuration_id(run_config), epoch=int(epoch),
        miou=float(metrics["mIoU"]) * 100.0,
        sek=float(metrics["Sek"]) * 100.0,
        fscd=float(metrics["Fscd"]) * 100.0,
        oa=float(metrics["accuracy"]) * 100.0,
    )


def save_best_checkpoint(directory, run_config, epoch, metrics, payload, save_fn=torch.save):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    identity = configuration_id(run_config)
    final_path = directory / checkpoint_filename(run_config, epoch, metrics)
    temporary_path = directory / (final_path.name + ".tmp")
    previous_paths = list(directory.glob(identity + "_*.pth"))

    try:
        save_fn(payload, temporary_path)
        with temporary_path.open("rb") as checkpoint_file:
            os.fsync(checkpoint_file.fileno())
        os.replace(str(temporary_path), str(final_path))
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    for previous_path in previous_paths:
        if previous_path != final_path and previous_path.exists():
            previous_path.unlink()
    return final_path
