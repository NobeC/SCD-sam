from pathlib import Path

import pytest

from utils.checkpoint_manager import (
    checkpoint_filename,
    configuration_id,
    save_best_checkpoint,
)


def pseudo_config(lambda_sem_u=0.2, alpha_start=0.1, seed=42):
    return {
        "dataset": "Landsat",
        "mode": "pseudo",
        "seed": seed,
        "lambda_sem_u": lambda_sem_u,
        "pseudo_alpha_start": alpha_start,
        "pseudo_alpha_end": 0.3,
        "pseudo_min_ratio": 0.02,
        "pseudo_change_thr": 0.3,
        "pseudo_start_epoch": 30,
        "pseudo_rampup_epoch": 10,
        "ema_decay": 0.999,
    }


def metrics(fscd=0.8347):
    return {
        "mIoU": 0.8481,
        "Sek": 0.4842,
        "Fscd": fscd,
        "accuracy": 0.9421,
        "loss": 0.13,
    }


def write_payload(payload, path):
    Path(path).write_text(str(payload))


def test_pseudo_configuration_id_contains_all_varying_parameters():
    assert configuration_id(pseudo_config()) == (
        "Landsat_pseudo_lam0.2_a0.1-0.3_min0.02_thr0.3_"
        "start30_ramp10_ema0.999_seed42"
    )


def test_filename_contains_epoch_and_all_validation_metrics():
    name = checkpoint_filename(pseudo_config(), 98, metrics())
    assert name.endswith(
        "_e098_mIoU84.81_Sek48.42_Fscd83.47_OA94.21.pth"
    )


def test_higher_fscd_replaces_only_same_configuration(tmp_path):
    config = pseudo_config()
    old = save_best_checkpoint(
        tmp_path, config, 90, metrics(0.82), {"epoch": 90}, save_fn=write_payload)
    other = save_best_checkpoint(
        tmp_path, pseudo_config(lambda_sem_u=0.3), 90, metrics(0.81),
        {"epoch": 90}, save_fn=write_payload)

    new = save_best_checkpoint(
        tmp_path, config, 98, metrics(0.8347), {"epoch": 98}, save_fn=write_payload)

    assert new.exists()
    assert not old.exists()
    assert other.exists()
    assert len(list(tmp_path.glob(configuration_id(config) + "_*.pth"))) == 1


def test_failed_save_preserves_previous_best(tmp_path):
    config = pseudo_config()
    old = save_best_checkpoint(
        tmp_path, config, 90, metrics(0.82), {"epoch": 90}, save_fn=write_payload)

    def fail_save(payload, path):
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        save_best_checkpoint(
            tmp_path, config, 98, metrics(0.8347), {"epoch": 98}, save_fn=fail_save)

    assert old.exists()
    assert list(tmp_path.glob("*.tmp")) == []
