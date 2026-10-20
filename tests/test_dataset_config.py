import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from configs.datasets import (
    get_dataset_config,
    resolve_training_config,
    validate_dataset_model_config,
)


def test_landsat_config_is_self_consistent(tmp_path):
    config = get_dataset_config("landsat")
    assert config.name == "Landsat"
    assert config.num_classes == 5
    assert config.root == "datasets/Landsat/"
    assert len(config.classes) == 5
    assert len(config.colormap) == 5
    validate_dataset_model_config(
        config, num_classes=5, output_root=tmp_path / "Landsat")


def test_second_config_is_self_consistent(tmp_path):
    config = get_dataset_config("second")
    assert config.name == "SECOND"
    assert config.num_classes == 7
    assert config.root == "datasets/SECOND/"
    assert len(config.classes) == 7
    assert len(config.colormap) == 7
    validate_dataset_model_config(
        config, num_classes=7, output_root=tmp_path / "SECOND")


def test_dataset_mismatch_fails_before_training(tmp_path):
    config = get_dataset_config("second")
    with pytest.raises(ValueError, match="num_classes"):
        validate_dataset_model_config(
            config, num_classes=5, output_root=tmp_path / "SECOND")
    with pytest.raises(ValueError, match="output"):
        validate_dataset_model_config(
            config, num_classes=7, output_root=tmp_path / "Landsat")


def test_selected_config_controls_dataset_metadata_without_leaking_state():
    script = """
import os
os.environ["SCD_DATASET"] = "second"
import datasets.RS_ST as rs
assert rs.num_classes == 7
assert rs.root == "datasets/SECOND/"
assert rs.ST_CLASSES[0] == "unchanged"
"""
    env = os.environ.copy()
    env.pop("SCD_DATASET", None)
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


def test_selected_config_test_leaves_default_import_unchanged(monkeypatch):
    monkeypatch.delenv("SCD_DATASET", raising=False)
    import datasets.RS_ST as rs

    rs = importlib.reload(rs)
    assert rs.num_classes == 5
    assert rs.root == "datasets/Landsat/"


def test_public_second_selection_resolves_metadata_model_and_output_paths(tmp_path):
    resolved = resolve_training_config(
        ["--dataset", "second"],
        working_path=tmp_path,
        model_num_classes=7,
    )

    assert resolved.dataset.name == "SECOND"
    assert resolved.model_num_classes == 7
    assert resolved.pred_dir == tmp_path / "results" / "SECOND"
    assert resolved.checkpoint_dir == tmp_path / "checkpoints" / "SECOND"
    assert resolved.log_dir == tmp_path / "logs" / "SECOND" / "SAMNet"
    assert resolved.load_path == tmp_path / "checkpoints" / "SECOND"


def test_public_selection_rejects_independent_model_class_mismatch(tmp_path):
    with pytest.raises(ValueError, match="num_classes"):
        resolve_training_config(
            ["--dataset", "second"],
            working_path=tmp_path,
            model_num_classes=5,
        )


def test_train_consumes_resolved_config_without_private_dataset_state():
    source = (Path(__file__).parents[1] / "train.py").read_text()
    assert "resolve_training_config" in source
    assert "RS._CONFIG" not in source
    assert "training_config.model_num_classes" in source
    assert "training_config.checkpoint_dir" in source


def test_public_second_selection_rejects_explicit_cli_class_mismatch(tmp_path):
    with pytest.raises(ValueError, match="num_classes"):
        resolve_training_config(
            ["--dataset", "second", "--num-classes", "5"],
            working_path=tmp_path,
            model_num_classes=7,
        )


def test_main_validates_parsed_class_count_before_model_creation():
    source = (Path(__file__).parents[1] / "train.py").read_text()
    main_source = source.split("def main(opt):", 1)[1].split("def train(", 1)[0]
    assert "model_num_classes=opt.num_classes" in main_source
    assert "opt.num_classes = training_config.model_num_classes" not in main_source
    assert main_source.index("resolve_training_config") < main_source.index("ChangeDetection(opt)")
