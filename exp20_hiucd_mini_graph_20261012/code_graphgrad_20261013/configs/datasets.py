import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: str
    num_classes: int
    classes: Tuple[str, ...]
    colormap: Tuple[Tuple[int, int, int], ...]


_CONFIGS = {
    "landsat": DatasetConfig(
        name="Landsat",
        root="datasets/Landsat/",
        num_classes=5,
        classes=("No change", "Farmland", "Desert", "Building", "Water"),
        colormap=(
            (255, 255, 255),
            (0, 155, 0),
            (255, 165, 0),
            (230, 30, 100),
            (0, 170, 240),
        ),
    ),
    "second": DatasetConfig(
        name="SECOND",
        root="datasets/SECOND/",
        num_classes=7,
        classes=(
            "unchanged",
            "water",
            "ground",
            "low vegetation",
            "tree",
            "building",
            "sports field",
        ),
        colormap=(
            (255, 255, 255),
            (0, 0, 255),
            (128, 128, 128),
            (0, 128, 0),
            (0, 255, 0),
            (128, 0, 0),
            (255, 0, 0),
        ),
    ),
}


def get_dataset_config(name: str) -> DatasetConfig:
    key = name.lower()
    if key not in _CONFIGS:
        raise ValueError("Unsupported dataset: {}".format(name))
    return _CONFIGS[key]


def validate_dataset_num_classes(config, num_classes) -> None:
    if num_classes != config.num_classes:
        raise ValueError(
            "num_classes={} does not match {}={}".format(
                num_classes, config.name, config.num_classes))


def validate_dataset_model_config(config, num_classes, output_root) -> None:
    validate_dataset_num_classes(config, num_classes)
    if Path(output_root).name != config.name:
        raise ValueError(
            "output directory {!s} does not match dataset {}".format(
                output_root, config.name))


@dataclass(frozen=True)
class TrainingConfig:
    dataset: DatasetConfig
    model_num_classes: int
    pred_dir: Path
    vis_dir: Path
    checkpoint_dir: Path
    log_dir: Path
    load_path: Path


def resolve_training_config(argv, working_path, model_num_classes, net_name="SAMNet"):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", choices=("landsat", "second"),
                        default="landsat")
    parser.add_argument("--num-classes", type=int, default=model_num_classes)
    selected, _ = parser.parse_known_args(argv)
    config = get_dataset_config(selected.dataset)
    validate_dataset_num_classes(config, selected.num_classes)
    base = Path(working_path)
    return TrainingConfig(
        dataset=config,
        model_num_classes=selected.num_classes,
        pred_dir=base / "results" / config.name,
        vis_dir=base / "visualizations" / config.name / net_name / "pseudo_train",
        checkpoint_dir=base / "checkpoints" / config.name,
        log_dir=base / "logs" / config.name / net_name,
        load_path=base / "checkpoints" / config.name,
    )
