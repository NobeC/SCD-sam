import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/homenew/yang/SCD/SCD-SAM/SCD")
EXPERIMENT = ROOT / "exp22_second_graph_weight_ablation_20261018"
PYTHON = Path("/home/homenew/yang/anaconda3/envs/scd-sam-cuda12/bin/python")
CHECKPOINT = (
    ROOT
    / "exp21_second_graph_pseudo_complement_alpha01_03_20261016"
    / "output/checkpoints/SECOND"
    / "SECOND_pseudo_graph_stage2_seed42_epoch054_Fscd0.5961.pth"
)
BRANCHES = {"control": 0.0, "graph02": 0.02, "graph05": 0.05, "smoke02": 0.02}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_branch(branch, weight):
    if branch not in BRANCHES:
        raise ValueError("unknown branch: {}".format(branch))
    if weight < 0:
        raise ValueError("graph pseudo-label weight must be non-negative")
    expected = BRANCHES[branch]
    if abs(float(weight) - expected) > 1e-12:
        raise ValueError(
            "branch {} requires weight {}, got {}".format(branch, expected, weight)
        )


def build_arguments(branch, weight, smoke_steps=0):
    validate_branch(branch, weight)
    arguments = [
        str(PYTHON), "-u", str(EXPERIMENT / "train_ablation.py"),
        "--dataset", "second", "--num-classes", "7", "--cuda", "0",
        "--epochs", "70", "--seed", "42", "--learning-rate", "3.5e-4",
        "--batch-size", "2", "--effective-batch-size", "8",
        "--val-batch-size", "2", "--input-size", "512", "--num-workers", "4",
        "--grad-clip", "1.0", "--timing-interval", "20", "--ema-decay", "0.99",
        "--lambda-sem-sup", "1.0", "--lambda-sem-u", "0.10",
        "--pseudo-start-epoch", "40", "--pseudo-rampup-epoch", "5",
        "--pseudo-alpha-start", "0.10", "--pseudo-alpha-end", "0.30",
        "--pseudo-min-ratio", "0.0", "--pseudo-change-thr", "0.30",
        "--pseudo-ignore-index", "255", "--pseudo-precision-init", "0.80",
        "--pseudo-precision-momentum", "0.90", "--sam-cache-root", "sam_mask_cache",
        "--sam-node-min-area-ratio", "0.01", "--sam-node-max-area-ratio", "0.05",
        "--sam-tau-gt", "0.80", "--sam-tau-purity", "0.90",
        "--sam-node-min-mask-pixels", "4", "--sam-node-min-changed-pixels", "4",
        "--lambda-graph-gt", "0.10", "--lambda-edge-gt", "0.10",
        "--graph-stage2-start-epoch", "55", "--graph-local-topk", "8",
        "--graph-edge-threshold", "0.50", "--graph-edge-bias-beta", "1.0",
        "--graph-gat-heads", "4", "--graph-temp-weight", "0.02",
        "--graph-pixel-pseudo-weight", str(float(weight)),
        "--graph-pseudo-ramp-epochs", "10", "--graph-confidence-threshold", "0.70",
        "--graph-teacher-threshold", "0.90", "--teacher-fallback-threshold", "0.95",
        "--amp", "--no-gradient-checkpointing", "--cudnn-benchmark",
        "--persistent-workers", "--use-ema-teacher", "--use-semantic-pseudo",
        "--use-sam-graph-stage1", "--use-sam-graph-stage2",
        "--graph-pseudo-semantic-mode", "complement", "--resume", str(CHECKPOINT),
    ]
    if smoke_steps:
        arguments.extend(["--smoke-steps", str(int(smoke_steps))])
    return arguments


def query(command):
    return subprocess.check_output(command, text=True).strip()


def verify_gpu_is_idle(gpu):
    memory_used, utilization = map(
        int,
        query([
            "nvidia-smi", "-i", str(gpu),
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]).split(","),
    )
    gpu_uuid = query([
        "nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"
    ])
    applications = query([
        "nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid",
        "--format=csv,noheader,nounits",
    ])
    occupied = [line for line in applications.splitlines() if gpu_uuid in line]
    if memory_used > 100 or utilization > 5 or occupied:
        raise RuntimeError(
            "GPU {} is not idle: memory={} MiB utilization={}% apps={}".format(
                gpu, memory_used, utilization, occupied
            )
        )


def launch(branch, weight, gpu, run_dir=None, smoke_steps=0):
    validate_branch(branch, weight)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError("resume checkpoint is missing: {}".format(CHECKPOINT))
    if not PYTHON.is_file():
        raise FileNotFoundError("Python interpreter is missing: {}".format(PYTHON))
    verify_gpu_is_idle(gpu)

    run = Path(run_dir).resolve() if run_dir else EXPERIMENT / branch
    output = run / "output"
    pid_path = run / "training.pid"
    if output.exists() or pid_path.exists():
        raise FileExistsError("refusing to overwrite existing run: {}".format(run))
    run.mkdir(parents=True, exist_ok=True)
    output.mkdir()

    arguments = build_arguments(branch, weight, smoke_steps=smoke_steps)
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(ROOT),
        "SCD_DATASET": "second",
        "SCD_PROJECT_ROOT": str(ROOT),
        "SCD_RUN_DIR": str(output),
    })
    log_path = run / "train.log"
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            arguments,
            cwd=str(ROOT),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    pid_path.write_text(str(process.pid) + "\n", encoding="ascii")
    command_text = (
        "CUDA_VISIBLE_DEVICES={} SCD_PROJECT_ROOT={} SCD_RUN_DIR={} {}\n".format(
            gpu, shlex.quote(str(ROOT)), shlex.quote(str(output)), shlex.join(arguments)
        )
    )
    (run / "command.txt").write_text(command_text, encoding="utf-8")
    manifest = {
        "created_epoch": int(time.time()),
        "branch": branch,
        "graph_pixel_pseudo_weight": float(weight),
        "physical_gpu": int(gpu),
        "pid": process.pid,
        "dataset": "SECOND",
        "num_classes": 7,
        "resume_checkpoint": str(CHECKPOINT),
        "resume_checkpoint_sha256": sha256(CHECKPOINT),
        "next_epoch_expected": 55,
        "stop_epoch_inclusive": 69,
        "smoke_steps": int(smoke_steps),
        "command_sha256": hashlib.sha256(command_text.encode("utf-8")).hexdigest(),
        "source_sha256": {
            "train_ablation.py": sha256(EXPERIMENT / "train_ablation.py"),
            "graph_pseudo_complement.py": sha256(
                EXPERIMENT / "graph_pseudo_complement.py"
            ),
        },
    }
    (run / "launch.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", required=True, choices=sorted(BRANCHES))
    parser.add_argument("--weight", required=True, type=float)
    parser.add_argument("--gpu", required=True, type=int, choices=range(4))
    parser.add_argument("--run-dir")
    parser.add_argument("--smoke-steps", type=int, default=0)
    options = parser.parse_args()
    if options.smoke_steps < 0:
        parser.error("--smoke-steps must be non-negative")
    try:
        result = launch(
            options.branch,
            options.weight,
            options.gpu,
            run_dir=options.run_dir,
            smoke_steps=options.smoke_steps,
        )
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
