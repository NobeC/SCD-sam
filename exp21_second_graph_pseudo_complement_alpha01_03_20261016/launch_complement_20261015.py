import hashlib
import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/homenew/yang/SCD/SCD-SAM/SCD")
RUN = ROOT / "exp21_second_graph_pseudo_complement_20261015"
PYTHON = Path("/home/homenew/yang/anaconda3/envs/scd-sam-cuda12/bin/python")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def query(command):
    return subprocess.check_output(command, text=True).strip()


def verify_gpu_is_idle(gpu):
    memory_used, utilization = map(int, query([
        "nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]).split(","))
    gpu_uuid = query(["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"])
    apps = query([
        "nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid",
        "--format=csv,noheader,nounits",
    ])
    occupied = [line for line in apps.splitlines() if line.strip() and gpu_uuid in line]
    if memory_used > 100 or utilization > 5 or occupied:
        raise SystemExit(
            "GPU {} is not idle; refusing to launch (memory={} MiB, utilization={}%, apps={})".format(
                gpu, memory_used, utilization, occupied))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(RUN))
    parser.add_argument("--gpu", type=int, choices=range(4), default=0)
    parser.add_argument("--pseudo-alpha-start", type=float, default=0.03)
    parser.add_argument("--pseudo-alpha-end", type=float, default=0.10)
    options = parser.parse_args()
    run = Path(options.run_dir).resolve()
    gpu = str(options.gpu)
    if not 0.0 <= options.pseudo_alpha_start <= options.pseudo_alpha_end <= 1.0:
        raise SystemExit("pseudo alpha bounds must satisfy 0 <= start <= end <= 1")
    train_script = run / "train_complement_20261015.py"
    if not train_script.is_file() or not (run / "graph_pseudo_complement.py").is_file():
        raise SystemExit("training source files are missing")
    if not PYTHON.is_file():
        raise SystemExit("scd-sam-cuda12 Python interpreter is missing")
    output_dir = run / "output"
    if output_dir.exists() or (run / "training.pid").exists():
        raise SystemExit("production output or PID marker already exists; refusing to overwrite")
    verify_gpu_is_idle(options.gpu)
    output_dir.mkdir()

    arguments = [
        str(PYTHON), "-u", str(train_script),
        "--dataset", "second", "--num-classes", "7", "--cuda", "0",
        "--epochs", "100", "--seed", "42", "--learning-rate", "3.5e-4",
        "--batch-size", "2", "--effective-batch-size", "8", "--val-batch-size", "2",
        "--input-size", "512", "--num-workers", "4", "--grad-clip", "1.0",
        "--timing-interval", "20", "--ema-decay", "0.99",
        "--lambda-sem-sup", "1.0", "--lambda-sem-u", "0.10",
        "--pseudo-start-epoch", "40", "--pseudo-rampup-epoch", "5",
        "--pseudo-alpha-start", str(options.pseudo_alpha_start),
        "--pseudo-alpha-end", str(options.pseudo_alpha_end),
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
        "--graph-pixel-pseudo-weight", "0.10", "--graph-pseudo-ramp-epochs", "10",
        "--graph-confidence-threshold", "0.70", "--graph-teacher-threshold", "0.90",
        "--teacher-fallback-threshold", "0.95", "--amp", "--no-gradient-checkpointing",
        "--cudnn-benchmark", "--persistent-workers", "--use-ema-teacher",
        "--use-semantic-pseudo", "--use-sam-graph-stage1", "--use-sam-graph-stage2",
        "--graph-pseudo-semantic-mode", "complement",
    ]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": gpu,
        "PYTHONDONTWRITEBYTECODE": "1",
        "SCD_PROJECT_ROOT": str(ROOT),
        "SCD_RUN_DIR": str(output_dir),
        "SCD_DATASET": "second",
        "PYTHONPATH": str(ROOT),
    })
    log_path = run / "train.log"
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            arguments, cwd=str(ROOT), env=environment,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)

    (run / "training.pid").write_text(str(process.pid) + "\n", encoding="ascii")
    (run / "command.txt").write_text(
        "CUDA_VISIBLE_DEVICES={} SCD_PROJECT_ROOT={} SCD_RUN_DIR={} {}\n".format(
            gpu, shlex.quote(str(ROOT)), shlex.quote(str(output_dir)),
            shlex.join(arguments)), encoding="utf-8")
    launch = {
        "created_local_epoch": int(time.time()),
        "pid": process.pid,
        "physical_gpu": options.gpu,
        "cwd": str(ROOT),
        "python": str(PYTHON),
        "environment": "scd-sam-cuda12",
        "dataset": "SECOND",
        "semantic_graph_mode": "complement",
        "checkpoint_resume": None,
        "historical_best_reference": str(
            ROOT / "checkpoints/SECOND/SECOND_pseudo_graph_stage2_seed42_epoch097_Fscd0.6112.pth"),
        "source_sha256": {
            "train_complement_20261015.py": sha256(train_script),
            "graph_pseudo_complement.py": sha256(run / "graph_pseudo_complement.py"),
            "test_graph_pseudo_complement.py": sha256(run / "test_graph_pseudo_complement.py"),
        },
        "config": {
            "epochs": 100, "seed": 42, "batch_size": 2, "effective_batch_size": 8,
            "validation_batch_size": 2, "input_size": 512, "workers": 4,
            "learning_rate": 0.00035, "lambda_sem_u": 0.10,
            "pseudo_alpha": [options.pseudo_alpha_start, options.pseudo_alpha_end],
            "pseudo_start_epoch": 40,
            "graph_gt_weight": 0.10, "graph_edge_weight": 0.10,
            "graph_pixel_pseudo_weight": 0.10, "graph_stage2_start_epoch": 55,
            "graph_pseudo_ramp_epochs": 10, "sam_graph_stage1": True,
            "sam_graph_stage2": True, "ema_teacher": True,
            "semantic_pseudo_labels": True, "gradient_checkpointing": False,
            "amp": True,
        },
    }
    (run / "launch.json").write_text(
        json.dumps(launch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pid": process.pid, "log": str(log_path), "gpu": options.gpu}, indent=2))


if __name__ == "__main__":
    main()
