import argparse
import json
import re
from pathlib import Path


EPOCH_RE = re.compile(r"\[epoch (\d+)\]")
VAL_RE = re.compile(
    r"Val loss: ([0-9.]+) Fscd: ([0-9.]+) IoU: ([0-9.]+) "
    r"Sek: ([0-9.]+) Accuracy: ([0-9.]+)"
)


def parse_log(path):
    current_epoch = None
    rows = []
    nonfinite_skips = 0
    for line in Path(path).read_text(errors="replace").splitlines():
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        if "non-finite gradient" in line:
            nonfinite_skips += 1
        val_match = VAL_RE.search(line)
        if val_match and current_epoch is not None:
            loss, fscd, miou, sek, oa = map(float, val_match.groups())
            rows.append({
                "epoch": current_epoch,
                "loss": loss,
                "fscd": fscd,
                "miou": miou,
                "sek": sek,
                "oa": oa,
            })
    best = max(rows, key=lambda row: row["fscd"]) if rows else None
    return {"epochs": rows, "best_fscd": best, "nonfinite_skips": nonfinite_skips}


def tail_summary(parsed, start=65, end=69):
    rows = [row for row in parsed["epochs"] if start <= row["epoch"] <= end]
    return {
        "completed_epochs": [row["epoch"] for row in rows],
        "mean_fscd": sum(row["fscd"] for row in rows) / len(rows) if rows else None,
        "mean_sek": sum(row["sek"] for row in rows) / len(rows) if rows else None,
        "nonfinite_skips": parsed["nonfinite_skips"],
    }


def summarize_experiment(experiment):
    root = Path(experiment)
    result = {}
    for branch in ("control", "graph02", "graph05"):
        log = root / branch / "train.log"
        if not log.is_file():
            continue
        parsed = parse_log(log)
        result[branch] = {
            **parsed,
            "epochs_65_69": tail_summary(parsed),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        default=str(Path(__file__).resolve().parent),
    )
    parser.add_argument("--output")
    options = parser.parse_args()
    result = summarize_experiment(options.experiment)
    text = json.dumps(result, indent=2) + "\n"
    if options.output:
        Path(options.output).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
