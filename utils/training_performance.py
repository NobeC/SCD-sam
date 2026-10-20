import argparse
import time

import torch


def loader_worker_options(num_workers, persistent_workers=True):
    """Return DataLoader options that are valid only with worker processes."""
    num_workers = int(num_workers)
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if num_workers == 0:
        return {}
    return {
        "persistent_workers": bool(persistent_workers),
        "prefetch_factor": 2,
    }


def configure_cudnn(deterministic=False):
    """Select reproducible or fixed-shape throughput-oriented cuDNN behavior."""
    deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    return {
        "benchmark": torch.backends.cudnn.benchmark,
        "deterministic": torch.backends.cudnn.deterministic,
    }


class IntervalTimer:
    """Measure average data and compute durations at a sampled cadence."""

    def __init__(self, interval):
        self.interval = int(interval)
        if self.interval < 0:
            raise ValueError("interval must be non-negative")
        self._data_started = None
        self._compute_started = None
        self._data_total = 0.0
        self._compute_total = 0.0
        self._step_total = 0.0
        self._count = 0

    def start_data(self):
        self._data_started = time.perf_counter()

    def start_compute(self):
        if self._data_started is None:
            raise RuntimeError("start_data must be called before start_compute")
        self._compute_started = time.perf_counter()
        self._data_total += self._compute_started - self._data_started

    def finish_step(self, iteration, synchronize=None):
        if self._data_started is None or self._compute_started is None:
            raise RuntimeError("timer step was not started")
        sampled = self.interval > 0 and int(iteration) % self.interval == 0
        if sampled and synchronize is not None:
            synchronize()
        finished = time.perf_counter()
        self._compute_total += finished - self._compute_started
        self._step_total += finished - self._data_started
        self._count += 1
        self._data_started = None
        self._compute_started = None
        if not sampled:
            return None
        count = max(self._count, 1)
        result = {
            "data_seconds": self._data_total / count,
            "compute_seconds": self._compute_total / count,
            "step_seconds": self._step_total / count,
        }
        self._data_total = self._compute_total = self._step_total = 0.0
        self._count = 0
        return result


def _non_negative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def add_performance_arguments(parser):
    """Add training throughput and reproducibility controls."""
    parser.add_argument(
        "--gradient-checkpointing", dest="gradient_checkpointing",
        action="store_true", default=False,
        help="Recompute backbone activations during backward to save memory.")
    parser.add_argument(
        "--no-gradient-checkpointing", dest="gradient_checkpointing",
        action="store_false", help="Disable backbone gradient checkpointing.")

    cudnn_group = parser.add_mutually_exclusive_group()
    cudnn_group.add_argument(
        "--cudnn-benchmark", dest="deterministic", action="store_false",
        help="Tune cuDNN kernels for fixed-size inputs (default).")
    cudnn_group.add_argument(
        "--deterministic", dest="deterministic", action="store_true",
        help="Use deterministic cuDNN behavior for exact reproduction.")
    parser.set_defaults(deterministic=False)

    worker_group = parser.add_mutually_exclusive_group()
    worker_group.add_argument(
        "--persistent-workers", dest="persistent_workers", action="store_true",
        help="Keep DataLoader workers alive between epochs.")
    worker_group.add_argument(
        "--no-persistent-workers", dest="persistent_workers", action="store_false",
        help="Restart DataLoader workers each epoch (default).")
    parser.set_defaults(persistent_workers=False)

    parser.add_argument(
        "--timing-interval", type=_non_negative_int, default=50,
        help="Report averaged data/compute time every N steps; 0 disables it.")
    return parser


def is_sample_step(iteration, interval, include_first=True):
    """Return whether a one-based iteration should run sampled diagnostics."""
    iteration = int(iteration)
    interval = int(interval)
    if iteration <= 0:
        raise ValueError("iteration must be positive")
    if interval < 0:
        raise ValueError("interval must be non-negative")
    if interval == 0:
        return False
    return (bool(include_first) and iteration == 1) or iteration % interval == 0


def change_accuracy(outputs_a, outputs_b, change_logits, labels_a, labels_b):
    """Match the legacy per-sample A/B accuracy while staying on-device."""
    if outputs_a.ndim != 4 or outputs_b.shape != outputs_a.shape:
        raise ValueError("semantic outputs must have matching [B, C, H, W] shapes")
    expected_labels = (outputs_a.shape[0], outputs_a.shape[2], outputs_a.shape[3])
    if tuple(labels_a.shape) != expected_labels or labels_b.shape != labels_a.shape:
        raise ValueError("labels must match semantic output spatial dimensions")
    if tuple(change_logits.shape) != (outputs_a.shape[0], 1, *expected_labels[1:]):
        raise ValueError("change logits must have shape [B, 1, H, W]")

    changed = change_logits.detach().squeeze(1) > 0
    pred_a = outputs_a.detach().argmax(dim=1) * changed.long()
    pred_b = outputs_b.detach().argmax(dim=1) * changed.long()

    def per_sample(prediction, labels):
        valid = labels >= 0
        correct = (valid & (prediction == labels)).flatten(1).sum(dim=1).float()
        total = valid.flatten(1).sum(dim=1).clamp_min(1).float()
        return correct / total

    return ((per_sample(pred_a, labels_a) + per_sample(pred_b, labels_b)) * 0.5).mean()


class TrainingMetricAccumulator:
    """Accumulate scalar training metrics without synchronizing every step."""

    def __init__(self, names, device):
        self.names = tuple(names)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("metric names must be non-empty and unique")
        self.device = torch.device(device)
        self.last = {
            name: torch.zeros((), device=self.device, dtype=torch.float32)
            for name in self.names
        }
        self.total = {
            name: torch.zeros((), device=self.device, dtype=torch.float32)
            for name in self.names
        }
        self.count = 0

    @torch.no_grad()
    def update(self, **values):
        if set(values) != set(self.names):
            raise ValueError("metric names do not match accumulator")
        for name in self.names:
            value = torch.as_tensor(
                values[name], device=self.device, dtype=torch.float32).detach()
            if value.numel() != 1:
                raise ValueError("training metrics must be scalar")
            value = value.reshape(())
            self.last[name].copy_(value)
            self.total[name].add_(value)
        self.count += 1

    def snapshot(self):
        if self.count == 0:
            return {
                name: {"last": 0.0, "average": 0.0}
                for name in self.names
            }
        packed = []
        for name in self.names:
            packed.extend((self.last[name], self.total[name] / self.count))
        values = torch.stack(packed).detach().cpu().tolist()
        return {
            name: {
                "last": float(values[2 * index]),
                "average": float(values[2 * index + 1]),
            }
            for index, name in enumerate(self.names)
        }
