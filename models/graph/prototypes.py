import torch
import torch.nn as nn
import torch.nn.functional as F


class EMAPrototypeBank(nn.Module):
    """FP32 class-ID-aligned EMA prototypes for true GT SAM nodes."""

    def __init__(self, num_classes, node_dim=128, momentum=0.99, temperature=0.1):
        super().__init__()
        self.num_classes = int(num_classes)
        self.node_dim = int(node_dim)
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        if self.num_classes < 2 or self.node_dim <= 0:
            raise ValueError("prototype bank needs at least two classes and positive node_dim")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("prototype momentum must be in [0, 1)")
        if self.temperature <= 0.0:
            raise ValueError("prototype temperature must be positive")
        self.register_buffer(
            "prototypes", torch.zeros(self.num_classes, self.node_dim, dtype=torch.float32))
        self.register_buffer(
            "initialized", torch.zeros(self.num_classes, dtype=torch.bool))

    def _validate(self, nodes, classes=None):
        if nodes.ndim != 2 or nodes.shape[1] != self.node_dim:
            raise ValueError("nodes must have shape [N, {}]".format(self.node_dim))
        if classes is not None:
            if classes.ndim != 1 or classes.shape[0] != nodes.shape[0]:
                raise ValueError("classes must have shape [N]")
            if classes.numel() and (int(classes.min()) < 0 or int(classes.max()) >= self.num_classes):
                raise ValueError("prototype class ID is outside the configured dataset range")

    @torch.no_grad()
    def update(self, nodes, classes):
        classes = torch.as_tensor(classes, dtype=torch.long, device=nodes.device)
        self._validate(nodes, classes)
        normalized = F.normalize(nodes.detach().float(), dim=1, eps=1e-6)
        for class_id in range(1, self.num_classes):
            selected = normalized[classes == class_id]
            if selected.shape[0] == 0:
                continue
            batch = F.normalize(selected.mean(dim=0), dim=0, eps=1e-6)
            if self.initialized[class_id]:
                updated = self.momentum * self.prototypes[class_id] + (1.0 - self.momentum) * batch
                self.prototypes[class_id].copy_(F.normalize(updated, dim=0, eps=1e-6))
            else:
                self.prototypes[class_id].copy_(batch)
                self.initialized[class_id] = True
        self.prototypes[0].zero_()
        self.initialized[0] = False

    def logits(self, nodes):
        self._validate(nodes)
        with torch.autocast(device_type=nodes.device.type, enabled=False):
            normalized_nodes = F.normalize(nodes.float(), dim=1, eps=1e-6)
            normalized_prototypes = F.normalize(self.prototypes.float(), dim=1, eps=1e-6)
            logits = normalized_nodes @ normalized_prototypes.t()
            logits = logits / self.temperature
            valid = self.initialized.clone()
            valid[0] = False
            return logits.masked_fill(
                ~valid.unsqueeze(0), torch.finfo(torch.float32).min)

    def classification_loss(self, nodes, classes):
        classes = torch.as_tensor(classes, dtype=torch.long, device=nodes.device)
        self._validate(nodes, classes)
        if nodes.shape[0] == 0:
            return nodes.sum() * 0.0
        eligible = (classes > 0) & self.initialized.to(classes.device)[classes]
        if not eligible.any():
            return nodes.sum() * 0.0
        return F.cross_entropy(self.logits(nodes[eligible]), classes[eligible])
