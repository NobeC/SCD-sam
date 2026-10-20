"""HRSCD supervision and streaming metrics, independent of sparse SCD labels."""
import torch
import torch.nn.functional as F
from utils.loss import weighted_BCE_logits


def supervised_loss(outputs, batch):
    a, b, change = (x.float() for x in outputs)
    losses = []
    for logits, target in ((a, batch['label_A']), (b, batch['label_B'])):
        valid = target != 0
        losses.append(F.cross_entropy(logits, target, ignore_index=0)
                      if valid.any() else logits.sum() * 0.)
    semantic = sum(losses) / 2
    binary = weighted_BCE_logits(change, batch['change'].unsqueeze(1).float())
    valid = (batch['label_A'] != 0) & (batch['label_B'] != 0)
    similarity = F.cosine_similarity(a[:, 1:].softmax(1), b[:, 1:].softmax(1), dim=1)
    similarity_loss = torch.where(batch['change'].bool(), similarity.clamp_min(0), 1 - similarity)
    consistency = similarity_loss[valid].mean() if valid.any() else (a.sum() + b.sum()) * 0.
    return semantic + binary + consistency, {'semantic': semantic, 'binary': binary, 'consistency': consistency}


class Metrics:
    def __init__(self):
        self.semantic = torch.zeros(6, 6, dtype=torch.int64)
        self.binary = torch.zeros(2, 2, dtype=torch.int64)
        self.scd = torch.zeros(6, 6, dtype=torch.int64)

    @torch.no_grad()
    def update(self, outputs, batch):
        for logits, target in zip(outputs[:2], (batch['label_A'], batch['label_B'])):
            pred = logits.argmax(1)
            valid = target > 0
            self.semantic += torch.bincount((target[valid] * 6 + pred[valid]).flatten(), minlength=36).reshape(6, 6).cpu()
            changed = batch['change'].bool()
            scd_target = target * changed.long()
            scd_pred = pred * (outputs[2][:, 0] > 0).long()
            # Unknown changed semantics are excluded, never relabeled unchanged.
            valid_scd = (~changed) | (target > 0)
            self.scd += torch.bincount((scd_target[valid_scd]*6+scd_pred[valid_scd]).flatten(), minlength=36).reshape(6,6).cpu()
        pred = (outputs[2][:, 0] > 0).long()
        self.binary += torch.bincount((batch['change'] * 2 + pred).flatten(), minlength=4).reshape(2, 2).cpu()

    def result(self):
        cm = self.semantic.double()
        union = cm.sum(0) + cm.sum(1) - cm.diag()
        ious = [float(cm[i, i] / union[i]) if union[i] > 0 else None for i in range(1, 6)]
        present = [v for v in ious if v is not None]
        tn, fp, fn, tp = self.binary.flatten().tolist()
        sc = self.scd.double()
        denom = sc[1:,:].sum() + sc[:,1:].sum()
        return {'Fscd': float(2*sc.diag()[1:].sum()/denom) if denom else None,
                'semantic_mIoU_present': sum(present) / len(present) if present else None,
                'semantic_IoU_classes_1_to_5': ious,
                'semantic_accuracy': float(cm.diag().sum() / cm.sum()) if cm.sum() else None,
                'change_F1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                'change_IoU': tp / (tp + fp + fn) if tp + fp + fn else None,
                'change_confusion_TN_FP_FN_TP': [tn, fp, fn, tp],
                'semantic_confusion': self.semantic.tolist()}
