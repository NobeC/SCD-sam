"""Supervised HRSCD adapter for the existing SAM region/prototype/GAT modules.

Only training imagery creates SAM masks. Unknown semantics never become stable
GT nodes. Actual binary change labels determine temporal consistency eligibility.
"""
from pathlib import Path
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from models.graph.graph_builder import flatten_stage2_nodes
from utils.sam_node_audit import SamNodeAudit, GT, IGNORE
from utils.graph_stage1_training import GraphStage1TrainingRuntime
from utils.sam_mask_cache import load_shared_sam_masks, SamMaskCacheConfig
from utils.sam_cache_format import save_sam_cache_v2


class TrainingMasks:
    def __init__(self, root, cache, device):
        self.root, self.cache, self.device = Path(root), Path(cache), device
        self.generator = None
        self.config = dict(model_type='vit_t', points_per_side=32, pred_iou_thresh=.75,
                           stability_score_thresh=.8, crop_n_layers=0, min_mask_region_area=16)
        self.filter = SamMaskCacheConfig(min_area_ratio=.001, max_area_ratio=.05)

    def get(self, name, geometry):
        if Path(name).name != name or not name.endswith('.png'):
            raise ValueError('Invalid training sample name')
        dest = self.cache / 'HRSCD/train' / (Path(name).stem + '.npz')
        if not dest.exists():
            if self.generator is None:
                from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
                model = sam_model_registry['vit_t'](checkpoint='weights/mobile_sam.pt').to(self.device).eval()
                self.generator = SamAutomaticMaskGenerator(model, **{k:v for k,v in self.config.items() if k != 'model_type'})
            records = []
            for side in ('A', 'B'):
                path = self.root / 'train' / side / name
                image = cv2.imread(str(path))
                if image is None:
                    raise OSError(str(path))
                with torch.inference_mode():
                    records.append(self.generator.generate(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_name(dest.stem + '.writing.npz')
            save_sam_cache_v2(temp, Path(name).stem, (512,512), records[0], records[1], self.config)
            os.replace(temp, dest)
        masks = load_shared_sam_masks(self.cache, 'HRSCD', 'train', Path(name).stem, (512,512), self.filter)
        # Deterministic region cap to bound graph memory. Same rule for every run.
        masks = masks[:128]
        k, horizontal, vertical = map(int, geometry)
        masks = torch.rot90(masks, k, (1,2))
        if horizontal:
            masks = masks.flip(2)
        if vertical:
            masks = masks.flip(1)
        return masks.contiguous()


def audit_regions(masks, label_a, label_b, change):
    masks = masks.cpu().bool()
    label_a, label_b, change = label_a.cpu(), label_b.cpu(), change.cpu().bool()
    n = len(masks)
    area = masks.flatten(1).sum(1)
    changed = (masks & change).flatten(1).sum(1)
    ratio = changed.float() / area.clamp_min(1)
    assignments = []
    for label in (label_a, label_b):
        states = torch.full((n,), IGNORE, dtype=torch.long)
        classes = torch.zeros(n, dtype=torch.long)
        purity = torch.zeros(n)
        for i, mask in enumerate(masks):
            valid = mask & (label > 0)
            count = int(valid.sum())
            if count < 4 or count / max(int(area[i]), 1) < .8:
                continue
            hist = torch.bincount(label[valid].long(), minlength=6)
            cls = int(hist[1:].argmax()) + 1
            purity[i] = hist[cls].float() / count
            if purity[i] >= .9:
                states[i], classes[i] = GT, cls
        assignments.append((states, classes, purity))
    sa, ca, pa = assignments[0]; sb, cb, pb = assignments[1]
    covered = masks.any(0) if n else torch.zeros_like(change)
    gt_masks = masks[(sa == GT) | (sb == GT)]
    gt_covered = gt_masks.any(0) if len(gt_masks) else torch.zeros_like(change)
    return SamNodeAudit(sa,sb,ca,cb,ratio,pa,pb,changed,int(change.sum()),
                        int((covered & change).sum()),int((gt_covered & change).sum()))


class SupervisedGraph(GraphStage1TrainingRuntime):
    def __init__(self, backbone):
        super().__init__(backbone, 6, use_stage2=True)

    def loss(self, features, masks, batch):
        audits = [audit_regions(m, batch['label_A'][i], batch['label_B'][i], batch['change'][i])
                  for i,m in enumerate(masks)]
        nodes = self.projector(*(x.detach() for x in features), masks)
        first = self.stage1(nodes, audits)
        live = self.projector(*features, masks)
        second = self.propagate_supervised(live, audits)
        loss = .1 * (first.loss_graph_gt + first.loss_edge_gt + second.loss_graph_gt) + .02 * second.loss_temporal
        return loss, {'graph_loss':float(loss.detach()), 'graph_nodes':second.node_count,
                      'graph_gt_nodes':second.graph_gt_count, 'graph_edges':second.edge_count,
                      'graph_spatial_edges':second.spatial_edges,'graph_feature_edges':second.feature_edges,
                      'graph_temporal_edges':second.temporal_edges}

    def propagate_supervised(self, nodes, audits):
        batch=flatten_stage2_nodes(nodes,audits,device=self.stage2.gat.query.weight.device)
        n=len(batch.nodes); device=batch.nodes.device
        source=[torch.arange(n,device=device)];destination=[source[0]]
        relation=[torch.full((n,),3,device=device,dtype=torch.long)]
        weights=[torch.ones(n,device=device)]; counts={0:0,1:0,2:0}
        temporal=[]
        def add(s,d,r,w):
            source.append(s);destination.append(d)
            relation.append(torch.full_like(s,r));weights.append(w);counts[r]+=len(s)
        for sample in range(len(audits)):
            for time in (0,1):
                pos=batch.positions(sample,time); size=len(pos)
                if size<2:continue
                k=min(8,size-1)
                with torch.no_grad():
                    geom=batch.geometry[pos,:2].float()
                    distance=torch.cdist(geom,geom)
                    distance.fill_diagonal_(float('inf'))
                    neighbors=distance.topk(k,largest=False).indices
                    d=pos[:,None].expand(-1,k).reshape(-1)
                    s=pos[neighbors.reshape(-1)]
                    w=1/(1+distance.gather(1,neighbors).reshape(-1))
                    add(s,d,1,w)
                    feats=F.normalize(batch.nodes[pos].float(),dim=1)
                    similarity=feats@feats.T;similarity.fill_diagonal_(-float('inf'))
                    near=similarity.topk(k).indices
                    s=pos[near.reshape(-1)]
                    with torch.autocast(device_type=device.type,enabled=False):
                        reliability=torch.sigmoid(self.stage1.edge_net(batch.nodes[s].float(),batch.nodes[d].float(),torch.zeros_like(s)))
                    keep=reliability>=.5
                    add(s[keep],d[keep],0,reliability[keep].detach())
            lookup={}
            for time in (0,1):
                for p in batch.positions(sample,time).tolist():
                    lookup[(time,int(batch.region_indices[p]))]=p
            for (time,region),left in lookup.items():
                right=lookup.get((1,region))
                if time==0 and right is not None and audits[sample].change_ratio[region]<=.05:
                    if int(batch.classes[left])==int(batch.classes[right]) and int(batch.classes[left])>0:
                        temporal.append((left,right))
        if temporal:
            pairs=torch.tensor(temporal,device=device,dtype=torch.long)
            s=torch.cat([pairs[:,0],pairs[:,1]]);d=torch.cat([pairs[:,1],pairs[:,0]])
            add(s,d,2,torch.ones_like(s,dtype=torch.float32))
        source=torch.cat(source);destination=torch.cat(destination);relation=torch.cat(relation);weights=torch.cat(weights)
        propagated=self.stage2.gat(batch.nodes,source,destination,relation,weights)
        zero=propagated.sum()*0.
        initialized=self.stage1.prototypes.initialized.to(device).clone();initialized[0]=False
        valid=(batch.states==GT)&initialized[batch.classes]
        if n and initialized.any():
            logits=self.stage1.prototypes.logits(self.stage1.prototype_projection(propagated.float()))
            gt_loss=F.cross_entropy(logits[valid],batch.classes[valid]) if valid.any() else zero
            probs=logits.float().softmax(1)
            temporal_losses=[self.stage2._symmetric_kl(probs[a],probs[b]) for a,b in temporal]
            temp_loss=torch.stack(temporal_losses).mean() if temporal_losses else zero
        else:gt_loss=temp_loss=zero
        return SimpleNamespace(loss_graph_gt=gt_loss,loss_temporal=temp_loss,node_count=n,
                               graph_gt_count=int(valid.sum()),edge_count=len(source),
                               spatial_edges=counts[1],feature_edges=counts[0],temporal_edges=counts[2])
