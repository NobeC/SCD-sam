from dataclasses import dataclass
from itertools import chain

import torch

from models.graph.edges import ClassBalancedNodeQueue
from models.graph.node_projector import TemporalNodeProjector
from models.graph.region_features import SamGraphFeatureProvider
from models.graph.stage1 import SamGraphStage1
from models.graph.stage2 import SamGraphStage2
from utils.sam_mask_cache import SamMaskCacheConfig
from utils.sam_node_audit import SamNodeAuditConfig, audit_sam_nodes


def add_graph_stage1_arguments(parser):
    group = parser.add_argument_group("SAM graph Stage1")
    group.add_argument("--use-sam-graph-stage1", action="store_true", default=False,
                       help="Enable GT-only SAM node and edge representation losses.")
    group.add_argument("--sam-cache-root", type=str, default="sam_mask_cache",
                       help="Root containing <Dataset>/train/<sample_id>.npz caches.")
    group.add_argument("--sam-node-min-area-ratio", type=float, default=0.001,
                       help="Minimum shared SAM mask area ratio admitted as a graph node.")
    group.add_argument("--sam-node-max-area-ratio", type=float, default=0.05,
                       help="Maximum shared SAM mask area ratio admitted as a graph node.")
    group.add_argument("--lambda-graph-gt", type=float, default=0.1)
    group.add_argument("--lambda-edge-gt", type=float, default=0.1)
    group.add_argument("--graph-learning-rate", type=float, default=1e-3)
    group.add_argument("--graph-weight-decay", type=float, default=1e-3)
    group.add_argument("--graph-grad-clip", type=float, default=1.0)
    group.add_argument("--graph-node-dim", type=int, default=128)
    group.add_argument("--graph-queue-capacity", type=int, default=256)
    group.add_argument("--graph-max-pairs", type=int, default=256)
    group.add_argument("--graph-hard-negative-ratio", type=float, default=0.5)
    group.add_argument("--sam-tau-gt", type=float, default=0.8)
    group.add_argument("--sam-tau-purity", type=float, default=0.9)
    group.add_argument("--sam-tau-u", type=float, default=0.95)
    group.add_argument("--sam-node-min-mask-pixels", type=int, default=4)
    group.add_argument("--sam-node-min-changed-pixels", type=int, default=4)
    group.add_argument("--use-sam-graph-stage2", action="store_true", default=False,
                       help="Enable batch-local SAM graph propagation and graph-veto pseudo labels.")
    group.add_argument("--graph-stage2-start-epoch", type=int, default=55)
    group.add_argument("--graph-local-topk", type=int, default=8)
    group.add_argument("--graph-edge-threshold", type=float, default=0.5)
    group.add_argument("--graph-edge-bias-beta", type=float, default=1.0)
    group.add_argument("--graph-gat-heads", type=int, default=4)
    group.add_argument("--graph-gat-dropout", type=float, default=0.1)
    group.add_argument("--graph-temp-weight", type=float, default=0.02)
    group.add_argument("--graph-pixel-pseudo-weight", type=float, default=0.1)
    group.add_argument("--graph-pseudo-ramp-epochs", type=int, default=10)
    group.add_argument("--graph-confidence-threshold", type=float, default=0.7)
    group.add_argument("--graph-teacher-threshold", type=float, default=0.9)
    group.add_argument("--teacher-fallback-threshold", type=float, default=0.95)
    return parser


def build_sam_cache_config(options):
    return SamMaskCacheConfig(
        min_area_ratio=options.sam_node_min_area_ratio,
        max_area_ratio=options.sam_node_max_area_ratio,
    )


def build_graph_stage1_runtime(backbone, num_classes, options):
    audit_config = SamNodeAuditConfig(
        tau_gt=options.sam_tau_gt,
        tau_purity=options.sam_tau_purity,
        tau_u=options.sam_tau_u,
        min_mask_pixels=options.sam_node_min_mask_pixels,
        min_changed_pixels=options.sam_node_min_changed_pixels,
    )
    return GraphStage1TrainingRuntime(
        backbone=backbone,
        num_classes=num_classes,
        node_dim=options.graph_node_dim,
        learning_rate=options.graph_learning_rate,
        weight_decay=options.graph_weight_decay,
        audit_config=audit_config,
        queue_capacity=options.graph_queue_capacity,
        max_pairs=options.graph_max_pairs,
        hard_negative_ratio=options.graph_hard_negative_ratio,
        use_stage2=options.use_sam_graph_stage2,
        graph_local_topk=options.graph_local_topk,
        graph_edge_threshold=options.graph_edge_threshold,
        graph_edge_bias_beta=options.graph_edge_bias_beta,
        graph_gat_heads=options.graph_gat_heads,
        graph_gat_dropout=options.graph_gat_dropout,
    )


@dataclass
class GraphRuntimeResult:
    loss_graph_gt: torch.Tensor
    loss_edge_gt: torch.Tensor
    loss_stage2_graph_gt: torch.Tensor
    loss_graph_temp: torch.Tensor
    gt_node_count: int
    positive_pair_count: int
    negative_pair_count: int
    prototype_initialized: torch.Tensor
    stage2_result: object
    audits: list
    masks: list


class GraphStage1TrainingRuntime:
    """Own graph-only Stage1 modules, optimizer, and resumable state."""

    checkpoint_key = "sam_graph_stage1"

    def __init__(
        self,
        backbone,
        num_classes,
        stage_channels=320,
        neck_channels=256,
        node_dim=128,
        learning_rate=1e-3,
        weight_decay=1e-3,
        audit_config=None,
        prototype_momentum=0.99,
        prototype_temperature=0.1,
        queue_capacity=256,
        max_pairs=256,
        hard_negative_ratio=0.5,
        use_stage2=False,
        graph_local_topk=8,
        graph_edge_threshold=0.5,
        graph_edge_bias_beta=1.0,
        graph_gat_heads=4,
        graph_gat_dropout=0.1,
    ):
        self.num_classes = int(num_classes)
        self.node_dim = int(node_dim)
        device = next(backbone.parameters(), torch.empty(0)).device
        self.use_stage2 = bool(use_stage2)
        self.feature_provider = SamGraphFeatureProvider(
            backbone, detach=not self.use_stage2)
        self.projector = TemporalNodeProjector(
            stage_channels=stage_channels,
            neck_channels=neck_channels,
            node_dim=self.node_dim,
        ).to(device)
        self.stage1 = SamGraphStage1(
            num_classes=self.num_classes,
            node_dim=self.node_dim,
            prototype_momentum=prototype_momentum,
            prototype_temperature=prototype_temperature,
            queue_capacity=queue_capacity,
            max_pairs=max_pairs,
            hard_negative_ratio=hard_negative_ratio,
        ).to(device)
        self.stage2 = None
        if self.use_stage2:
            self.stage2 = SamGraphStage2(
                num_classes=self.num_classes,
                node_dim=self.node_dim,
                heads=int(graph_gat_heads),
                dropout=float(graph_gat_dropout),
                local_topk=int(graph_local_topk),
                edge_threshold=float(graph_edge_threshold),
                edge_bias_beta=float(graph_edge_bias_beta),
            ).to(device)
        self.audit_config = audit_config or SamNodeAuditConfig()
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
        self.config = {
            "num_classes": self.num_classes,
            "stage_channels": int(stage_channels),
            "neck_channels": int(neck_channels),
            "node_dim": self.node_dim,
            "queue_capacity": int(queue_capacity),
            "use_stage2": self.use_stage2,
            "graph_local_topk": int(graph_local_topk),
            "graph_edge_threshold": float(graph_edge_threshold),
            "graph_edge_bias_beta": float(graph_edge_bias_beta),
            "graph_gat_heads": int(graph_gat_heads),
            "graph_gat_dropout": float(graph_gat_dropout),
        }
        self.stage2_counters = {"batches": 0, "accepted_pixels": 0, "nonfinite_skips": 0}
        self.loaded_format_version = None

    def parameters(self):
        groups = [self.projector.parameters(), self.stage1.parameters()]
        if self.stage2 is not None:
            groups.append(self.stage2.parameters())
        return chain(*groups)

    def _stage1_parameters(self):
        return chain(self.projector.parameters(), self.stage1.parameters())

    def train(self):
        self.projector.train()
        self.stage1.train()
        if self.stage2 is not None:
            self.stage2.train()

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def compute(
        self,
        images_a,
        images_b,
        labels_a,
        labels_b,
        graph_data,
        backbone_features=None,
        stage2_enabled=False,
    ):
        if len(graph_data) != images_a.shape[0]:
            raise ValueError("graph metadata length must match image batch size")
        masks = [item["shared_masks"] for item in graph_data]
        audits = [
            audit_sam_nodes(
                source_masks,
                labels_a[index].detach().cpu(),
                labels_b[index].detach().cpu(),
                self.num_classes,
                self.audit_config,
            )
            for index, source_masks in enumerate(masks)
        ]
        if stage2_enabled and self.stage2 is None:
            raise ValueError("stage2_enabled requires a Stage2-configured graph runtime")
        if backbone_features is None:
            stage_a, neck_a, stage_b, neck_b = self.feature_provider(images_a, images_b)
        else:
            if len(backbone_features) != 4:
                raise ValueError("backbone_features must contain stage/neck tensors for A/B")
            stage_a, neck_a, stage_b, neck_b = backbone_features
        detached_features = tuple(
            tensor.detach() for tensor in (stage_a, neck_a, stage_b, neck_b))
        stage1_nodes = self.projector(*detached_features, masks)
        stage1_result = self.stage1(stage1_nodes, audits)
        zero = stage1_result.loss_graph_gt * 0.0
        stage2_result = None
        if stage2_enabled:
            live_nodes = self.projector(
                stage_a, neck_a, stage_b, neck_b, masks)
            stage2_result = self.stage2(live_nodes, audits, self.stage1)
            self.stage2_counters["batches"] += 1
        return GraphRuntimeResult(
            loss_graph_gt=stage1_result.loss_graph_gt,
            loss_edge_gt=stage1_result.loss_edge_gt,
            loss_stage2_graph_gt=(
                stage2_result.loss_graph_gt if stage2_result is not None else zero),
            loss_graph_temp=(
                stage2_result.loss_temporal if stage2_result is not None else zero),
            gt_node_count=stage1_result.gt_node_count,
            positive_pair_count=stage1_result.positive_pair_count,
            negative_pair_count=stage1_result.negative_pair_count,
            prototype_initialized=stage1_result.prototype_initialized,
            stage2_result=stage2_result,
            audits=audits,
            masks=masks,
        )

    def state_dict(self):
        queue_nodes, queue_classes = self.stage1.queue.snapshot("cpu")
        state = {
            "format_version": 2 if self.stage2 is not None else 1,
            "graph_stage": 2 if self.stage2 is not None else 1,
            "config": dict(self.config),
            "projector": self.projector.state_dict(),
            "stage1": self.stage1.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "queue_nodes": queue_nodes,
            "queue_classes": queue_classes,
        }
        if self.stage2 is not None:
            state["stage2"] = self.stage2.state_dict()
            state["stage2_counters"] = dict(self.stage2_counters)
        return state

    def _restore_v1_optimizer(self, saved_state):
        current_group = self.optimizer.param_groups[0]
        temporary = torch.optim.AdamW(
            self._stage1_parameters(),
            lr=current_group["lr"], weight_decay=current_group["weight_decay"])
        temporary.load_state_dict(saved_state)
        self.optimizer.state.clear()
        for parameter, restored_parameter in zip(
                self._stage1_parameters(), temporary.param_groups[0]["params"]):
            if restored_parameter in temporary.state:
                self.optimizer.state[parameter] = temporary.state[restored_parameter]
        for key, value in temporary.param_groups[0].items():
            if key != "params":
                current_group[key] = value

    def load_state_dict(self, state):
        format_version = int(state.get("format_version", 1))
        if format_version not in (1, 2):
            raise ValueError("Unsupported SAM graph checkpoint format version")
        saved_config = state.get("config", {})
        for key in ("num_classes", "stage_channels", "neck_channels", "node_dim"):
            if key in saved_config and int(saved_config[key]) != int(self.config[key]):
                raise ValueError(
                    "Graph checkpoint {}={} does not match active {}={}".format(
                        key, saved_config[key], key, self.config[key]))
        self.projector.load_state_dict(state["projector"], strict=True)
        self.stage1.load_state_dict(state["stage1"], strict=True)
        if format_version == 1 and self.stage2 is not None:
            self._restore_v1_optimizer(state["optimizer"])
        else:
            self.optimizer.load_state_dict(state["optimizer"])
        self.stage1.queue = ClassBalancedNodeQueue(
            self.num_classes,
            self.node_dim,
            capacity_per_class=self.config["queue_capacity"],
        )
        self.stage1.queue.enqueue(state["queue_nodes"], state["queue_classes"])
        if format_version == 2:
            if self.stage2 is None:
                raise ValueError("Stage2 checkpoint requires --use-sam-graph-stage2")
            self.stage2.load_state_dict(state["stage2"], strict=True)
            self.stage2_counters.update(state.get("stage2_counters", {}))
        self.loaded_format_version = format_version

    def add_to_checkpoint_payload(self, payload):
        payload[self.checkpoint_key] = self.state_dict()
        return payload

    def load_checkpoint_payload(self, payload):
        state = payload.get(self.checkpoint_key)
        if state is None:
            return False
        self.load_state_dict(state)
        return True
