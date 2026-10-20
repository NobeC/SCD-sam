from models.graph.edges import ClassBalancedNodeQueue


GRAPH_CHECKPOINT_KEY = "sam_graph_stage1"


def add_graph_training_state(
    payload,
    projector,
    stage1,
    optimizer,
    graph_config=None,
    graph_stage=1,
):
    """Attach graph-only state without changing the base checkpoint schema."""
    queue_nodes, queue_classes = stage1.queue.snapshot("cpu")
    payload[GRAPH_CHECKPOINT_KEY] = {
        "format_version": 1,
        "graph_stage": int(graph_stage),
        "config": dict(graph_config or {}),
        "projector": projector.state_dict(),
        "stage1": stage1.state_dict(),
        "optimizer": optimizer.state_dict(),
        "queue_nodes": queue_nodes,
        "queue_classes": queue_classes,
    }
    return payload


def restore_graph_training_state(payload, projector, stage1, optimizer):
    """Restore optional graph state; old base-only checkpoints return False."""
    state = payload.get(GRAPH_CHECKPOINT_KEY)
    if state is None:
        return False
    projector.load_state_dict(state["projector"], strict=True)
    stage1.load_state_dict(state["stage1"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    capacity = int(state.get("config", {}).get(
        "queue_capacity", stage1.queue.capacity_per_class))
    stage1.queue = ClassBalancedNodeQueue(
        stage1.num_classes, stage1.node_dim, capacity_per_class=capacity)
    stage1.queue.enqueue(state["queue_nodes"], state["queue_classes"])
    return True
