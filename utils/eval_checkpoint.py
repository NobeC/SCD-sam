from pathlib import Path


def resolve_checkpoint_path(pth_path, project_root):
    """Resolve an evaluation checkpoint independently of the current directory."""
    checkpoint = Path(pth_path).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = Path(project_root) / checkpoint
    checkpoint = checkpoint.resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(
            "Evaluation checkpoint does not exist: {}".format(checkpoint)
        )
    return str(checkpoint)
