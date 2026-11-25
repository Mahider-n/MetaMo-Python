"""JSON writes shared by evidence, scheduler and feedback persistence."""
import os
from pathlib import Path

from .contracts import canonical


def write_json(path, data):
    """Write deterministic JSON and flush its contents to disk."""
    with Path(path).open("w", encoding="utf-8") as stream:
        stream.write(canonical(data) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_json(path, data):
    """Replace one JSON file durably; callers enforce single-writer ownership."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    write_json(temporary, data)
    temporary.replace(path)
    fsync_dir(path.parent)
