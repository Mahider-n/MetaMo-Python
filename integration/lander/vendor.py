"""Load the pinned submodule using its existing top-level import layout."""
from pathlib import Path
import sys
import subprocess

SUBREP = Path(__file__).resolve().parents[2] / "external" / "subrep"


def source_revision(path):
    """Return the commit recorded for one source checkout."""
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def enable_subrep():
    if not (SUBREP / "certification/certificate_schema.py").is_file():
        raise RuntimeError("Initialize SubRep: git submodule update --init --recursive")
    # Upstream uses absolute imports (certification, utils, env, library, pilot).
    # Reject collisions rather than silently binding another installed package.
    for name in ("certification", "utils", "env", "library", "pilot", "baseline"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__file__", None):
            if not Path(module.__file__).resolve().is_relative_to(SUBREP):
                raise RuntimeError(f"SubRep import namespace collision: {name}")
    if str(SUBREP) not in sys.path:
        sys.path.append(str(SUBREP))
