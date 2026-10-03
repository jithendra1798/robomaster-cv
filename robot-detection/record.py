"""
Write small reproducibility records next to datasets and training runs (see EXPERIMENTS.md).

Each record says when and how something was built: the exact command, the git commit (and whether
files had uncommitted changes), checksums of the scripts used, and package versions. A copy of the
scripts goes into <folder>/code/, so the output can be rebuilt even if the code changed later.
"""

import hashlib
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def folder_fingerprint(root):
    """md5 of the sorted 'relative path + size' list: changes if any file is added, removed or resized."""
    root = Path(root)
    listing = sorted(f"{p.relative_to(root)} {p.stat().st_size}" for p in root.rglob("*") if p.is_file())
    return {"files": len(listing), "md5": hashlib.md5("\n".join(listing).encode()).hexdigest()}


def code_state():
    """Git commit of this folder and whether its files differ from that commit."""
    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(HERE), *args], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"commit": git("rev-parse", "--short", "HEAD") or "unknown",
            "uncommitted_changes": bool(git("status", "--porcelain", "--", "."))}


def environment():
    env = {"host": platform.node(), "python": platform.python_version()}
    for name in ("torch", "ultralytics", "albumentations"):
        try:
            env[name] = str(__import__(name).__version__)
        except Exception:
            pass
    return env


def write_record(folder, filename, info, scripts=()):
    """Write <folder>/<filename> as YAML and copy the scripts used into <folder>/code/."""
    folder = Path(folder)
    record = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(["python", *sys.argv]),
        "code": {**code_state(), "scripts_md5": {Path(s).name: md5(s) for s in scripts}},
        "environment": environment(),
        **info,
    }
    (folder / filename).write_text(yaml.safe_dump(record, sort_keys=False))
    if scripts:
        (folder / "code").mkdir(parents=True, exist_ok=True)
        for s in scripts:
            shutil.copy2(s, folder / "code" / Path(s).name)
