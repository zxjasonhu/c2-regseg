#!/usr/bin/env python
"""Fail when the standalone release contains common private-data artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".ini",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
WEIGHT_SUFFIXES = {".ckpt", ".h5", ".onnx", ".pt", ".pth", ".safetensors"}
PRIVATE_DOCUMENT_SUFFIXES = {".doc", ".docx", ".xls", ".xlsx"}
PRIVATE_BASENAMES = {".DS_Store", "Thumbs.db"}
PRIVATE_PATH_PREFIXES = {
    ("writeup",),
}
EXPECTED_FIGURES = {
    "c2_anatomy.jpg",
    "example_sagittal.gif",
    "example_sagittal_center.png",
    "method_overview.jpg",
}
EXPECTED_ATLASES = {
    "atlas_01_embedded.npz": "9558dbffa7330e3b531ea5c9654f7196bdebb1c9be7db67d898135eeca3eb4a6",
    "atlas_02_embedded.npz": "d4f18b19c7ad5fbdc533078c880dfba77fd3dfe17b913dd6b8e11fec6f6bf9a5",
    "atlas_03_embedded.npz": "39dfe25ed3feec18cf7e806bf5539284e3492785867eb0153c984bf96af81fcf",
}
EXPECTED_ATLAS_LABELS = {
    0,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    14,
    15,
    16,
    17,
    20,
    21,
    22,
}
PATTERNS = {
    "absolute user path": re.compile(
        r"(?:/" + "home/" + r"[^/\s]+|/" + "Users/" + r"[^/\s]+|[A-Za-z]:\\\\" + "Users" + r"\\\\[^\\\s]+)"
    ),
    "DICOM UID value": re.compile(
        r"(?<![A-Za-z0-9_])1\.2(?:\.[0-9]+){5,}(?![A-Za-z0-9_])"
    ),
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
}


def release_candidates() -> set[Path]:
    """Return files destined for Git, or every file in an exported tree."""
    if (ROOT / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        return {
            ROOT / raw.decode("utf-8")
            for raw in result.stdout.split(b"\0")
            if raw
        }
    return {path for path in ROOT.rglob("*") if path.is_file()}


def main() -> None:
    problems: list[str] = []
    candidates = release_candidates()
    for path in sorted(candidates):
        relative = path.relative_to(ROOT)
        if any(
            relative.parts[:len(prefix)] == prefix
            for prefix in PRIVATE_PATH_PREFIXES
        ):
            problems.append(f"private publication directory: {relative}")
        if path.suffix.lower() in PRIVATE_DOCUMENT_SUFFIXES:
            problems.append(f"private office document: {relative}")
        if path.name in PRIVATE_BASENAMES:
            problems.append(f"OS metadata file: {relative}")
        if path.name == ".env" or (
            path.name.startswith(".env.") and path.name != ".env.example"
        ):
            problems.append(f"private environment file: {relative}")

    included_figures = {
        path.name
        for path in candidates
        if path.parent == ROOT / "resources" / "figures"
    }
    for name in sorted(EXPECTED_FIGURES - included_figures):
        problems.append(f"missing public figure: resources/figures/{name}")
    for name in sorted(included_figures - EXPECTED_FIGURES):
        problems.append(f"unexpected public figure: resources/figures/{name}")

    included_atlases = {
        path.name: path
        for path in candidates
        if path.parent == ROOT / "resources" / "atlases"
        and path.suffix.lower() == ".npz"
    }
    for name in sorted(EXPECTED_ATLASES.keys() - included_atlases.keys()):
        problems.append(f"missing public atlas: resources/atlases/{name}")
    for name in sorted(included_atlases.keys() - EXPECTED_ATLASES.keys()):
        problems.append(f"unexpected public atlas: resources/atlases/{name}")

    for path in sorted(candidates):
        if any(
            part in {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
            for part in path.parts
        ):
            if path.is_file():
                problems.append(f"generated cache: {path.relative_to(ROOT)}")
            continue
        if path.is_symlink():
            problems.append(f"symlink: {path.relative_to(ROOT)}")
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in WEIGHT_SUFFIXES:
            problems.append(f"model weight: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                problems.append(f"{label}: {relative}")
        if path.suffix.lower() == ".ipynb":
            try:
                notebook = json.loads(text)
            except json.JSONDecodeError:
                problems.append(f"invalid notebook JSON: {relative}")
                continue
            for index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("outputs"):
                    problems.append(f"saved notebook output in cell {index}: {relative}")
                if cell.get("attachments"):
                    problems.append(f"notebook attachment in cell {index}: {relative}")

    for name, expected_digest in EXPECTED_ATLASES.items():
        atlas = included_atlases.get(name)
        if atlas is None:
            continue
        digest = hashlib.sha256(atlas.read_bytes()).hexdigest()
        if digest != expected_digest:
            problems.append(f"atlas checksum mismatch: resources/atlases/{name}")
        try:
            with zipfile.ZipFile(atlas) as archive:
                members = archive.infolist()
                if archive.comment or len(members) != 1 or members[0].filename != "mask.npy":
                    problems.append(f"unexpected atlas archive contents: {name}")
                elif members[0].comment or members[0].extra:
                    problems.append(f"unexpected atlas ZIP metadata: {name}")

            import numpy as np

            with np.load(atlas, allow_pickle=False) as archive:
                if archive.files != ["mask"]:
                    problems.append(f"unexpected atlas fields {archive.files}: {name}")
                    continue
                mask = archive["mask"]
            if mask.shape != (2, 96, 96, 96):
                problems.append(f"unexpected atlas shape {mask.shape}: {name}")
            elif mask.dtype.kind not in "biuf" or not np.isfinite(mask).all():
                problems.append(f"non-finite or non-numeric atlas: {name}")
            elif set(np.unique(mask[0]).tolist()) != {0.0, 1.0}:
                problems.append(f"non-binary atlas C2 channel: {name}")
            elif set(np.unique(mask[1]).tolist()) != EXPECTED_ATLAS_LABELS:
                problems.append(f"unexpected atlas anatomical labels: {name}")
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            problems.append(f"unreadable atlas {name}: {error}")

    if problems:
        print("Release audit failed:", *[f"- {item}" for item in problems], sep="\n")
        raise SystemExit(1)
    print(
        "Release audit passed: no local paths, UID values, secrets, symlinks, "
        "or model weights found; public atlases match the approved files."
    )


if __name__ == "__main__":
    main()
