#!/usr/bin/env python3
"""Run one local pipeline command and write latency/RAM/VRAM/storage profile JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import configured_data_root, load_yaml_config, repository_root
from hardening.profiling import profile_subprocess
from hardening.reproducibility import configure_determinism


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Override configured data root")
    parser.add_argument("--seed", type=int, help="Deterministic child-process seed")
    parser.add_argument("--storage-path", action="append", type=Path, default=[], help="Path measured after run")
    parser.add_argument("--output", type=Path, help="Profile JSON destination")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --, for example -- python scripts/run_kis.py ...")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    command = tuple(args.command)
    while command[:1] == ("--",):
        command = command[1:]
    if not command:
        raise SystemExit("Supply a command after --")
    root = repository_root()
    data_config = load_yaml_config(root / "configs" / "data.yaml")
    hardening_config = load_yaml_config(root / "configs" / "hardening.yaml")
    data_root = configured_data_root(data_config, args.data_root)
    seed = args.seed if args.seed is not None else int(hardening_config["reproducibility"]["random_seed"])
    deterministic = configure_determinism(seed)
    storage_paths = tuple(args.storage_path) or (data_root,)
    try:
        completed, profile = profile_subprocess(command, storage_paths=storage_paths)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    output = args.output or root / str(hardening_config["profile"]["output_dir"]) / "profile.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "command": list(command),
                "returncode": completed.returncode,
                "seed": deterministic.seed,
                "torch_determinism_configured": deterministic.torch_configured,
                "torch_determinism_error": deterministic.torch_error,
                "profile": profile.as_dict(),
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(f"Profile JSON: {output}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
