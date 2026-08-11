#!/usr/bin/env python3
"""Preview or explicitly remove configured model artifacts after a benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import configured_data_root, load_yaml_config, repository_root
from hardening.cleanup import StorageCleanupError, execute_storage_cleanup, plan_storage_cleanup


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", action="append", default=[], help="Configured artifact target; repeat as needed")
    parser.add_argument("--delete", action="store_true", help="Actually delete the selected paths")
    parser.add_argument("--data-root", type=Path, help="Override configured data root")
    parser.add_argument("--config", type=Path, help="Hardening config path")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    root = repository_root()
    data_config = load_yaml_config(root / "configs" / "data.yaml")
    hardening_config = load_yaml_config(args.config or root / "configs" / "hardening.yaml")
    data_root = configured_data_root(data_config, args.data_root)
    try:
        targets = hardening_config["storage_cleanup"]["targets"]
        plan = plan_storage_cleanup(data_root, targets, args.encoder)
        for target in plan.targets:
            print(f"{target.name}: {target.bytes_on_disk} bytes")
            for path in target.paths:
                print(f"  {path}")
        print(f"Total: {plan.total_bytes} bytes")
        if args.delete:
            removed = execute_storage_cleanup(plan, delete=True)
            print(f"Removed {len(removed)} path(s)")
        else:
            print("Dry run only. Re-run with --delete to remove these paths.")
    except (KeyError, StorageCleanupError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
