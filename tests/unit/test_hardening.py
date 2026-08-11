from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from experiments.logging import ExperimentLogError, ExperimentLogger, ExperimentRecord, dataset_snapshot_descriptor
from hardening.cache import CacheCorruptionError, JsonResultCache
from hardening.cleanup import StorageCleanupError, execute_storage_cleanup, plan_storage_cleanup
from hardening.profiling import profile_callable
from hardening.reproducibility import configure_determinism


class HardeningTests(unittest.TestCase):
    def test_seed_cache_profile_cleanup_and_experiment_log_are_explicit(self) -> None:
        configure_determinism(2026)
        first = (random.random(), float(np.random.random()))
        configure_determinism(2026)
        self.assertEqual(first, (random.random(), float(np.random.random())))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = JsonResultCache(root / "cache")
            calls = 0

            def factory() -> dict[str, int]:
                nonlocal calls
                calls += 1
                return {"value": calls}

            self.assertEqual(cache.get_or_compute("query", factory), {"value": 1})
            self.assertEqual(cache.get_or_compute("query", factory), {"value": 1})
            self.assertEqual(calls, 1)
            next((root / "cache").glob("*.json")).write_text("broken", encoding="utf-8")
            with self.assertRaises(CacheCorruptionError):
                cache.get("query")

            artifact = root / "data" / "processed" / "embeddings" / "fg"
            artifact.mkdir(parents=True)
            (artifact / "feature.npy").write_bytes(b"1234")
            plan = plan_storage_cleanup(
                root / "data",
                {"fg": ["processed/embeddings/fg"]},
                ("fg",),
            )
            self.assertEqual(plan.total_bytes, 4)
            with self.assertRaises(StorageCleanupError):
                execute_storage_cleanup(plan)
            self.assertTrue(artifact.exists())
            execute_storage_cleanup(plan, delete=True)
            self.assertFalse(artifact.exists())

            profile_value, profile = profile_callable(lambda: sum(range(100)), storage_paths=(root / "data",))
            self.assertEqual(profile_value, 4950)
            self.assertGreaterEqual(profile.latency_ms, 0)
            self.assertGreaterEqual(profile.storage_bytes, 0)

            snapshot = root / "snapshot.json"
            snapshot.write_text("{}", encoding="utf-8")
            record = ExperimentRecord(
                experiment_id="fixture-1",
                task="kis",
                dataset_snapshot=dataset_snapshot_descriptor(snapshot),
                encoder="fgclip2_large",
                sources=("fgclip2",),
                model_revisions={"fgclip2": "fixture"},
                config={"retrieval": "hash"},
                metrics={"R@1": 0.1, "R@5": 0.2, "R@20": 0.3, "R@50": 0.4, "R@100": 0.5, "final_score": 0.3},
                latency_ms={"pipeline": 2.0},
            )
            logger = ExperimentLogger(root / "experiments.jsonl")
            logger.append(record)
            with self.assertRaises(ExperimentLogError):
                logger.append(record)

    def test_cleanup_rejects_paths_outside_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(StorageCleanupError):
                plan_storage_cleanup(Path(temporary) / "data", {"bad": ["../outside"]}, ("bad",))
