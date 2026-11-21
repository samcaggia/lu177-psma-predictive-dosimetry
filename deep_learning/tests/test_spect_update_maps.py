"""M3 runner: a parallel chunk that receives no phantom writes an empty result and returns.

Run with the CPU torch environment: .venv-torch/bin/python -m unittest discover -s deep_learning/tests
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import spect_update_maps as m3  # noqa: E402


class FakeCohort:
    def __init__(self, *args, **kwargs):
        self.phantoms = pd.DataFrame({"study_id": ["psma_x_2020-01-01"], "subject": ["x"], "grid_mm": [4.0]})
        self.index = {"psma_x_2020-01-01": 0}
        self.n_lesions = [0]


class EmptyChunkTest(unittest.TestCase):
    def run_part(self, folder, part, parts):
        (folder / "gated_prior_A.json").write_text(json.dumps({"subject_half": {"x": "tuning"}}))
        argv = ["spect_update_maps.py", "tuning", "2", "48", "5", "0.02", "1", "0", "mismatch", "0.15", f"{part}/{parts}"]
        with mock.patch.object(m3, "EVAL", folder), mock.patch.object(m3, "OUT", folder / "spect"), \
                mock.patch.object(m3, "GpuCohort", FakeCohort), mock.patch.object(m3, "import_net_data", lambda path: None), \
                mock.patch.object(m3, "provenance", lambda *a: {"code_commit": "test", "net_data_md5": "test"}), \
                mock.patch.object(m3.update, "case_data", side_effect=AssertionError("an empty chunk must not load cases")), \
                mock.patch.object(sys, "argv", argv):
            m3.main()

    def test_empty_chunk_writes_header_only_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.run_part(folder, 1, 2)
            path = folder / "spect" / "maps_rows_tuning_mismatch_A_lr0.02_rs1.1.csv"
            self.assertTrue(path.exists())
            table = pd.read_csv(path)
            self.assertEqual(len(table), 0)
            self.assertEqual(list(table.columns[: len(m3.ROW_COLUMNS)]), m3.ROW_COLUMNS)
            self.assertIn("residual_scale", table.columns)

    def test_empty_chunk_joins_with_a_full_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.run_part(folder, 1, 2)
            empty = pd.read_csv(folder / "spect" / "maps_rows_tuning_mismatch_A_lr0.02_rs1.1.csv")
            full = pd.DataFrame([dict.fromkeys(empty.columns, 1)])
            joined = pd.concat([full, empty], ignore_index=True)
            self.assertEqual(len(joined), 1)
            self.assertEqual(list(joined.columns), list(empty.columns))


if __name__ == "__main__":
    unittest.main()
