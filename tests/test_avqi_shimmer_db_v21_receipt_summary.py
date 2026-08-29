from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


WRAPPER_PATH = Path(
    "/scratch/work/lil14/avqi_route_c_diagnostic_launchers/"
    "shimmer_db_v21_candidate_d_deterministic_repeat/"
    "run_avqi_shimmer_db_candidate_d_deterministic_repeat_v21.py"
)


def load_wrapper_module():
    spec = importlib.util.spec_from_file_location(
        "avqi_shimmer_db_v21_wrapper",
        WRAPPER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load wrapper module: {WRAPPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class V21ReceiptSummaryTest(unittest.TestCase):
    def test_successful_summary_serializes_authoritative_repeat_field(self) -> None:
        wrapper = load_wrapper_module()
        report = {
            "decision": "DETERMINISTIC_FULL_HASH_REPEAT_ESTABLISHED_ONE_CASE_BOUND_CONTRACT",
            "v20_v21_repeat_comparison": {
                "output_wav_pcm24_byte_repeat_observed": True,
            },
        }

        summary = wrapper.build_final_summary(report)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertEqual(
            json.loads(serialized)["v20_v21_byte_equivalence_observed"],
            True,
        )
        self.assertEqual(
            summary["authoritative_training_decision"],
            "NO_GO_AVQI_T2_TRAINING",
        )

    def test_summary_rejects_missing_or_legacy_repeat_field(self) -> None:
        wrapper = load_wrapper_module()

        for repeat_comparison in ({}, {"byte_equivalence_observed": True}):
            with self.subTest(repeat_comparison=repeat_comparison):
                with self.assertRaises(KeyError):
                    wrapper.build_final_summary(
                        {
                            "decision": "diagnostic",
                            "v20_v21_repeat_comparison": repeat_comparison,
                        }
                    )

    def test_summary_rejects_non_boolean_repeat_field(self) -> None:
        wrapper = load_wrapper_module()

        with self.assertRaises(TypeError):
            wrapper.build_final_summary(
                {
                    "decision": "diagnostic",
                    "v20_v21_repeat_comparison": {
                        "output_wav_pcm24_byte_repeat_observed": "true",
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()
