from __future__ import annotations

from pathlib import Path

import pytest
import soundfile as sf

from scripts.evaluate_avqi_shimmer_db_trust_region_v16 import (
    ALPHA_LADDER,
    FIXED_ALPHA,
    PCM24_MIN_CHANGED_SAMPLES,
    PROTOTYPE_CASE_IDS,
    SELECTOR_KEYS,
    TRUST_REGION_CANDIDATE_NAME,
    pcm24_effective_step,
    select_topology_certified_step,
    selector_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def selector_row(index: int, *, topology: bool = True) -> dict[str, object]:
    return {
        "alpha": ALPHA_LADDER[index],
        "backtrack_index": index,
        "topology_stability_pass": topology,
        "finite_safety_pass": True,
        "proxy_nonregression_pass": True,
        "pcm24_effective_step_pass": True,
    }


def test_schedule_is_fixed_half_step_ladder_with_four_case_prototype() -> None:
    assert FIXED_ALPHA == 0.001
    assert ALPHA_LADDER == (0.001, 0.0005, 0.00025, 0.000125)
    assert len(PROTOTYPE_CASE_IDS) == 4
    assert PROTOTYPE_CASE_IDS == (
        "sealed_final__FD23__cs__rir_only",
        "sealed_final__PD_37__cs__snr20",
        "sealed_final__FD23__sv__snr20",
        "sealed_final__PD_37__sv__snr10",
    )
    assert "pulse_linear" not in TRUST_REGION_CANDIDATE_NAME


def test_selector_returns_largest_certified_step() -> None:
    attempts = [
        selector_row(0, topology=False),
        selector_row(1),
        selector_row(2),
        selector_row(3),
    ]
    selected = select_topology_certified_step(attempts)
    assert selected is not None
    assert selected["alpha"] == 0.0005
    assert selected["backtrack_index"] == 1


def test_selector_fails_closed_and_rejects_exact_outcome_fields() -> None:
    attempts = [selector_row(index, topology=False) for index in range(4)]
    assert select_topology_certified_step(attempts) is None
    attempts[0]["candidate_exact_shimmer_db"] = 0.1
    with pytest.raises(ValueError, match="selector input contract drift"):
        select_topology_certified_step(attempts)
    assert "candidate_exact_shimmer_db" not in SELECTOR_KEYS


def test_selector_contract_counts_every_candidate_and_keeps_500ms() -> None:
    contract = selector_contract()
    assert contract["worker_count"] == 4
    assert contract["formal_total_metric_step_runtime_ms"] == 500.0
    assert "all_four_pcm24_writes" in contract["runtime_includes"]
    assert "all_four_candidate_exact_topology_refreshes" in contract[
        "runtime_includes"
    ]
    assert "candidate_exact_shimmer_db" in contract["forbidden_information"]


def test_pcm24_effective_step_rejects_noop_and_accepts_quantized_change(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.wav"
    no_op_path = tmp_path / "no_op.wav"
    changed_path = tmp_path / "changed.wav"
    base = [0.0] * 16_000
    changed = [2.0 / (2**23)] * 16_000
    sf.write(base_path, base, 16_000, subtype="PCM_24")
    sf.write(no_op_path, base, 16_000, subtype="PCM_24")
    sf.write(changed_path, changed, 16_000, subtype="PCM_24")
    assert pcm24_effective_step(base_path, no_op_path)[
        "pcm24_effective_step_pass"
    ] is False
    result = pcm24_effective_step(base_path, changed_path)
    assert result["pcm24_changed_samples"] >= PCM24_MIN_CHANGED_SAMPLES
    assert result["pcm24_effective_step_pass"] is True


def test_selector_seal_precedes_candidate_exact_scoring() -> None:
    source = (
        REPO_ROOT / "scripts" / "evaluate_avqi_shimmer_db_trust_region_v16.py"
    ).read_text(encoding="utf-8")
    seal = 'write_json(selector_seal_path, selector_seal)'
    score = "exact_after = run_exact(exact_items"
    assert source.index(seal) < source.index(score)
    assert '"candidate_exact_outcomes_present": False' in source
    assert '"generator_optimizer_steps": 0' in source


def test_runner_is_hash_bound_and_does_not_train_generator() -> None:
    source = (
        REPO_ROOT / "scripts" / "run_avqi_shimmer_db_trust_region_v16.sh"
    ).read_text(encoding="utf-8")
    assert "PANEL_CONTRACT_SHA256" in source
    assert "FRESH_RESULTS_SHA256" in source
    assert "RUNTIME_WORKER_SCRIPT_SHA256" in source
    assert "CONFIRM_SLURM_SUBMIT" in source
    assert "generator" not in source.lower()
