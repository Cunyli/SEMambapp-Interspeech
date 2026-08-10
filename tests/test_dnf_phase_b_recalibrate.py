import pytest

from scripts.recalibrate_dnf_phase_b_probe import (
    recalibrate,
    verify_probe_identity,
)


def row(index: int, *, hard_pass: bool, bak: float) -> dict:
    return {
        "uid": f"uid-{index}",
        "probe_family": "source",
        "dnsmos": {
            "status": "ok",
            "bak": bak,
            "sig": 3.0,
            "ovrl": 3.0,
        },
        "technical_gate": {
            "hard_pass": hard_pass,
            "hard_reasons": [] if hard_pass else ["clipping_above_maximum"],
        },
        "training_ready": False,
    }


def test_recalibration_excludes_hard_fail_and_uses_valid_population():
    rows = [
        row(0, hard_pass=False, bak=1.0),
        row(1, hard_pass=True, bak=2.0),
        row(2, hard_pass=True, bak=3.0),
        row(3, hard_pass=True, bak=4.0),
        row(4, hard_pass=True, bak=5.0),
    ]
    output, summary = recalibrate(rows)
    decisions = {item["uid"]: item["probe_decision"] for item in output}
    assert decisions["uid-0"]["route"] == "exclude_invalid"
    assert (
        summary["groups"]["source"]["technical_hard_pass_bak"]["p25"]
        == 2.75
    )
    assert not any(item["training_ready"] for item in output)


def test_probe_identity_drift_is_rejected():
    probe = [{"key": "a", "shard": "one"}]
    scored = [{"key": "a", "shard": "two"}]
    with pytest.raises(ValueError, match="identities differ"):
        verify_probe_identity(probe, scored)
