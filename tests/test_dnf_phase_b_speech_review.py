import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_dnf_phase_b_speech_review_pack.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_dnf_phase_b_speech_review_pack",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(group: str, index: int, bak: float) -> dict:
    return {
        "key": f"{group}-{index}",
        "probe_family": group,
        "dnsmos": {"status": "ok", "bak": bak},
        "technical_gate": {"hard_pass": True},
        "probe_decision": {"route": "clean_candidate"},
    }


def test_select_strata_balances_groups_and_avoids_duplicates():
    rows = [
        row(group, index, float(index))
        for group in ("mls", "libri")
        for index in range(20)
    ]
    selected = MODULE.select_strata(rows, per_stratum=3)
    assert len(selected) == 18
    assert len({item["key"] for item in selected}) == 18
    for group in ("mls", "libri"):
        for stratum in ("low_bak", "p25_boundary", "high_bak"):
            assert (
                sum(
                    item["review_group"] == group
                    and item["review_stratum"] == stratum
                    for item in selected
                )
                == 3
            )


def test_select_strata_excludes_technical_failures():
    rows = [row("mls", index, float(index)) for index in range(20)]
    rows[0]["technical_gate"]["hard_pass"] = False
    rows[0]["probe_decision"]["route"] = "exclude_invalid"
    selected = MODULE.select_strata(rows, per_stratum=2)
    assert rows[0]["key"] not in {item["key"] for item in selected}


def test_blind_ids_are_deterministic_and_hide_score_strata():
    rows = [
        row(group, index, float(index))
        for group in ("mls", "libri")
        for index in range(6)
    ]
    selected = MODULE.select_strata(rows, per_stratum=1)
    first = MODULE.assign_blind_ids(selected, 3407)
    second = MODULE.assign_blind_ids(list(reversed(selected)), 3407)
    assert first == second
    assert [item["blind_id"] for item in first] == [
        f"clip_{index:04d}" for index in range(1, len(first) + 1)
    ]
    assert all(
        item["review_group"] not in item["blind_id"]
        and item["review_stratum"] not in item["blind_id"]
        for item in first
    )
