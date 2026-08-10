from scripts.build_dnf_phase_b_candidate_proposals import (
    libri_item_proposals,
    mls_shard_proposals,
)


def scored_row(shard: str, index: int, bak: float, hard_pass: bool = True):
    return {
        "_shard_dir": "/data",
        "shard": shard,
        "key": f"{shard}-{index}",
        "dnsmos": {"bak": bak},
        "technical_gate": {"hard_pass": hard_pass},
    }


def test_mls_strict_proposal_remains_review_only():
    rows = [
        scored_row("a.tar", index, 4.5) for index in range(4)
    ] + [
        scored_row("b.tar", index, 3.0) for index in range(4)
    ]
    shard_rows = [
        {"_shard_dir": "/data", "shard": "a.tar", "sample_count": 100},
        {"_shard_dir": "/data", "shard": "b.tar", "sample_count": 100},
    ]
    strict, _, summary = mls_shard_proposals(
        rows,
        shard_rows,
        expected_items_per_shard=4,
    )
    assert len(strict) == 1
    assert strict[0]["shard"] == "a.tar"
    assert not strict[0]["training_ready"]
    assert summary["strict_samples"] == 100


def test_libri_technical_fail_is_not_item_candidate():
    rows = [
        scored_row("a.tar", 0, 1.0, hard_pass=False),
        scored_row("a.tar", 1, 2.0),
        scored_row("a.tar", 2, 3.0),
        scored_row("a.tar", 3, 4.0),
        scored_row("a.tar", 4, 5.0),
    ]
    candidates, summary = libri_item_proposals(rows)
    assert rows[0]["key"] not in {row["key"] for row in candidates}
    assert not any(row["training_ready"] for row in candidates)
    assert not summary["shard_level_promotion"]
