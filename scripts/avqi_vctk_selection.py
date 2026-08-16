"""Deterministic exact-valid selection for the AVQI VCTK external panel."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


CONDITIONS = ("clean", "rir_only", "snr20", "snr10")


def select_exact_complete_external_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    required_utterances_per_speaker: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep the first exact-complete external utterances in input order.

    Non-external rows are retained unchanged. An external utterance is eligible
    only when each frozen condition occurs exactly once and exact Praat reports
    ``scoring_status=ok`` for all four rows. Metric values never participate in
    selection.
    """
    if required_utterances_per_speaker <= 0:
        raise ValueError("required utterances per speaker must be positive")

    candidate_order: dict[str, list[str]] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    normalized_rows = [dict(row) for row in rows]
    for row in normalized_rows:
        required_fields = (
            "split",
            "speaker_id",
            "sample_id",
            "condition",
            "scoring_status",
        )
        missing = [field for field in required_fields if field not in row]
        if missing:
            raise ValueError(f"scored row missing fields: {missing}")
        if row["split"] != "vctk_external":
            continue
        speaker = str(row["speaker_id"])
        sample = str(row["sample_id"])
        key = (speaker, sample)
        if key not in grouped:
            grouped[key] = []
            candidate_order.setdefault(speaker, []).append(sample)
        grouped[key].append(row)

    if not candidate_order:
        raise ValueError("no VCTK external candidate rows")

    selected_keys: set[tuple[str, str]] = set()
    speaker_receipts: dict[str, Any] = {}
    replacement_count = 0
    for speaker, samples in candidate_order.items():
        complete_samples: list[str] = []
        sample_receipts: dict[str, Any] = {}
        for sample in samples:
            sample_rows = grouped[(speaker, sample)]
            conditions = [str(row["condition"]) for row in sample_rows]
            if len(sample_rows) != len(CONDITIONS) or set(conditions) != set(
                CONDITIONS
            ):
                raise ValueError(
                    "external candidate does not contain exactly one row per "
                    f"condition: {speaker}/{sample}={conditions}"
                )
            exact_complete = all(
                row["scoring_status"] == "ok" for row in sample_rows
            )
            sample_receipts[sample] = {
                "exact_complete": exact_complete,
                "error_types": sorted(
                    {
                        str(row.get("error_type", ""))
                        for row in sample_rows
                        if row["scoring_status"] != "ok"
                    }
                ),
            }
            if exact_complete:
                complete_samples.append(sample)
        selected_samples = complete_samples[:required_utterances_per_speaker]
        if len(selected_samples) != required_utterances_per_speaker:
            raise ValueError(
                f"speaker {speaker} has only {len(selected_samples)} "
                "exact-complete external utterances"
            )
        selected_keys.update((speaker, sample) for sample in selected_samples)
        frozen_primary = samples[:required_utterances_per_speaker]
        replacements = [
            sample for sample in selected_samples if sample not in frozen_primary
        ]
        replaced = [
            sample for sample in frozen_primary if sample not in selected_samples
        ]
        if len(replacements) != len(replaced):
            raise ValueError(f"replacement accounting mismatch for {speaker}")
        replacement_count += len(replacements)
        speaker_receipts[speaker] = {
            "candidate_sample_ids": samples,
            "selected_sample_ids": selected_samples,
            "replaced_sample_ids": replaced,
            "replacement_sample_ids": replacements,
            "sample_status": sample_receipts,
        }

    selected_rows = [
        row
        for row in normalized_rows
        if row["split"] != "vctk_external"
        or (str(row["speaker_id"]), str(row["sample_id"])) in selected_keys
    ]
    selected_external = [
        row for row in selected_rows if row["split"] == "vctk_external"
    ]
    expected_external_rows = (
        len(candidate_order)
        * required_utterances_per_speaker
        * len(CONDITIONS)
    )
    if len(selected_external) != expected_external_rows:
        raise ValueError(
            f"selected external row mismatch: {len(selected_external)} != "
            f"{expected_external_rows}"
        )
    if any(row["scoring_status"] != "ok" for row in selected_external):
        raise ValueError("selected external rows are not exact-complete")
    receipt = {
        "policy": "first_ranked_all_four_conditions_exact_valid",
        "metric_values_used_for_selection": False,
        "required_conditions": list(CONDITIONS),
        "required_utterances_per_speaker": required_utterances_per_speaker,
        "speaker_count": len(candidate_order),
        "candidate_external_rows": sum(
            row["split"] == "vctk_external" for row in normalized_rows
        ),
        "selected_external_rows": len(selected_external),
        "selected_external_valid_rows": len(selected_external),
        "replacement_count": replacement_count,
        "speakers": speaker_receipts,
    }
    return selected_rows, receipt
