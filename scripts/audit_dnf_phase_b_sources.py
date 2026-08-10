#!/usr/bin/env python3
"""Build fail-closed Phase B source allowlists and deterministic score probes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import re
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "dnf_phase_b_source_audit_v2"
SPLIT_BITS = {"train": 1, "valid": 2, "test": 4}
SPLIT_MASK_NAMES = {
    1: "train_only",
    2: "valid_only",
    3: "train_valid",
    4: "test_only",
    5: "train_test",
    6: "valid_test",
    7: "train_valid_test",
}

LIBRI_DATA_WAV_TOKEN = "/data_wav_24k_10-folders/"
LIBRI_SIMULATED_ORIGINAL_RE = re.compile(r"/simulated-data/\d+/original/")
LIBRI_G711_RE = re.compile(r"/simulated-data/\d+/g711/")
LIBRI_TTV_TOKEN = "/train-test-validation/"

DNS_FREESOUND_FAN_RE = re.compile(
    r"(?:^|/)datasets_fullband\.noise_fullband\.freesound_\d{3}/"
    r".*/fan_Freesound_validated_.*\.wav$"
)

ARNI_RIR_RE = re.compile(
    r"(?:^|/)IR_numClosed_\d+_numComb_\d+_mic_\d+_sweep_\d+\.wav$"
)
SLR26_RIR_RE = re.compile(
    r"/SLR26/simulated_rirs_48k/(smallroom|mediumroom|largeroom)/"
    r"Room\d+/Room\d+-\d+\.wav$"
)
SLR28_RIR_DIR_TOKEN = "/SLR28/RIRS_NOISES/real_rirs_isotropic_noises/"
SLR28_RIR_NAME_RE = re.compile(r"(?:^|/)(?:RWCP_.*_rir_.*|air_.*)\.wav$")

FSD_TARGET_DIRECT = {
    "Mechanical_fan",
    "Traffic_noise_and_roadway_noise",
    "Car_passing_by",
}
FSD_ENGINE_TARGET = {"Engine", "Idling"}
FSD_ROAD_CONTEXT = {
    "Motor_vehicle_(road)",
    "Car",
    "Bus",
    "Truck",
    "Motorcycle",
    "Race_car_and_auto_racing",
}
FSD_VARIABLE_VEHICLE_REASONS = {
    "fsd50k_roadway_noise_fail_closed",
    "fsd50k_road_engine_fail_closed",
}
FSD_FORBIDDEN_EXACT = {
    "Human_voice",
    "Speech",
    "Conversation",
    "Chatter",
    "Crowd",
    "Singing",
    "Female_singing",
    "Male_singing",
    "Speech_synthesizer",
    "Music",
    "Musical_instrument",
    "Animal",
    "Wild_animals",
    "Domestic_animals_and_pets",
    "Livestock_and_farm_animals_and_working_animals",
    "Alarm",
    "Siren",
    "Vehicle_horn_and_car_horn_and_honking",
    "Bell",
    "Church_bell",
    "Doorbell",
    "Bicycle_bell",
    "Ringtone",
    "Chime",
    "Wind_chime",
    "Gunshot_and_gunfire",
    "Explosion",
    "Fireworks",
    "Crack",
    "Crash_cymbal",
    "Knock",
    "Slam",
    "Shatter",
    "Thump_and_thud",
    "Hammer",
    "Door",
    "Sliding_door",
    "Engine_starting",
    "Aircraft",
    "Fixed-wing_aircraft_and_airplane",
    "Boat_and_Water_vehicle",
    "Train",
    "Subway_and_metro_and_underground",
}
FSD_FORBIDDEN_FRAGMENTS = (
    "speech",
    "voice",
    "singing",
    "instrument",
    "animal",
    "bird",
    "dog",
    "cat",
    "frog",
    "insect",
    "rooster",
    "laughter",
    "screaming",
    "shout",
    "yell",
    "crying",
    "cough",
    "sneeze",
    "breathing",
    "gasp",
    "sigh",
    "whisper",
    "cheering",
    "clapping",
    "applause",
)


JsonRow = dict[str, Any]
ShardKey = tuple[str, str]


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def stable_rank(seed: int, *parts: object) -> int:
    payload = "\0".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), byteorder="big")


def sha256_file(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size_bytes = 0
    line_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size_bytes += len(chunk)
            line_count += chunk.count(b"\n")
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": size_bytes,
        "line_count": line_count,
    }


class InputReceiptTracker:
    """Track hashes while JSONL inputs are consumed exactly once."""

    def __init__(self) -> None:
        self.receipts: dict[str, dict[str, Any]] = {}

    def iter_jsonl(self, path: Path) -> Iterator[JsonRow]:
        digest = hashlib.sha256()
        size_bytes = 0
        line_count = 0
        with path.open("rb") as handle:
            for raw_line in handle:
                digest.update(raw_line)
                size_bytes += len(raw_line)
                line_count += 1
                yield json.loads(raw_line)
        self.receipts[str(path)] = {
            "path": str(path),
            "sha256": digest.hexdigest(),
            "size_bytes": size_bytes,
            "line_count": line_count,
        }

    def add_file(self, path: Path) -> None:
        self.receipts[str(path)] = sha256_file(path)

    def sorted_receipts(self) -> list[dict[str, Any]]:
        return [self.receipts[path] for path in sorted(self.receipts)]


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row))
            count += 1
    return count


def normalize_source_name(source: str) -> str:
    return re.sub(r"_chunk\d+(?:_patch)?$", "", source)


def shard_sources(row: Mapping[str, Any]) -> set[str]:
    dataset_counts = row.get("dataset_counts")
    if isinstance(dataset_counts, dict) and dataset_counts:
        return {normalize_source_name(str(name)) for name in dataset_counts}
    source = row.get("dataset") or row.get("source") or ""
    return {normalize_source_name(str(source))}


def shard_key(row: Mapping[str, Any]) -> ShardKey:
    return str(row["_shard_dir"]), str(row["shard"])


def item_identity(row: Mapping[str, Any]) -> str:
    return "|".join(
        str(row.get(field, ""))
        for field in ("dataset", "_shard_dir", "shard", "audio_member", "key")
    )


def load_split_rows(
    split_root: Path,
    split: str,
    role: str,
    tracker: InputReceiptTracker,
) -> list[JsonRow]:
    path = split_root / split / f"{role}_shards.jsonl"
    rows = list(tracker.iter_jsonl(path))
    for row in rows:
        row["_split"] = split
    return rows


def select_source_shards(
    rows: Iterable[JsonRow],
    source_name: str,
) -> list[JsonRow]:
    selected = []
    for row in rows:
        sources = shard_sources(row)
        if source_name in sources:
            if sources != {source_name}:
                raise ValueError(
                    f"Mixed-source shard cannot be fail-closed selected: {sources} {row}"
                )
            selected.append(row)
    return selected


def iter_selected_items(
    selected_shards: Sequence[JsonRow],
    tracker: InputReceiptTracker,
) -> Iterator[JsonRow]:
    split_by_key: dict[ShardKey, str] = {}
    role_dirs: set[Path] = set()
    for shard in selected_shards:
        key = shard_key(shard)
        split = str(shard["_split"])
        previous = split_by_key.setdefault(key, split)
        if previous != split:
            raise ValueError(f"Shard occurs in multiple splits: {key}")
        role_dirs.add(Path(key[0]))

    seen_keys: Counter[ShardKey] = Counter()
    for role_dir in sorted(role_dirs):
        manifest_path = role_dir / "manifest.jsonl"
        for item in tracker.iter_jsonl(manifest_path):
            key = (str(role_dir), str(item["shard"]))
            split = split_by_key.get(key)
            if split is None:
                continue
            updated = dict(item)
            updated["_shard_dir"] = str(role_dir)
            updated["_split"] = split
            seen_keys[key] += 1
            yield updated

    empty = sorted(key for key in split_by_key if seen_keys[key] == 0)
    if empty:
        raise ValueError(f"Selected shards have no manifest items: {empty[:10]}")


def classify_libri_provenance(source_path: str) -> str:
    normalized = source_path.replace("\\", "/")
    if LIBRI_DATA_WAV_TOKEN in normalized:
        return "data_wav_24k"
    if LIBRI_SIMULATED_ORIGINAL_RE.search(normalized):
        return "simulated_original"
    if LIBRI_G711_RE.search(normalized):
        return "g711"
    if LIBRI_TTV_TOKEN in normalized:
        return "train_test_validation"
    return "unknown"


def libri_content_group_id(source_path: str, provenance: str | None = None) -> str:
    selected_provenance = provenance or classify_libri_provenance(source_path)
    stem = Path(source_path).stem
    if selected_provenance == "train_test_validation":
        stripped = re.sub(r"_\d+$", "", stem)
        if stripped == stem:
            raise ValueError(f"TTv path has no removable variant suffix: {source_path}")
        stem = stripped
    if not stem:
        raise ValueError(f"Cannot derive Libri content group: {source_path}")
    return stem


def candidate_pending_row(
    row: Mapping[str, Any],
    *,
    content_group_id: str,
    selection_reason: str,
) -> JsonRow:
    return {
        "schema_version": SCHEMA_VERSION,
        "route": "clean_candidate",
        "audit_status": "audit_pending_scores",
        "training_ready": False,
        "content_group_id": content_group_id,
        "selection_reason": selection_reason,
        "dataset": str(row.get("dataset", "")),
        "split": str(row.get("_split", "")),
        "_shard_dir": str(row.get("_shard_dir", "")),
        "shard": str(row.get("shard", "")),
        "audio_member": str(row.get("audio_member", "")),
        "json_member": str(row.get("json_member", "")),
        "key": str(row.get("key", "")),
        "source_path": str(row.get("source_path", "")),
    }


def build_libri_candidate_rows(
    items: Iterable[JsonRow],
) -> tuple[list[JsonRow], dict[str, Any]]:
    group_masks: dict[str, int] = {}
    canonical_rows: dict[str, JsonRow] = {}
    canonical_counts: Counter[str] = Counter()
    provenance_by_split: Counter[tuple[str, str]] = Counter()

    for item in items:
        split = str(item["_split"])
        if split not in SPLIT_BITS:
            raise ValueError(f"Unsupported split: {split}")
        source_path = str(item["source_path"])
        provenance = classify_libri_provenance(source_path)
        group_id = libri_content_group_id(source_path, provenance)
        group_masks[group_id] = group_masks.get(group_id, 0) | SPLIT_BITS[split]
        provenance_by_split[(split, provenance)] += 1

        if split != "train" or provenance != "data_wav_24k":
            continue
        canonical_counts[group_id] += 1
        current = canonical_rows.get(group_id)
        candidate_key = (
            source_path,
            str(item.get("shard", "")),
            str(item.get("audio_member", "")),
            str(item.get("key", "")),
        )
        if current is None:
            canonical_rows[group_id] = dict(item)
            continue
        current_key = (
            str(current["source_path"]),
            str(current.get("shard", "")),
            str(current.get("audio_member", "")),
            str(current.get("key", "")),
        )
        if candidate_key < current_key:
            canonical_rows[group_id] = dict(item)

    mask_counts = Counter(group_masks.values())
    allowlist = []
    for group_id in sorted(canonical_rows):
        if group_masks[group_id] != SPLIT_BITS["train"]:
            continue
        row = candidate_pending_row(
            canonical_rows[group_id],
            content_group_id=group_id,
            selection_reason="libri_train_only_unique_canonical_data_wav_24k",
        )
        row["provenance"] = "data_wav_24k"
        allowlist.append(row)

    summary = {
        "all_content_groups": len(group_masks),
        "content_groups_by_split_mask": {
            SPLIT_MASK_NAMES[mask]: mask_counts.get(mask, 0)
            for mask in sorted(SPLIT_MASK_NAMES)
        },
        "cross_split_content_groups": sum(
            count
            for mask, count in mask_counts.items()
            if mask not in {SPLIT_BITS["train"], SPLIT_BITS["valid"], SPLIT_BITS["test"]}
        ),
        "provenance_by_split": {
            split: {
                provenance: provenance_by_split.get((split, provenance), 0)
                for provenance in (
                    "data_wav_24k",
                    "simulated_original",
                    "g711",
                    "train_test_validation",
                    "unknown",
                )
            }
            for split in ("train", "valid", "test")
        },
        "train_data_wav_groups": len(canonical_rows),
        "train_data_wav_duplicate_groups": sum(
            count > 1 for count in canonical_counts.values()
        ),
        "canonical_pending_score_candidates": len(allowlist),
        "candidate_contract": {
            "route": "clean_candidate",
            "audit_status": "audit_pending_scores",
            "training_ready": False,
        },
    }
    return allowlist, summary


def deterministic_select(
    rows: Sequence[JsonRow],
    count: int,
    seed: int,
    namespace: str,
) -> list[JsonRow]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if len(rows) < count:
        raise ValueError(
            f"Insufficient rows for {namespace}: requested={count} available={len(rows)}"
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            stable_rank(seed, namespace, item_identity(row)),
            item_identity(row),
        ),
    )
    return [dict(row) for row in ranked[:count]]


def select_mls_probe_shards(
    shards: Sequence[JsonRow],
    count: int,
    seed: int,
) -> list[JsonRow]:
    if len(shards) < count:
        raise ValueError(
            f"Insufficient MLS shards: requested={count} available={len(shards)}"
        )
    ranked = sorted(
        shards,
        key=lambda row: (
            stable_rank(seed, "mls_shard", *shard_key(row)),
            shard_key(row),
        ),
    )
    return [dict(row) for row in ranked[:count]]


def build_mls_probe_rows(
    selected_shards: Sequence[JsonRow],
    items: Iterable[JsonRow],
    items_per_shard: int,
    seed: int,
) -> list[JsonRow]:
    selected_keys = {shard_key(row) for row in selected_shards}
    heaps: dict[ShardKey, list[tuple[int, str, JsonRow]]] = {
        key: [] for key in selected_keys
    }
    for item in items:
        key = shard_key(item)
        if key not in heaps:
            continue
        identity = item_identity(item)
        rank = stable_rank(seed, "mls_item", *key, identity)
        entry = (-rank, identity, dict(item))
        heap = heaps[key]
        if len(heap) < items_per_shard:
            heapq.heappush(heap, entry)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, entry)

    shard_order = {shard_key(row): index for index, row in enumerate(selected_shards)}
    output = []
    for key in sorted(selected_keys, key=shard_order.__getitem__):
        heap = heaps[key]
        if len(heap) != items_per_shard:
            raise ValueError(
                f"MLS shard has insufficient probe items: {key} "
                f"expected={items_per_shard} found={len(heap)}"
            )
        selected = sorted(
            (entry[2] for entry in heap),
            key=lambda row: (
                stable_rank(seed, "mls_item", *key, item_identity(row)),
                item_identity(row),
            ),
        )
        for item_rank, item in enumerate(selected):
            group_id = Path(str(item["source_path"])).stem
            row = candidate_pending_row(
                item,
                content_group_id=group_id,
                selection_reason="mls_deterministic_quality_probe",
            )
            row.update(
                {
                    "probe_family": "MLS_HQ_en",
                    "selection_seed": seed,
                    "selected_shard_rank": shard_order[key],
                    "selected_item_rank": item_rank,
                }
            )
            output.append(row)
    return output


def build_libri_probe_rows(
    candidates: Sequence[JsonRow],
    count: int,
    seed: int,
) -> list[JsonRow]:
    selected = deterministic_select(candidates, count, seed, "libri_candidate_probe")
    output = []
    for index, row in enumerate(selected):
        updated = dict(row)
        updated["selection_reason"] = "libri_deterministic_quality_probe"
        updated["probe_family"] = "LibriTTS_augmented_data_wav_24k"
        updated["selection_seed"] = seed
        updated["probe_rank"] = index
        output.append(updated)
    return output


def load_fsd_ground_truth(path: Path) -> dict[str, frozenset[str]]:
    labels_by_id: dict[str, frozenset[str]] = {}
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            candidates = sorted(
                name for name in archive.namelist() if name.endswith("/dev.csv")
            )
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected one FSD dev.csv in {path}, found {candidates}"
                )
            with archive.open(candidates[0]) as handle:
                text_rows = (line.decode("utf-8") for line in handle)
                for row in csv.DictReader(text_rows):
                    labels_by_id[str(row["fname"])] = frozenset(
                        str(row["labels"]).split(",")
                    )
        return labels_by_id

    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            labels_by_id[str(row["fname"])] = frozenset(
                str(row["labels"]).split(",")
            )
    return labels_by_id


def fsd_has_forbidden_label(labels: Iterable[str]) -> bool:
    label_set = set(labels)
    if label_set & FSD_FORBIDDEN_EXACT:
        return True
    return any(
        fragment in label.lower()
        for label in label_set
        for fragment in FSD_FORBIDDEN_FRAGMENTS
    )


def fsd_allow_reason(labels: Iterable[str]) -> str | None:
    label_set = set(labels)
    if fsd_has_forbidden_label(label_set):
        return None
    if label_set & {"Traffic_noise_and_roadway_noise", "Car_passing_by"}:
        return "fsd50k_roadway_noise_fail_closed"
    if label_set & FSD_ENGINE_TARGET and label_set & FSD_ROAD_CONTEXT:
        return "fsd50k_road_engine_fail_closed"
    if "Mechanical_fan" in label_set:
        return "fsd50k_mechanical_fan_fail_closed"
    return None


def fsd_proposed_route(reason: str) -> str:
    if reason in FSD_VARIABLE_VEHICLE_REASONS:
        return "variable_vehicle_event_candidate"
    if reason == "fsd50k_mechanical_fan_fail_closed":
        return "noise_only"
    raise ValueError(f"Unsupported FSD candidate reason: {reason}")


def dns_fan_allow_reason(row: Mapping[str, Any]) -> str | None:
    if str(row.get("dataset", "")) != "DNS5_noise":
        return None
    if DNS_FREESOUND_FAN_RE.search(str(row.get("source_path", ""))):
        return "dns5_freesound_validated_fan"
    return None


def pending_noise_candidate_row(
    row: Mapping[str, Any],
    *,
    reason: str,
    labels: Sequence[str],
    proposed_route: str,
) -> JsonRow:
    if proposed_route not in {"noise_only", "variable_vehicle_event_candidate"}:
        raise ValueError(f"Unsupported proposed noise route: {proposed_route}")
    return {
        "schema_version": SCHEMA_VERSION,
        "route": "audit_pending",
        "proposed_route": proposed_route,
        "audit_status": (
            "metadata_allow_acoustic_review_pending"
            if proposed_route == "noise_only"
            else "metadata_allow_domain_and_event_review_pending"
        ),
        "training_ready": False,
        "selection_reason": reason,
        "dataset": str(row.get("dataset", "")),
        "split": str(row.get("_split", "")),
        "_shard_dir": str(row.get("_shard_dir", "")),
        "shard": str(row.get("shard", "")),
        "audio_member": str(row.get("audio_member", "")),
        "key": str(row.get("key", "")),
        "source_path": str(row.get("source_path", "")),
        "joined_labels": sorted(labels),
    }


def build_indoor_noise_rows(
    dns_items: Iterable[JsonRow],
    fsd_items: Iterable[JsonRow],
    fsd_labels_by_id: Mapping[str, frozenset[str]],
) -> tuple[list[JsonRow], dict[str, Any]]:
    output = []
    dns_seen = 0
    dns_allowed = 0
    for item in dns_items:
        dns_seen += 1
        reason = dns_fan_allow_reason(item)
        if reason is None:
            continue
        dns_allowed += 1
        output.append(
            pending_noise_candidate_row(
                item,
                reason=reason,
                labels=["fan"],
                proposed_route="noise_only",
            )
        )

    fsd_seen = 0
    fsd_allowed = 0
    fsd_missing_labels = 0
    fsd_reasons: Counter[str] = Counter()
    for item in fsd_items:
        fsd_seen += 1
        source_member = str(item.get("source_member_path", ""))
        item_id = Path(source_member).stem
        labels = fsd_labels_by_id.get(item_id)
        if labels is None:
            fsd_missing_labels += 1
            continue
        reason = fsd_allow_reason(labels)
        if reason is None:
            continue
        fsd_allowed += 1
        fsd_reasons[reason] += 1
        output.append(
            pending_noise_candidate_row(
                item,
                reason=reason,
                labels=sorted(labels),
                proposed_route=fsd_proposed_route(reason),
            )
        )

    output.sort(
        key=lambda row: (
            row["dataset"],
            row["_shard_dir"],
            row["shard"],
            row["audio_member"],
        )
    )
    proposed_routes = Counter(row["proposed_route"] for row in output)
    summary = {
        "dns5_seen": dns_seen,
        "dns5_freesound_fan_allowed": dns_allowed,
        "dns5_rejected_or_unknown": dns_seen - dns_allowed,
        "fsd50k_seen": fsd_seen,
        "fsd50k_metadata_candidates": fsd_allowed,
        "fsd50k_missing_ground_truth": fsd_missing_labels,
        "fsd50k_allowed_by_reason": dict(sorted(fsd_reasons.items())),
        "fsd50k_rejected_or_unknown": fsd_seen - fsd_allowed,
        "metadata_candidate_total": len(output),
        "noise_only_candidate_total": proposed_routes["noise_only"],
        "variable_vehicle_event_candidate_total": proposed_routes[
            "variable_vehicle_event_candidate"
        ],
        "unknown_policy": "exclude",
    }
    return output, summary


def rir_allow_reason(row: Mapping[str, Any]) -> str | None:
    if str(row.get("role", "")) != "rir":
        return None
    dataset = str(row.get("dataset", ""))
    source_path = str(row.get("source_path", "")).replace("\\", "/")
    source_member = str(row.get("source_member", "")).replace("\\", "/")
    if dataset == "Arni" and ARNI_RIR_RE.search(source_path):
        return "arni_indoor_variable_acoustics_room"
    if dataset == "DNS5_RIR_SLR26" and SLR26_RIR_RE.search(source_member):
        return "slr26_simulated_indoor_room"
    if (
        dataset == "DNS5_RIR_SLR28"
        and SLR28_RIR_DIR_TOKEN in source_member
        and SLR28_RIR_NAME_RE.search(source_member)
    ):
        return "slr28_real_indoor_room_rir"
    return None


def build_indoor_rir_rows(
    items: Iterable[JsonRow],
) -> tuple[list[JsonRow], dict[str, Any]]:
    output = []
    seen_by_dataset: Counter[str] = Counter()
    allowed_by_dataset: Counter[str] = Counter()
    rejected_by_dataset: Counter[str] = Counter()
    for item in items:
        dataset = str(item.get("dataset", ""))
        seen_by_dataset[dataset] += 1
        reason = rir_allow_reason(item)
        if reason is None:
            rejected_by_dataset[dataset] += 1
            continue
        allowed_by_dataset[dataset] += 1
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "route": "audit_pending",
                "proposed_route": "rir",
                "audit_status": "metadata_allow_decode_review_pending",
                "training_ready": False,
                "selection_reason": reason,
                "dataset": dataset,
                "role": str(item.get("role", "")),
                "split": str(item.get("_split", "")),
                "_shard_dir": str(item.get("_shard_dir", "")),
                "shard": str(item.get("shard", "")),
                "audio_member": str(item.get("audio_member", "")),
                "key": str(item.get("key", "")),
                "source_path": str(item.get("source_path", "")),
                "source_member": str(item.get("source_member", "")),
            }
        )
    output.sort(
        key=lambda row: (
            row["dataset"],
            row["_shard_dir"],
            row["shard"],
            row["audio_member"],
        )
    )
    summary = {
        "seen_by_dataset": dict(sorted(seen_by_dataset.items())),
        "allowed_by_dataset": dict(sorted(allowed_by_dataset.items())),
        "rejected_by_dataset": dict(sorted(rejected_by_dataset.items())),
        "allowlist_total": len(output),
        "unknown_policy": "exclude",
    }
    return output, summary


def assert_expected_count(label: str, actual: int, expected: int | None) -> None:
    if expected is not None and actual != expected:
        raise ValueError(f"{label} count drift: expected={expected} actual={actual}")


def prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--fsd-ground-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--mls-shards", type=int, default=256)
    parser.add_argument("--mls-items-per-shard", type=int, default=16)
    parser.add_argument("--libri-probe-items", type=int, default=2048)
    parser.add_argument("--expected-dns-fan", type=int, default=989)
    parser.add_argument("--expected-fsd-allow", type=int, default=304)
    parser.add_argument("--expected-fsd-noise-only", type=int, default=41)
    parser.add_argument("--expected-fsd-variable-vehicle", type=int, default=263)
    parser.add_argument("--expected-arni-rir", type=int, default=101629)
    parser.add_argument("--expected-slr26-rir", type=int, default=60000)
    parser.add_argument("--expected-slr28-rir", type=int, default=248)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir)
    tracker = InputReceiptTracker()

    split_rows = {
        split: {
            role: load_split_rows(args.split_root, split, role, tracker)
            for role in ("clean", "noise", "rir")
        }
        for split in ("train", "valid", "test")
    }

    libri_shards = [
        row
        for split in ("train", "valid", "test")
        for row in select_source_shards(
            split_rows[split]["clean"], "LibriTTS_augmented"
        )
    ]
    libri_candidates, libri_summary = build_libri_candidate_rows(
        iter_selected_items(libri_shards, tracker)
    )

    mls_train_shards = select_source_shards(
        split_rows["train"]["clean"], "MLS_HQ_en"
    )
    selected_mls_shards = select_mls_probe_shards(
        mls_train_shards, args.mls_shards, args.seed
    )
    mls_probe = build_mls_probe_rows(
        selected_mls_shards,
        iter_selected_items(selected_mls_shards, tracker),
        args.mls_items_per_shard,
        args.seed,
    )
    libri_probe = build_libri_probe_rows(
        libri_candidates, args.libri_probe_items, args.seed
    )

    dns_shards = select_source_shards(
        split_rows["train"]["noise"], "DNS5_noise"
    )
    fsd_shards = select_source_shards(
        split_rows["train"]["noise"], "FSD50K"
    )
    tracker.add_file(args.fsd_ground_truth)
    fsd_labels = load_fsd_ground_truth(args.fsd_ground_truth)
    noise_asset_candidates, noise_summary = build_indoor_noise_rows(
        iter_selected_items(dns_shards, tracker),
        iter_selected_items(fsd_shards, tracker),
        fsd_labels,
    )
    indoor_noise = [
        row
        for row in noise_asset_candidates
        if row["proposed_route"] == "noise_only"
    ]
    variable_vehicle_events = [
        row
        for row in noise_asset_candidates
        if row["proposed_route"] == "variable_vehicle_event_candidate"
    ]

    # RIR tar shards may contain both SLR26 and SLR28 items.  Select all
    # train-RIR shards once, then fail closed at item level using dataset,
    # role, source-member path, and filename.
    rir_shards = list(split_rows["train"]["rir"])
    indoor_rir, rir_summary = build_indoor_rir_rows(
        iter_selected_items(rir_shards, tracker)
    )

    assert_expected_count(
        "DNS5 Freesound fan",
        noise_summary["dns5_freesound_fan_allowed"],
        args.expected_dns_fan,
    )
    assert_expected_count(
        "FSD50K metadata candidates",
        noise_summary["fsd50k_metadata_candidates"],
        args.expected_fsd_allow,
    )
    assert_expected_count(
        "FSD50K proposed noise-only",
        sum(row["dataset"] == "FSD50K" for row in indoor_noise),
        args.expected_fsd_noise_only,
    )
    assert_expected_count(
        "FSD50K variable vehicle event candidates",
        len(variable_vehicle_events),
        args.expected_fsd_variable_vehicle,
    )
    assert_expected_count(
        "Arni RIR",
        rir_summary["allowed_by_dataset"].get("Arni", 0),
        args.expected_arni_rir,
    )
    assert_expected_count(
        "SLR26 RIR",
        rir_summary["allowed_by_dataset"].get("DNS5_RIR_SLR26", 0),
        args.expected_slr26_rir,
    )
    assert_expected_count(
        "SLR28 RIR",
        rir_summary["allowed_by_dataset"].get("DNS5_RIR_SLR28", 0),
        args.expected_slr28_rir,
    )

    output_paths = {
        "libri_pending_candidates": args.output_dir
        / "libri_train_only_content_groups.jsonl",
        "mls_probe": args.output_dir / "mls_clean_candidate_probe.jsonl",
        "libri_probe": args.output_dir / "libri_clean_candidate_probe.jsonl",
        "indoor_noise": args.output_dir
        / "indoor_noise_candidates_pending.jsonl",
        "variable_vehicle_events": args.output_dir
        / "variable_vehicle_event_candidates.jsonl",
        "indoor_rir": args.output_dir
        / "indoor_rir_candidates_pending.jsonl",
    }
    output_counts = {
        "libri_pending_candidates": write_jsonl(
            output_paths["libri_pending_candidates"], libri_candidates
        ),
        "mls_probe": write_jsonl(output_paths["mls_probe"], mls_probe),
        "libri_probe": write_jsonl(output_paths["libri_probe"], libri_probe),
        "indoor_noise": write_jsonl(output_paths["indoor_noise"], indoor_noise),
        "variable_vehicle_events": write_jsonl(
            output_paths["variable_vehicle_events"],
            variable_vehicle_events,
        ),
        "indoor_rir": write_jsonl(output_paths["indoor_rir"], indoor_rir),
    }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "audit_status": "complete_but_clean_candidates_pending_scores",
        "training_ready": False,
        "seed": args.seed,
        "split_root": str(args.split_root),
        "probe_contract": {
            "mls_shards": args.mls_shards,
            "mls_items_per_shard": args.mls_items_per_shard,
            "mls_total": len(mls_probe),
            "libri_total": len(libri_probe),
        },
        "libri": libri_summary,
        "noise": noise_summary,
        "rir": rir_summary,
        "output_counts": output_counts,
        "input_receipts": tracker.sorted_receipts(),
        "promotion_gate": (
            "clean_candidate rows remain audit_pending_scores and must not be "
            "consumed by training until score calibration and listening review pass"
        ),
    }
    summary_path = args.output_dir / "audit_summary.json"
    write_json(summary_path, summary)

    receipt_targets = [*output_paths.values(), summary_path]
    receipts = {
        "schema_version": SCHEMA_VERSION,
        "inputs": tracker.sorted_receipts(),
        "outputs": [
            {
                **sha256_file(path),
                "path": path.name,
            }
            for path in sorted(receipt_targets)
        ],
    }
    write_json(args.output_dir / "sha256_receipts.json", receipts)
    print(json.dumps(summary["output_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
