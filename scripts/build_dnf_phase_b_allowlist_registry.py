"""Build a versioned, fail-closed Phase-B indoor allowlist registry.

The registry distinguishes metadata-selected review candidates from approved
training assets.  It deliberately writes empty approved allowlists until
manual noise review and RIR parameter approval are represented by a new,
versioned input contract.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the immutable Phase-B indoor allowlist registry."
    )
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_candidate_rows(
    path: Path,
    *,
    required_route: str,
    required_proposed_route: str,
    allowed_datasets: set[str] | None = None,
    allowed_selection_reasons: set[str] | None = None,
) -> tuple[set[str], Counter[str], Counter[str]]:
    keys: set[str] = set()
    datasets: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for row in iter_jsonl(path):
        key = str(row["key"])
        if key in keys:
            raise ValueError(f"duplicate candidate key in {path}: {key}")
        if row.get("route") != required_route:
            raise ValueError(f"{path} contains non-pending route for {key}")
        if row.get("proposed_route") != required_proposed_route:
            raise ValueError(f"{path} contains wrong proposed route for {key}")
        if row.get("training_ready") is not False:
            raise ValueError(f"{path} contains training-ready pending row: {key}")
        dataset = str(row["dataset"])
        reason = str(row["selection_reason"])
        if allowed_datasets is not None and dataset not in allowed_datasets:
            raise ValueError(f"{path} contains unapproved dataset {dataset}")
        if (
            allowed_selection_reasons is not None
            and reason not in allowed_selection_reasons
        ):
            raise ValueError(f"{path} contains unapproved reason {reason}")
        keys.add(key)
        datasets[dataset] += 1
        reasons[reason] += 1
    if not keys:
        raise ValueError(f"candidate file is empty: {path}")
    return keys, datasets, reasons


def validate_review_rows(
    path: Path,
    *,
    candidate_keys: set[str],
) -> dict:
    reviewed: set[str] = set()
    automatic_pass = 0
    for row in iter_jsonl(path):
        key = str(row["key"])
        if key in reviewed:
            raise ValueError(f"duplicate review key in {path}: {key}")
        if key not in candidate_keys:
            raise ValueError(f"review key is not in candidate registry: {key}")
        if row.get("route") != "audit_pending":
            raise ValueError(f"review row is not fail-closed: {key}")
        if row.get("training_ready") is not False:
            raise ValueError(f"review row is unexpectedly training-ready: {key}")
        automatic_gate = row.get("automatic_gate")
        if not isinstance(automatic_gate, dict):
            raise ValueError(f"review row lacks automatic gate: {key}")
        automatic_pass += int(bool(automatic_gate.get("automatic_pass")))
        reviewed.add(key)
    if not reviewed:
        raise ValueError(f"review file is empty: {path}")
    return {
        "reviewed_count": len(reviewed),
        "automatic_pass_count": automatic_pass,
        "manual_or_parameter_approval_count": 0,
        "training_ready_count": 0,
    }


def artifact_receipt(path: Path, *, base: Path) -> dict:
    return {
        "path": str(path.relative_to(base)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "line_count": sum(
            1
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ),
    }


def validate_config(config: dict) -> None:
    if config.get("schema_version") != "dnf_phase_b_indoor_allowlist_registry_v1":
        raise ValueError("unexpected allowlist registry schema")
    if config.get("status") != "review_allowlist_only_not_training_ready":
        raise ValueError("allowlist registry must be review-only")
    if config.get("formal_submit_allowed") is not False:
        raise ValueError("formal submission must remain disabled")
    approved = config["approved_training_allowlists"]
    if approved.get("initial_state") != "empty_fail_closed":
        raise ValueError("approved allowlists must start empty")
    if approved.get("training_ready") is not False:
        raise ValueError("approved allowlists cannot be training-ready")
    if approved.get("promotion_requires_new_version") is not True:
        raise ValueError("promotion must require an immutable version bump")


def build_registry(
    *,
    audit_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict:
    audit_dir = audit_dir.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite registry: {output_dir}")
    config = read_json(config_path)
    validate_config(config)

    noise_contract = config["noise_review_allowlist"]
    vehicle_contract = config["variable_vehicle_exclusion"]
    rir_contract = config["rir_review_allowlist"]
    noise_path = audit_dir / noise_contract["candidate_file"]
    vehicle_path = audit_dir / vehicle_contract["candidate_file"]
    rir_path = audit_dir / rir_contract["candidate_file"]
    noise_keys, noise_datasets, noise_reasons = validate_candidate_rows(
        noise_path,
        required_route=noise_contract["required_route"],
        required_proposed_route=noise_contract["required_proposed_route"],
        allowed_datasets=set(noise_contract["allowed_datasets"]),
        allowed_selection_reasons=set(
            noise_contract["allowed_selection_reasons"]
        ),
    )
    vehicle_keys, vehicle_datasets, vehicle_reasons = validate_candidate_rows(
        vehicle_path,
        required_route=vehicle_contract["required_route"],
        required_proposed_route=vehicle_contract["required_proposed_route"],
    )
    if noise_keys & vehicle_keys:
        raise ValueError("stable-noise and variable-vehicle registries overlap")
    rir_keys, rir_datasets, rir_reasons = validate_candidate_rows(
        rir_path,
        required_route=rir_contract["required_route"],
        required_proposed_route=rir_contract["required_proposed_route"],
        allowed_datasets=set(rir_contract["allowed_datasets"]),
    )

    review_contract = config["review_evidence"]
    noise_review_path = audit_dir / review_contract["noise_acoustic_review"]
    rir_review_path = audit_dir / review_contract["rir_parameter_review"]
    noise_review = validate_review_rows(
        noise_review_path,
        candidate_keys=noise_keys,
    )
    rir_review = validate_review_rows(
        rir_review_path,
        candidate_keys=rir_keys,
    )

    approved_contract = config["approved_training_allowlists"]
    registry = {
        "schema_version": config["schema_version"],
        "allowlist_id": config["allowlist_id"],
        "status": config["status"],
        "training_ready": False,
        "formal_submit_allowed": False,
        "config_sha256": sha256_file(config_path),
        "source_audit": {
            "run_dir": str(audit_dir),
            "audit_summary_sha256": sha256_file(
                audit_dir / "audit_summary.json"
            ),
            "artifact_index_sha256": sha256_file(
                audit_dir / "artifact_sha256.txt"
            ),
        },
        "review_allowlists": {
            "indoor_stable_noise": {
                "candidate_count": len(noise_keys),
                "datasets": dict(sorted(noise_datasets.items())),
                "selection_reasons": dict(sorted(noise_reasons.items())),
                "artifact": artifact_receipt(noise_path, base=audit_dir),
                "review_evidence": {
                    **noise_review,
                    "artifact": artifact_receipt(
                        noise_review_path,
                        base=audit_dir,
                    ),
                },
                "manual_approval_required": True,
            },
            "indoor_rir": {
                "candidate_count": len(rir_keys),
                "datasets": dict(sorted(rir_datasets.items())),
                "selection_reasons": dict(sorted(rir_reasons.items())),
                "artifact": artifact_receipt(rir_path, base=audit_dir),
                "review_evidence": {
                    **rir_review,
                    "artifact": artifact_receipt(
                        rir_review_path,
                        base=audit_dir,
                    ),
                },
                "operator_scope": rir_contract["operator_scope"],
                "parameter_approval_required": True,
            },
        },
        "explicit_exclusions": {
            "variable_vehicle_events": {
                "candidate_count": len(vehicle_keys),
                "datasets": dict(sorted(vehicle_datasets.items())),
                "selection_reasons": dict(sorted(vehicle_reasons.items())),
                "artifact": artifact_receipt(vehicle_path, base=audit_dir),
                "disjoint_from_indoor_stable_noise": True,
                "prohibited_training_routes": vehicle_contract[
                    "prohibited_training_routes"
                ],
            }
        },
        "approved_training_allowlists": {
            "noise_file": approved_contract["noise_file"],
            "rir_file": approved_contract["rir_file"],
            "noise_count": 0,
            "rir_count": 0,
            "training_ready": False,
            "promotion_requires_new_version": True,
        },
        "future_consumer_must_require": config[
            "future_consumer_must_require"
        ],
        "claim_limit": (
            "The versioned review allowlists define the only indoor assets "
            "eligible for further review. They are not training allowlists. "
            "Approved noise and RIR lists are intentionally empty until a "
            "new immutable version records manual/parameter approval."
        ),
    }

    output_dir.mkdir(parents=True)
    approved_noise = output_dir / approved_contract["noise_file"]
    approved_rir = output_dir / approved_contract["rir_file"]
    approved_noise.write_text("", encoding="utf-8")
    approved_rir.write_text("", encoding="utf-8")
    registry["approved_training_allowlists"]["noise_sha256"] = sha256_file(
        approved_noise
    )
    registry["approved_training_allowlists"]["rir_sha256"] = sha256_file(
        approved_rir
    )
    registry_path = output_dir / "allowlist_registry.json"
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config_snapshot = output_dir / "allowlist_config_snapshot.json"
    config_snapshot.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipts = {
        path.name: sha256_file(path)
        for path in (
            approved_noise,
            approved_rir,
            registry_path,
            config_snapshot,
        )
    }
    (output_dir / "artifact_sha256.txt").write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(receipts.items())
        ),
        encoding="utf-8",
    )
    return registry


def main() -> None:
    args = parse_args()
    registry = build_registry(
        audit_dir=args.audit_dir,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(json.dumps(registry, sort_keys=True))


if __name__ == "__main__":
    main()
