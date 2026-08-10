import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_semambapp_dnf_phase_a.py"
CONTRACT = ROOT / "configs" / "train" / "dnf_phase_ab_v2_contract.json"
WRAPPER = ROOT / "scripts" / "cluster" / "slurm_semambapp_dnf_phase_a_array.sh"


def test_training_script_is_valid_python_and_has_no_resume_or_gan_cli() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert tree is not None
    assert "--resume" not in source
    assert "--checkpoint-init" not in source
    assert "--gan" not in source
    assert "validate_manifest_noise_pairing" in source
    assert "validate_manifest_speech_partition" in source
    assert '"paper_mechanism_gate": paper_mechanism_gate' in source


def test_contract_freezes_two_modes_routes_loss_and_checkpoints() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    schedule = contract["data"]["route_schedule_20"]
    routes = [row["route"] for row in schedule]
    variants = [row.get("variant") for row in schedule]

    assert contract["allowed_modes"] == ["standard", "dnf"]
    assert len(schedule) == 20
    assert routes.count("noisy_eq13") == 15
    assert variants.count("regular") == 4
    assert variants.count("weak") == 1
    assert next(row for row in schedule if row.get("variant") == "weak")["snr_db"] == 20.0
    assert contract["loss"]["reduction"] == "sum_over_total_microbatch"
    variants = contract["loss"]["active_log_rms"]["variants"]
    assert variants["paper_exact"]["weight"] == 0.0
    assert variants["matched_scale"]["weight"] == 1.0
    assert (
        contract["loss"]["active_log_rms"][
            "primary_dnf_efficacy_variant"
        ]
        == "paper_exact"
    )
    assert contract["loss"]["active_log_rms"]["routes"] == [
        "clean_eq15_final_output",
        "noisy_eq13_speech_output",
    ]
    assert contract["data"]["noise_pairing_policy"] == "same_family_iid"
    assert contract["data"]["speech_partition_policy"] == "disjoint_item_pools"
    assert contract["data"]["training_input"] == "x=s+n1+n2"
    assert contract["data"]["deployment_validation_input"] == "y=s+n1"
    assert contract["data"]["snr_definition"] == "10log10(E_s/E_n1)"
    assert contract["training"]["checkpoint_steps"] == [250, 500, 1000, 2000]
    assert contract["training"]["batch_size"] == 4
    assert contract["training"]["gradient_accumulation_steps"] == 5
    assert contract["evaluation_gates"]["eq14_mean_sdri_db_must_be_nonnegative"]
    assert (
        contract["evaluation_gates"]["eq14_active_gain_median_abs_max_db"]
        == 1.0
    )


def test_contract_uses_only_parameterized_indoor_noise() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["data"]["noise_sources"] == [
        "parameterized_indoor_hvac",
        "parameterized_indoor_fan",
        "parameterized_indoor_vehicle_cabin",
    ]
    assert contract["forbidden_features"]["clean_candidate"]
    assert contract["forbidden_features"]["tau_training"]


def test_wrapper_is_single_gpu_pair_and_requires_frozen_manifests() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert 'ARRAY_SPEC="${ARRAY_SPEC:-0-1%2}"' in source
    assert '"0-1%1"|"0-1%2"' in source
    assert 'GPUS="${GPUS:-1}"' in source
    assert 'GPU_TYPE="${GPU_TYPE:-v100}"' in source
    assert 'MAX_STEPS="${MAX_STEPS:-2000}"' in source
    assert 'BATCH_SIZE="${BATCH_SIZE:-4}"' in source
    assert 'GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-5}"' in source
    assert 'CUT_DURATION="${CUT_DURATION:-1.0}"' in source
    assert 'LOSS_VARIANT="${LOSS_VARIANT:-paper_exact}"' in source
    assert '--loss-variant "$LOSS_VARIANT"' in source
    assert 'TRAIN_MANIFEST="${TRAIN_MANIFEST:-}"' in source
    assert 'VALID_MANIFEST="${VALID_MANIFEST:-}"' in source


def test_contract_json_has_no_duplicate_keys() -> None:
    def reject_duplicates(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate key: {key}")
            output[key] = value
        return output

    json.loads(CONTRACT.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
