import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_dnf_phase_a_blind_listening_pack.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_dnf_phase_a_blind_listening_pack",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def prepare_arm(root: Path, mode: str) -> None:
    rows = []
    outputs = ("standard",) if mode == "standard" else ("eq14", "speech_head")
    for view in ("single_noise_s_plus_n1", "identity_clean_s"):
        uid = f"uid-{view}"
        rows.append(
            {
                "sample_uid": uid,
                "evaluation_input_view": view,
                "route": "clean_weak",
                "noise_family": "fan",
                "target_snr_db": 20.0,
                "outputs": list(outputs),
            }
        )
        for output in ("input", "clean", *outputs):
            path = root / "listening" / view / output / f"{uid}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(output.encode("utf-8"))
    manifest = root / "listening" / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_blind_pack_is_deterministic_and_contains_all_three_outputs(tmp_path):
    standard = tmp_path / "standard"
    dnf = tmp_path / "dnf"
    prepare_arm(standard, "standard")
    prepare_arm(dnf, "dnf")
    blind, private = MODULE.build_pack(
        standard,
        dnf,
        tmp_path / "pack",
        seed=1234,
    )
    assert len(blind) == len(private) == 2
    assert all(set(row["candidate_mapping"].values()) == set(MODULE.OUTPUTS) for row in private)
    assert all(
        Path(row[f"candidate_{label}"]).is_file()
        for row in blind
        for label in ("A", "B", "C")
    )
