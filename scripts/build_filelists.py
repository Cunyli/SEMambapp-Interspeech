import json
from pathlib import Path


DATA_DIR = Path("data")
SIM_ROOT = Path("/scratch/elec/t412-speechcom/symptonic-r2b/simulation-r2b")

VCTK_ROOT = SIM_ROOT / "VCTK" / "wav48_silence_trimmed" / "mic2"
DNS_NOISE_ROOT = SIM_ROOT / "DNS_noise"
WHAM_ROOT = SIM_ROOT / "WHAM" / "extracted" / "high_res_wham" / "audio"
ARNI_ROOT = SIM_ROOT / "Arni" / "extracted"
DNS_RIR_ROOT = SIM_ROOT / "DNS" / "room-impulse"

EXTS = {".wav", ".flac"}
VCTK_EXCLUDE = {"p280", "p315"}


def collect_files(roots, exts):
    files = []
    for root in roots:
        if not root.exists():
            print(f"Warning: missing root {root}")
            continue
        files.extend(
            str(path.resolve())
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in exts
        )
    return sorted(files)


def build_train_speech():
    if not VCTK_ROOT.exists():
        raise FileNotFoundError(f"Missing VCTK root: {VCTK_ROOT}")

    files = sorted(
        str(path.resolve())
        for path in VCTK_ROOT.rglob("*.flac")
        if len(path.parts) >= 2 and path.parts[-2] not in VCTK_EXCLUDE
    )
    (DATA_DIR / "train_speech.json").write_text(json.dumps(files, indent=2))
    print("train_speech.json:", len(files))


def build_train_noise():
    files = collect_files([DNS_NOISE_ROOT, WHAM_ROOT], EXTS)
    (DATA_DIR / "train_noise.json").write_text(json.dumps(files, indent=2))
    print("train_noise.json:", len(files))


def build_train_rir():
    files = collect_files([ARNI_ROOT, DNS_RIR_ROOT], EXTS)
    (DATA_DIR / "train_rir.json").write_text(json.dumps(files, indent=2))
    print("train_rir.json:", len(files))


def print_counts():
    for name in ["train_speech", "train_noise", "train_rir", "val_clean", "val_degraded"]:
        path = DATA_DIR / f"{name}.json"
        if not path.exists():
            print(f"{name} missing")
            continue
        with path.open() as f:
            items = json.load(f)
        print(name, len(items))


if __name__ == "__main__":
    build_train_speech()
    build_train_noise()
    build_train_rir()
    print_counts()
