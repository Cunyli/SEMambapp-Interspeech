import json
from pathlib import Path


# 1. train_speech.json: 43873
# root = Path("/m/teamwork/t412_symptosonic/databases/VCTK/wav48_silence_trimmed/mic2")
# exclude = {"p280", "p315"}

# files = sorted(
#     str(p.resolve())
#     for p in root.rglob("*.flac")
#     if len(p.parts) >= 2 and p.parts[-2] not in exclude
# )

# Path("data/train_speech.json").write_text(json.dumps(files, indent=2))
# print("train_speech.json:", len(files))


# 2. train_noise.json: 65388
# roots = [
#     Path("/m/teamwork/t412_symptosonic/databases/DNS/noise"),
#     Path("/m/teamwork/t412_symptosonic/databases/WHAM/extracted/high_res_wham/audio"),
# ]
# exts = {".wav", ".flac"}
# files = sorted(
#     str(p.resolve())
#     for root in roots
#     for p in root.rglob("*")
#     if p.is_file() and p.suffix.lower() in exts
# )
# Path("data/train_noise.json").write_text(json.dumps(files, indent=2))
# print("train_noise.json:", len(files))



# 3. train_rir.json: 172285
# roots = [
#     Path("/m/teamwork/t412_symptosonic/databases/Arni/extracted"),
#     Path("/m/teamwork/t412_symptosonic/databases/DNS/room-impulse"),
# ]
# exts = {".wav", ".flac"}

# files = sorted(
#     str(p.resolve())
#     for root in roots
#     for p in root.rglob("*")
#     if p.is_file() and p.suffix.lower() in exts
# )

# Path("data/train_rir.json").write_text(json.dumps(files, indent=2))
# print("train_rir.json:", len(files))

import json
for name in ["train_speech", "train_noise", "train_rir", "val_clean", "val_degraded"]:
    with open(f"data/{name}.json") as f:
        items = json.load(f)
    print(name, len(items))