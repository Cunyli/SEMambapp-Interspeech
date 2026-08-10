import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from dataloaders.legacy_online_degradation import (
    LegacyOnlineDegradationDataset,
    _rir_direct_path_delay,
    _shift_clean_by_delay,
    _target_audio_for_selected_degradations,
)


class ShiftedAnechoicTargetTest(unittest.TestCase):
    def test_rir_delay_uses_argmax_abs(self):
        rir = np.zeros((1, 8), dtype=np.float32)
        rir[0, 2] = -0.4
        rir[0, 5] = 0.8

        self.assertEqual(_rir_direct_path_delay(rir), 5)

    def test_shifted_target_pads_front_by_rir_delay(self):
        clean = np.arange(1, 9, dtype=np.float32).reshape(1, -1)
        rir = np.zeros((1, 8), dtype=np.float32)
        rir[0, 5] = 1.0

        target = _target_audio_for_selected_degradations(
            clean,
            rir,
            ["reverb", "noise"],
            target_type="shifted_anechoic",
            rir_delay_mode="argmax_abs",
        )

        expected = np.array([[0, 0, 0, 0, 0, 1, 2, 3]], dtype=np.float32)
        np.testing.assert_array_equal(target, expected)

    def test_no_reverb_keeps_clean_target(self):
        clean = np.arange(1, 9, dtype=np.float32).reshape(1, -1)
        rir = np.zeros((1, 8), dtype=np.float32)
        rir[0, 5] = 1.0

        target = _target_audio_for_selected_degradations(
            clean,
            rir,
            ["noise"],
            target_type="shifted_anechoic",
            rir_delay_mode="argmax_abs",
        )

        np.testing.assert_array_equal(target, clean)

    def test_legacy_target_keeps_clean_even_with_reverb(self):
        clean = np.arange(1, 9, dtype=np.float32).reshape(1, -1)
        rir = np.zeros((1, 8), dtype=np.float32)
        rir[0, 5] = 1.0

        target = _target_audio_for_selected_degradations(
            clean,
            rir,
            ["reverb"],
            target_type="legacy_clean",
            rir_delay_mode="argmax_abs",
        )

        np.testing.assert_array_equal(target, clean)

    def test_full_rir_convolution_reference_differs_from_shifted_target(self):
        clean = np.zeros((1, 10), dtype=np.float32)
        clean[0, 0] = 1.0
        rir = np.zeros((1, 8), dtype=np.float32)
        rir[0, 2] = 1.0
        rir[0, 5] = 0.5

        degraded = np.convolve(clean.reshape(-1), rir.reshape(-1), mode="full")[: clean.shape[1]]
        target = _shift_clean_by_delay(clean, 2).reshape(-1)

        np.testing.assert_array_equal(degraded[:6], np.array([0, 0, 1, 0, 0, 0.5], dtype=np.float32))
        np.testing.assert_array_equal(target[:6], np.array([0, 0, 1, 0, 0, 0], dtype=np.float32))

    def test_legacy_training_reads_rir_from_tar_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean = root / "clean.wav"
            noise = root / "noise.wav"
            rir = root / "rir.wav"
            sf.write(clean, np.full(3200, 0.1, dtype=np.float32), 16000)
            sf.write(noise, np.full(3200, 0.01, dtype=np.float32), 16000)
            sf.write(rir, np.r_[np.zeros(8, dtype=np.float32), np.ones(1, dtype=np.float32)], 16000)

            clean_json = root / "clean.json"
            noise_json = root / "noise.json"
            clean_valid_json = root / "clean_valid.json"
            degraded_valid_json = root / "degraded_valid.json"
            clean_json.write_text(json.dumps([str(clean)]), encoding="utf-8")
            noise_json.write_text(json.dumps([str(noise)]), encoding="utf-8")
            clean_valid_json.write_text(json.dumps([str(clean)]), encoding="utf-8")
            degraded_valid_json.write_text(json.dumps([str(clean)]), encoding="utf-8")

            shard = root / "rir-000000.tar"
            with tarfile.open(shard, "w") as tar:
                tar.add(rir, arcname="rir_000.wav")
            rir_manifest = root / "manifest.jsonl"
            rir_manifest.write_text(
                json.dumps(
                    {
                        "key": "rir_000",
                        "shard": shard.name,
                        "audio_member": "rir_000.wav",
                        "source_path": str(rir),
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            use_sim_root = root / "USE_simulation"
            use_sim_root.mkdir()
            (use_sim_root / "simulate_degradation.py").write_text(
                "def random_select_and_order(cfg, seed=None):\n"
                "    return {}, ['reverb']\n"
                "\n"
                "def apply_degradation(cfg, clean, noise, rir, degrad_cfgs, selected_degrads, seed=None):\n"
                "    return clean, clean\n",
                encoding="utf-8",
            )
            cfg = {
                "stft_cfg": {"sampling_rate": 16000, "n_fft": 64, "hop_size": 16, "win_size": 64},
                "training_cfg": {"segment_size": 1600, "legacy_validation_limit": 1},
                "model_cfg": {"compress_factor": 0.3},
                "target_cfg": {"type": "legacy_clean", "rir_delay_mode": "argmax_abs"},
            }
            dataset = LegacyOnlineDegradationDataset(
                cfg,
                clean_json=str(clean_json),
                noise_json=str(noise_json),
                rir_json=str(rir_manifest),
                clean_valid_json=str(clean_valid_json),
                degraded_valid_json=str(degraded_valid_json),
                use_simulation_root=str(use_sim_root),
                mode="Train",
                seed=1234,
            )

            item = dataset[0]

            self.assertEqual(len(dataset.rir_wavs_path), 1)
            self.assertEqual(item[0].shape[-1], 1600)

    def test_legacy_training_reads_all_roles_from_shard_level_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_valid = root / "clean_valid.wav"
            degraded_valid = root / "degraded_valid.wav"
            sf.write(clean_valid, np.full(3200, 0.1, dtype=np.float32), 16000)
            sf.write(degraded_valid, np.full(3200, 0.1, dtype=np.float32), 16000)

            clean_valid_json = root / "clean_valid.json"
            degraded_valid_json = root / "degraded_valid.json"
            clean_valid_json.write_text(json.dumps([str(clean_valid)]), encoding="utf-8")
            degraded_valid_json.write_text(json.dumps([str(degraded_valid)]), encoding="utf-8")

            def write_shard_manifest(role, values):
                source = root / f"{role}.wav"
                sf.write(source, np.asarray(values, dtype=np.float32), 16000)
                shard_dir = root / "shards" / role
                shard_dir.mkdir(parents=True)
                shard = shard_dir / f"{role}-000000.tar"
                with tarfile.open(shard, "w") as tar:
                    tar.add(source, arcname=f"{role}_000.wav")
                manifest = root / f"{role}_shards.jsonl"
                manifest.write_text(
                    json.dumps(
                        {
                            "_shard_dir": str(shard_dir),
                            "role": role,
                            "sample_count": 1,
                            "shard": shard.name,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return manifest

            clean_manifest = write_shard_manifest("clean", np.full(3200, 0.1, dtype=np.float32))
            noise_manifest = write_shard_manifest("noise", np.full(3200, 0.01, dtype=np.float32))
            rir_manifest = write_shard_manifest(
                "rir",
                np.r_[np.zeros(8, dtype=np.float32), np.ones(1, dtype=np.float32)],
            )

            use_sim_root = root / "USE_simulation"
            use_sim_root.mkdir()
            (use_sim_root / "simulate_degradation.py").write_text(
                "def random_select_and_order(cfg, seed=None):\n"
                "    return {}, ['reverb']\n"
                "\n"
                "def apply_degradation(cfg, clean, noise, rir, degrad_cfgs, selected_degrads, seed=None):\n"
                "    return clean, clean\n",
                encoding="utf-8",
            )
            cfg = {
                "stft_cfg": {"sampling_rate": 16000, "n_fft": 64, "hop_size": 16, "win_size": 64},
                "training_cfg": {"segment_size": 1600, "legacy_validation_limit": 1},
                "model_cfg": {"compress_factor": 0.3},
                "target_cfg": {"type": "legacy_clean", "rir_delay_mode": "argmax_abs"},
            }
            dataset = LegacyOnlineDegradationDataset(
                cfg,
                clean_json=str(clean_manifest),
                noise_json=str(noise_manifest),
                rir_json=str(rir_manifest),
                clean_valid_json=str(clean_valid_json),
                degraded_valid_json=str(degraded_valid_json),
                use_simulation_root=str(use_sim_root),
                mode="Train",
                seed=1234,
            )

            item = dataset[0]

            self.assertEqual(len(dataset.clean_wavs_path), 1)
            self.assertEqual(len(dataset.noise_wavs_path), 1)
            self.assertEqual(len(dataset.rir_wavs_path), 1)
            self.assertEqual(dataset.sample_id(0), "clean_000")
            self.assertEqual(dataset.clean_wavs_path[0]["audio_member"], "clean_000.wav")
            self.assertEqual(item[0].shape[-1], 1600)

if __name__ == "__main__":
    unittest.main()
