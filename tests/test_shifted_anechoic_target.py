import unittest

import numpy as np

from dataloaders.legacy_online_degradation import (
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


if __name__ == "__main__":
    unittest.main()
