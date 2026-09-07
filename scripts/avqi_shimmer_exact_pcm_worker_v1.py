#!/usr/bin/env python3
"""Expose the frozen Exact worker's actual PCM without changing its calculations."""
import hashlib
from pathlib import Path
import runpy

BASE_WORKER = Path("/scratch/work/lil14/SEMambapp-Interspeech-avqi-shimmer-db-candidate-e-v27/scripts/avqi_shimmer_exact_topology_worker.py")
BASE_SHA256 = "c78cdb277274a9f46153c80ca5ad8c47536e3c1009cf1b3c2b613aee744d276f"


def export_engine_type(base, pcm16):
    class PCMExportEngine(base):
        def metric_highpass(self, waveform, highpass_mode):
            result = super().metric_highpass(waveform, highpass_mode)
            self.current_pcm_codes = pcm16(result[0]).astype("<i4")
            return result

        def refresh_waveform(self, *args, **kwargs):
            self.current_pcm_codes = None
            topology = super().refresh_waveform(*args, **kwargs)
            codes = self.current_pcm_codes
            if codes is None or codes.size != topology["source_sample_count"]:
                raise ValueError("Exact PCM export must come from this refresh")
            digest = hashlib.sha256(codes.tobytes()).hexdigest()
            if digest != topology["highpass_pcm16_sha256"]:
                raise ValueError("Exact PCM export differs from scoring waveform")
            topology["highpass_pcm16_codes"] = codes.tolist()
            topology["highpass_pcm_transport_schema"] = "exact-worker-current-pcm16-v1"
            return topology
    return PCMExportEngine


def main():
    if hashlib.sha256(BASE_WORKER.read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError("frozen base Exact worker hash differs")
    namespace = runpy.run_path(str(BASE_WORKER))
    namespace["main"].__globals__["ExactTopologyEngine"] = export_engine_type(
        namespace["ExactTopologyEngine"], namespace["pcm16"])
    namespace["main"]()


if __name__ == "__main__":
    main()
