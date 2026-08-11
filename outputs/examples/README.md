# Selected listening examples

This folder is intentionally small. Select at most 3--5 sample IDs that expose
both strengths and failures, rather than publishing an entire evaluation set.

For each ID, include only the files needed for one fixed comparison, normally
`degraded`, `clean reference`, and one or two model outputs. The manifest must
record:

- sample ID, CS/SV task, health or pathology severity, and SNR/condition;
- model and checkpoint SHA-256;
- audio SHA-256 and sample rate;
- source dataset and redistribution status;
- why the case was selected.

Pathological recordings must not be added to a public branch until their
sharing terms and participant-data boundary are confirmed. A private listening
bundle may use the same manifest after authorization.
