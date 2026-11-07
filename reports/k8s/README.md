# Kubernetes test results

`scripts/k8s/load-test.sh` and `scripts/k8s/recovery-test.sh` each write a timestamped folder
here (raw k6 log, HPA/CPU/memory timeline, summary). Commit the folders you quote in
[`docs/results.md`](../../docs/results.md) so every number has its raw evidence next to it.
`sample.png` is git-ignored because it may be derived from MVTec AD (non-commercial licence).
