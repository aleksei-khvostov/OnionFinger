# OnionFinger Transformer verification

OnionFinger measures how offline changes to Tor traffic direction arrays affect a fixed website classifier. This release records a CPU verification of one preserved Transformer checkpoint on 4,000 reconstructed test rows from Rimmer et al.'s historical closed-world archive of clearnet visits through Tor.

## Verified results

| Evaluation condition | Correct / total | Top-1 accuracy |
|---|---:|---:|
| Clean input | 2,813 / 4,000 | 70.325% |
| Prepend 200 zeros, then crop | 491 / 4,000 | 12.275% |
| Prepend 200 outgoing symbols, then crop | 287 / 4,000 | 7.175% |
| Direction rewrite, window 12 | 2,813 / 4,000 | 70.325% |

The legacy `timing_jitter` operation returned elementwise-identical copies for every test row. The verification reused the clean metrics for that identity check. All four measured counts match the archived Transformer records. The historical records lack run-level checkpoint and split hashes, so the original run lineage remains unresolved. [Aggregate CPU result](transformer_cpu.json) · [CNN archive status](archive.md)

## Protocol and limitations

The Rimmer archive has 249,983 direction arrays, 100 classes, and 5,000 stored positions per row. The checkpoint's effective settings reconstruct a 20,000-row pool with 200 rows per class. A NumPy `RandomState(42)` selects the pool and the stratified 20% outer test split. A Torch generator seeded at 42 reserves 10% of the training pool for validation. This yields 14,400 optimization-training rows, 1,600 validation rows, and 4,000 test rows, with 40 test rows per class. No source row ID is shared across the three partitions. The release contains only aggregate counts and digests of ordered test membership and numeric class IDs.

Directions are mapped to −1, 0 or +1. Each perturbation is applied before cropping to 5,000 positions and sampling indices 0, 10, …, 4,990. The Transformer uses 500 tokens, model width 128, four attention heads, and three encoder layers. Its stored config gives a 30-epoch budget and dropout 0.1. The selected checkpoint is epoch 29 and records 71.25% validation accuracy. Verification used CPU batches of 32 and four Torch threads.

On the reconstructed split, 31 test direction arrays equal a training input, rising to 143 after stride-10 sampling. Acquisition and circuit identifiers are absent from this archive, so independence across collection events remains unknown. The setting is a passive local observer with known visit boundaries and 100 known classes. The results concern a fixed classifier on historical data. A +200 prefix shifts sampled content and omits the suffix while preserving sampling phase. Prefix symbols, position changes, and tail omission are confounded, so window displacement remains a mechanism hypothesis. Network costs, adaptive attackers, live traffic, onion services, and open-world identification are outside this verification.

## Reproduce the local verification

Python 3.13.5 was used with the versions in `requirements.txt`. Use the exact dataset and checkpoint below, obtained separately. The authors provide the dataset for research use. The archived checkpoint is locally preserved and unavailable in this package, so readers without the matching weights can inspect the code and aggregate result but cannot rerun model inference. Dataset traces, class names, per-example outputs and model weights are excluded from this release. [Authors’ dataset page](https://tor-wf-dl.distrinet-research.be/) · [Rimmer et al., NDSS 2018, §IV](https://www.ndss-symposium.org/wp-content/uploads/2018/02/ndss2018_03A-1_Rimmer_paper.pdf)

| Input | SHA-256 |
|---|---|
| `tor_100w_2500tr.npz` | `6b284576f2b482bc3676bd7d71b863bdd671c67bef611fac683f41a6d15bc750` |
| Transformer `best_model.pth` | `5ca6278abe08c6df725a3388dfa59c5215757c6920eb5689a897d51c15edb881` |

Run from the package root with a fresh output path. The verifier checks the exact local artifacts before loading, reconstructs the test split, and writes aggregate results.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
run_dir=$(mktemp -d)
python verify_transformer.py \
  --dataset /path/to/tor_100w_2500tr.npz \
  --checkpoint /path/to/best_model.pth > "$run_dir/transformer_cpu.json"
```

Use the [MIT license](LICENSE) for the code in this package. The dataset and cited work retain their own terms.
