# Pretrained weights

Two trained SVO-Probes models (3 text layers, `embedding_qubits=3`), each the best-validation seed of its 10-seed set.
Test numbers are on the full test sets (std n=1833, swap n=95); the 10-seed means are in the report.

| dir | method | seed | test std | test swap | 10-seed mean std |
|---|---|---|---|---|---|
| `post_selection_tl3/` | post-selection (`configs/final/vqc_nodiscard.yaml`, statevectors, FS_InfoNCE) | 6 | 0.882 | 0.874 | 0.874 ± 0.011 |
| `discard_warmstart_tl3/` | pure discard, density matrices + Pauli_InfoNCE, warm start from the same-seed post-selection model | 6 | 0.855 | 0.884 | 0.843 ± 0.010 |

Each directory has `best.pt`, the `config.yaml` it was trained with, and `metrics.json`.

## Evaluate
```bash
python scripts/retrieve.py svo-probes        # downloads data + images, parses captions, embeds images (see below)
python scripts/benchmark.py -cp weights/post_selection_tl3/best.pt -o out.json
```
`benchmark.py` reads the `config.yaml` next to the checkpoint. Data paths in the configs are the ones `retrieve.py` writes
(`data/svo-probes/processed/{train,val,test,swap}.pkl`, `image_embeddings.pt`).

## Data
`BaseDatasetRetriever("svo-probes").retrieve()` needs no local files: it downloads DeepMind's `svo_probes.csv`, rebuilds our
train/val/test split from `modules/data_pipeline/manifests/svo-probes/`, and downloads the images from their URLs. About 8%
of the image links have died; the affected pairs are dropped (`data/svo-probes/raw/images_missing.csv`), so the splits are
slightly smaller (train 4477, val 1535, test 1548, swap 83 on 2026-11-04). The `metrics.json` numbers above come from the
complete original data, so expect small differences when re-evaluating on the reduced set.
