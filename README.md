# SimulacraBench submission: Rui Codex (team `ruicodex`)

This repository holds the code of the Rui Codex entry to [SimulacraBench](https://simulacrabench.org/), the NeurIPS 2026 competition. It is released under Apache-2.0 (see `LICENSE`).

`main.py` defines `predict(frame, schema)`, as the competition contract requires. Everything is fitted at call time on the visible respondents that the harness passes in. The repository contains no survey data, no pretrained weights and no external datasets.

## What was submitted
The Test-phase submission is a ZIP of exactly two files from this repository, `main.py` and `requirements.txt`, at the archive root. These are byte-identical to our last Development submission (v6, Codabench submission 1020179).

| File | SHA-256 |
|---|---|
| `main.py` | `443d031e0e4d9209724719c02af571b618f12e42b0fbe68d0436eabc29825ed0` |
| `requirements.txt` | `695e0c9af1e719ddbfb665c57c5ab2248fd875a89259da07b79293573f85fd18` |

Development-phase skill of this code (privacy-rounded by the organisers):

| Mean | UNICEF | World Bank | UNHCR |
|---:|---:|---:|---:|
| 0.476 | 0.350 | 0.234 | 0.846 |

## Method
- **Features:** one-hot and ordinal codes of the GIVEN block. These are the only columns visible for held-out respondents.
- **Learners**, all trained jointly over the items of one instrument where possible:
  - multi-task logistic regression, with the ridge strength chosen per item;
  - a shared-trunk MLP, with up to 3 seeds bagged;
  - calibrated naive Bayes;
  - per-item LightGBM;
  - a GIVEN-profile nearest-neighbour model shrunk towards the logistic prediction;
  - logistic regression on pairwise GIVEN interactions;
  - the crowd marginal.
- **Stacking:** 5-fold out-of-fold probabilities, scored with the grader's own floor mixing, feed per-item convex weights. These are fitted by EM under a Dirichlet prior centred on the instrument-wide weights. A per-item temperature follows.
- **Skip logic:** deterministic rules apply where the gate parent is a GIVEN item. Where the parent is a PREDICT item, a factorisation computes P(asked | x) from the parent's predicted distribution.
- **Time ledger:** the phase budget (900 s in Development, 3600 s in Test, detected from the visible-row count) is shared over the three instruments by item count. Optional components are added only while the allowance lasts. Every stage falls back to the crowd marginal on error.

The full description is in our technical report on OpenReview (NeurIPS 2026 Competition: SimulacraBench).

## Reproduce
Use the organisers' harness at https://github.com/SituatedEvals/public. The real microdata are not public. The organisers run the code on the hidden data; anyone can run it on the synthetic sandbox:

```
git clone https://github.com/SituatedEvals/public && cd public
pip install -r requirements.txt
python make_sandbox.py --schema data/unicef.json --out _sandbox/unicef
python score.py --submission <path to this repository> --data _sandbox/unicef --schema data/unicef.json --phase 2
```

Repeat for `world_bank` and `unhcr`. Requirements: the hosted SimulacraBench image (Python 3.13, PyTorch, NumPy, pandas, scikit-learn) plus `lightgbm==4.6.0`.

Seeds are fixed (`SEED = 20261029`). How many optional components run depends on the wall-clock allowance, so a run on slower hardware can use fewer components than the hosted H100 run.
