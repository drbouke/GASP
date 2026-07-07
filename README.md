# GASP: Grounding-Aware Sensitivity by Perturbation

Reproducible code for the paper **Detecting Hallucinations in Retrieval-Augmented Generation through Grounding-Aware Sensitivity by Perturbation (GASP)** (Bouke, 2026).

<!-- Uncomment and fill once the preprint is public (replace XXXX.XXXXX):
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.XXXX.XXXXX-blue.svg)](https://doi.org/10.48550/arXiv.XXXX.XXXXX)
-->

**Paper:** preprint link to be added.

GASP is a span-level detector for hallucination in retrieval-augmented generation. It scores each answer sentence by how much its likelihood depends on the retrieved evidence, re-scoring a fixed answer under the full context, under no context, and under leave-one-context-out perturbations, then reading the resulting log-likelihood drops and Jensen-Shannon divergences. A grounded sentence reacts strongly when its evidence is removed; a hallucinated one barely reacts. The default detector is a training-free threshold on the standardized features; a gradient-boosted (LightGBM) classifier is provided for comparison.

---

## Repository Layout

```
.
├── src/                              # Core library modules
│   ├── config.py                     # Repository paths (results/, figures/)
│   ├── scorer.py                     # LM K+2 scoring; grounding-sensitivity features
│   └── analysis.py                   # Feature aggregation, leakage-clean CV, AUC, bootstrap
│
├── pipeline/                         # Experiment entry points
│   ├── experiments/
│   │   ├── run_threshold.py          # Training-free threshold vs trained classifier (span)
│   │   ├── run_threshold_resp.py     # Same, response level
│   │   ├── run_baseline_nli_strong.py    # Whole-context NLI entailment baseline
│   │   ├── run_baseline_nli_maxchunk.py  # Chunk-level (max-over-chunk) NLI verifier
│   │   ├── run_selfcheck.py          # SelfCheckGPT-style self-consistency baseline
│   │   ├── run_pertask.py            # Detection by RAGTruth task type
│   │   ├── run_truncation.py         # Effect of the context/answer token caps
│   │   ├── run_explanation.py        # Attribution quality vs NLI/lexical controls
│   │   ├── run_attractor_estimate.py # Hidden-state correlation-dimension estimate
│   │   ├── run_attractor_controls.py # Attractor-movement controls
│   │   ├── run_tofueval_score.py     # Score the TofuEval-MeetingBank benchmark
│   │   ├── run_tofueval_analyze.py   # TofuEval detection AUC
│   │   ├── run_ragbench_score.py     # Score the RAGBench scope probe
│   │   └── run_ragbench_analyze.py   # RAGBench detection AUC
│   ├── figures/
│   │   └── run_figures.py            # Response/span AUC bars, ROC, distribution, importances
│   └── scripts/
│       └── run_all.py                # Top-level orchestrator (runs all stages)
│
├── results/                          # Precomputed outputs (committed)
│   ├── <model>_K5/                   # Per-scorer feature CSVs (sentence.csv, response.csv)
│   ├── tofueval_<model>/             # TofuEval feature CSVs
│   ├── ragbench_<model>/             # RAGBench feature CSVs
│   ├── *.csv, *.md                   # Result tables and summaries
│   └── figures/                      # Figure PNGs
│
├── requirements.txt
└── README.md
```

---

## Datasets

The three benchmarks are loaded on demand via the Hugging Face `datasets` library; no raw data is stored in the repository.

| Benchmark | Role | Source |
|---|---|---|
| RAGTruth | Primary, span-level | [`wandb/RAGTruth-processed`](https://huggingface.co/datasets/wandb/RAGTruth-processed) |
| TofuEval (MeetingBank) | Cross-domain transfer | [`amazon-science/tofueval`](https://github.com/amazon-science/tofueval) + [`huuuyeah/meetingbank`](https://huggingface.co/datasets/huuuyeah/meetingbank) |
| RAGBench | Short-answer QA scope probe | [`rungalileo/ragbench`](https://huggingface.co/datasets/rungalileo/ragbench) |

Precomputed feature CSVs are included under `results/`, so the analysis and figure stages can be reproduced without re-scoring on a GPU.

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended. Scoring uses a single CUDA GPU (developed on a 6 GB RTX 3060 Laptop). All random seeds are fixed at `42`.

---

## Reproducing All Results

### Run the full pipeline (recommended)
```bash
python pipeline/scripts/run_all.py
```
This runs all stages in order and writes outputs to `results/` and `results/figures/`. The scoring and GPU stages download the models and benchmarks on first use.

### Run individual stages
```bash
# Score the answers on the GPU, one per scorer (writes results/<model>_K5/)
python src/scorer.py --model Qwen/Qwen2.5-0.5B-Instruct
python src/scorer.py --model Qwen/Qwen2.5-1.5B-Instruct
python src/scorer.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct

# RAGTruth detection AUC (response + span, per-type)
python src/analysis.py

# Training-free threshold vs trained classifier
python pipeline/experiments/run_threshold.py
python pipeline/experiments/run_threshold_resp.py

# Baselines (whole-context NLI, chunk-level NLI verifier, self-consistency)
python pipeline/experiments/run_baseline_nli_strong.py
python pipeline/experiments/run_baseline_nli_maxchunk.py
python pipeline/experiments/run_selfcheck.py

# Detection by task type, truncation, attribution quality, attractor evidence
python pipeline/experiments/run_pertask.py
python pipeline/experiments/run_truncation.py
python pipeline/experiments/run_explanation.py
python pipeline/experiments/run_attractor_estimate.py
python pipeline/experiments/run_attractor_controls.py

# Cross-domain transfer (TofuEval) and short-answer QA scope probe (RAGBench)
python pipeline/experiments/run_tofueval_analyze.py
python pipeline/experiments/run_ragbench_analyze.py

# Figures
python pipeline/figures/run_figures.py
```

All outputs are written to `results/figures/` (PNGs) and `results/` (CSVs and Markdown summaries).

---

## Key Design Decisions

**Leakage-clean evaluation.** Span-level cross-validation folds are grouped by response with `StratifiedGroupKFold`, so sentences from one answer never appear in both the training and test partitions. See `src/analysis.py`.

**Training-free default.** The recommended detector is a threshold on the negated standardized sum of the four grounding-sensitivity features, needing no labeled data. The LightGBM classifier is reported only for comparison.

**Scorer independent of the generator.** The scoring model needs only to return token likelihoods under a context, so a small model can audit answers from a larger or hosted generator whose internals are unavailable.

**Deterministic scoring.** The answer is never regenerated, so given the model and the seed the features are exactly reproducible; there is no sampling temperature to control.

---

## Citation

```bibtex
@misc{bouke2026gasp,
      title={Detecting Hallucinations in Retrieval-Augmented Generation through Grounding-Aware Sensitivity by Perturbation},
      author={Mohamed Aly Bouke},
      year={2026},
      eprint={XXXX.XXXXX},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      note={Preprint. arXiv identifier and DOI to be added.},
}
```
