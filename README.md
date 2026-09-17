# GASP

[![PyPI](https://img.shields.io/pypi/v/gasp-rag.svg)](https://pypi.org/project/gasp-rag/)
[![Python](https://img.shields.io/pypi/pyversions/gasp-rag.svg)](https://pypi.org/project/gasp-rag/)
[![arXiv](https://img.shields.io/badge/arXiv-2607.04223-b31b1b.svg)](https://arxiv.org/abs/2607.04223)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Grounding-Aware Sensitivity by Perturbation** — a span-level detector of ungrounded
content in retrieval-augmented generation (RAG).

GASP scores each answer sentence by its *grounding sensitivity*: the change in the
sentence's likelihood when the retrieved context is perturbed. A grounded sentence loses
much of its likelihood when its supporting passage is removed; an unsupported sentence
barely reacts. GASP needs only a probabilistic scorer, no trained verifier and no labeled
data, and it returns, for each sentence, the chunk that best supports it.

This repository is both the installable library (`pip install gasp-rag`) and the code that
reproduces the paper.

## Install

```bash
pip install gasp-rag          # core
pip install gasp-rag[torch]   # with PyTorch and transformers, needed to run a scorer
```

## Quickstart

```python
from gasp import GASP

detector = GASP("Qwen/Qwen2.5-1.5B-Instruct", k_chunks=5, threshold=0.5)

result = detector.detect(
    context="...",   # the retrieved passages, as one string
    answer="...",    # the generated answer to check
    query="...",     # the query (optional)
)

for s in result:
    print(f"{s.sensitivity:+.2f}  {s.text}")
    if s.supporting_chunk:
        print(f"        supported by: {s.supporting_chunk[:80]}...")

for s in result.flagged():
    print("likely unsupported:", s.text)
```

Higher sensitivity means the sentence depends more on the retrieved evidence and is more
likely grounded; lower sensitivity means it barely reacts to removing evidence and is more
likely unsupported. Thresholds are corpus dependent and best calibrated on held-out data.

## Options

```python
GASP(
    model_id,                       # any Hugging Face causal LM, small or large, CPU or GPU
    k_chunks=5,                     # number of context chunks
    threshold=None,                 # flag sentences below this sensitivity
    economical=False,               # two-pass variant: faster, no attribution
    sensitivity_feature="max_drop", # or "gap", "mean_drop", "top2_drop", "max_jsd"
    max_ctx_tokens=1800, max_ans_tokens=256,
    device=None, dtype=None,        # "cpu"/"cuda", "float16"/"bfloat16"/"float32"
)
```

## Many answers, files, and the command line

```python
detections = detector.detect_batch([
    {"context": ctx1, "answer": ans1, "query": q1},
    {"context": ctx2, "answer": ans2},
])
rows = [r for d in detections for r in d.to_records()]   # ready for pandas
```

```bash
# score a file of RAG outputs (jsonl/csv with context, answer, optional query)
gasp detect --input outputs.jsonl --model Qwen/Qwen2.5-1.5B-Instruct \
            --k-chunks 5 --threshold 0.5 --output results.jsonl

# metrics on a labeled results file (label 1 = unsupported)
gasp eval --input labeled.csv --label-col label --score-col sensitivity --threshold 0.5
```

## How it works

For each answer sentence GASP re-scores the fixed answer under three conditions, the full
context, no context, and each context chunk removed in turn, and reads the log-likelihood
drops and Jensen-Shannon divergences at the sentence's tokens. The largest per-chunk drop
is the sentence's grounding sensitivity, and the chunk that produced it is returned as the
candidate supporting passage. Every method sees the same character-span chunks and
sentences, so the segmentation is defined once and never re-tokenized.

## Reproducing the paper

The `pipeline/` directory holds the experiment code and `results/` holds the figures and the
human-study package.

```bash
pip install -r requirements.txt
# 1. score answers into canonical cases (RAGTruth / TofuEval / RAGBench)
python pipeline/run_gasp.py --model Qwen/Qwen2.5-14B-Instruct --dataset ragtruth --tag RT
# 2. run the advanced baselines on the same cases
python pipeline/run_baselines.py --cases canon_results/RT/cases.jsonl --out baselines_nli.csv
# 3. the decisive comparison and the complementarity analysis
python pipeline/compare_baselines.py --gasp canon_results/RT/sentence.csv --baselines *.csv
python pipeline/integration_analysis.py --gasp canon_results/RT/sentence.csv --baselines *.csv
# 4. cost panel and figures
python pipeline/cost_analysis.py --cases canon_results/RT/cases.jsonl
python pipeline/make_paper_figures.py
```

Key findings: GASP beats entailment and attribution baselines and is competitive with the
per-chunk trained fact-checkers, though a full-context fact-checker and an LLM judge rank spans
more accurately at higher compute. Adding GASP to a verifier improves the weaker entailment,
attribution, and per-chunk verifiers but not the strongest full-context fact-checker or the LLM
judge, whose ranking already reflects it, so GASP is best used as a cheap, training-free
standalone detector with built-in attribution and as a complement to weaker verifiers. Its
attribution is validated by a three-annotator human study (see `results/human_study/`).

## Citation

```bibtex
@article{bouke2026gasp,
  title   = {Detecting Hallucinations in Retrieval-Augmented Generation through
             Grounding-Aware Sensitivity by Perturbation (GASP)},
  author  = {Bouke, Mohamed Aly},
  journal = {arXiv preprint arXiv:2607.04223},
  year    = {2026},
  doi     = {10.48550/arXiv.2607.04223}
}
```

## License

MIT. See [LICENSE](LICENSE).
