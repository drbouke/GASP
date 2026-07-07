"""
Top-level orchestrator — runs all pipeline stages in order.

Usage:
  python pipeline/scripts/run_all.py

Each stage is run as a subprocess so stdout is streamed live. The scoring and
GPU stages require a CUDA device and download models/benchmarks on first use.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
PYTHON = sys.executable

STAGES = [
    ("Score RAGTruth (Qwen2.5-0.5B)",          ["src/scorer.py", "--model", "Qwen/Qwen2.5-0.5B-Instruct"]),
    ("Score RAGTruth (Qwen2.5-1.5B)",          ["src/scorer.py", "--model", "Qwen/Qwen2.5-1.5B-Instruct"]),
    ("Score RAGTruth (SmolLM2-1.7B)",          ["src/scorer.py", "--model", "HuggingFaceTB/SmolLM2-1.7B-Instruct"]),
    ("RAGTruth detection AUC (response + span, per-type)", ["src/analysis.py"]),
    ("Threshold-only vs trained (span)",       ["pipeline/experiments/run_threshold.py"]),
    ("Threshold-only vs trained (response)",   ["pipeline/experiments/run_threshold_resp.py"]),
    ("Detection by task type",                 ["pipeline/experiments/run_pertask.py"]),
    ("Truncation statistics",                  ["pipeline/experiments/run_truncation.py"]),
    ("Whole-context NLI baseline",             ["pipeline/experiments/run_baseline_nli_strong.py"]),
    ("Chunk-level NLI verifier",               ["pipeline/experiments/run_baseline_nli_maxchunk.py"]),
    ("Self-consistency baseline",              ["pipeline/experiments/run_selfcheck.py"]),
    ("Score TofuEval (Qwen2.5-0.5B)",          ["pipeline/experiments/run_tofueval_score.py", "--model", "Qwen/Qwen2.5-0.5B-Instruct"]),
    ("Score TofuEval (Qwen2.5-1.5B)",          ["pipeline/experiments/run_tofueval_score.py", "--model", "Qwen/Qwen2.5-1.5B-Instruct"]),
    ("Score TofuEval (SmolLM2-1.7B)",          ["pipeline/experiments/run_tofueval_score.py", "--model", "HuggingFaceTB/SmolLM2-1.7B-Instruct"]),
    ("TofuEval detection AUC",                 ["pipeline/experiments/run_tofueval_analyze.py"]),
    ("Score RAGBench (Qwen2.5-0.5B)",          ["pipeline/experiments/run_ragbench_score.py", "--model", "Qwen/Qwen2.5-0.5B-Instruct"]),
    ("Score RAGBench (Qwen2.5-1.5B)",          ["pipeline/experiments/run_ragbench_score.py", "--model", "Qwen/Qwen2.5-1.5B-Instruct"]),
    ("RAGBench detection AUC",                 ["pipeline/experiments/run_ragbench_analyze.py"]),
    ("Attribution quality",                    ["pipeline/experiments/run_explanation.py"]),
    ("Attractor dimension estimate",           ["pipeline/experiments/run_attractor_estimate.py"]),
    ("Attractor-movement controls",            ["pipeline/experiments/run_attractor_controls.py"]),
    ("Figures",                                ["pipeline/figures/run_figures.py"]),
    ("Grounding-movement figure",              ["pipeline/figures/run_ifs_figure.py"]),
    ("Attractor trajectory figure",            ["pipeline/figures/run_attractor_figure.py"]),
]


def run_stage(label, argv):
    script = (ROOT / argv[0])
    print(f"\n{'='*70}", flush=True)
    print(f"  STAGE: {label}", flush=True)
    print(f"  Script: {argv[0]}", flush=True)
    print(f"{'='*70}", flush=True)
    result = subprocess.run([PYTHON, str(script)] + argv[1:], cwd=str(ROOT))
    if result.returncode != 0:
        print(f"\n  ERROR: stage '{label}' exited with code {result.returncode}", flush=True)
        sys.exit(result.returncode)
    print(f"\n  DONE: {label}", flush=True)


if __name__ == "__main__":
    print("GASP Pipeline — running all stages", flush=True)
    for label, argv in STAGES:
        run_stage(label, argv)
    print("\n" + "="*70, flush=True)
    print("  ALL STAGES COMPLETE", flush=True)
    print("="*70, flush=True)
