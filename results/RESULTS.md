# Results

Headline numbers from the source-level evaluation (14B scorer unless noted).
Figures are in `results/figures/`; the human study is in `results/human_study/`. All AUC are
span-level ROC-AUC under a leakage-clean, source-level 60/40 split with paired source-level
bootstrap confidence intervals (2000 resamples). Regenerated on an NVIDIA L40S 48 GB.

## Benchmark composition (canonical case set, 14B scorer)

| Quantity | RAGTruth | TofuEval | RAGBench |
|---|---|---|---|
| Responses (hallucinated / clean) | 600 (300 / 300) | 884 (309 / 575) | 598 (267 / 331) |
| Sentences (hallucinated / grounded) | 4,191 (534 / 3,657) | 2,415 (446 / 1,969) | 2,951 (776 / 2,175) |
| Sources (leakage-clean split unit) | 600 | 50 | 524 |
| Scorers | 6 | 1 | 1 |

RAGTruth is class-balanced at the response level by sampling; TofuEval and RAGBench follow
their available distribution. Sentence counts use the Qwen tokenizer and vary slightly across
scorers.

## Span-level detection on RAGTruth (six scorers, four families)

GASP+base above perplexity throughout and stable across scale and family.

| Scorer | Perplexity | GASP-threshold (gap) | GASP-trained | GASP+base |
|---|---|---|---|---|
| Qwen2.5-1.5B | 0.602 | 0.717 | 0.724 | 0.741 |
| Qwen2.5-7B | 0.663 | 0.733 | 0.734 | 0.746 |
| Qwen2.5-14B | 0.657 | 0.745 | 0.763 | 0.773 |
| SmolLM2-1.7B | 0.597 | 0.717 | 0.723 | 0.744 |
| Llama-3.1-8B | 0.626 | 0.721 | 0.728 | 0.744 |
| Phi-3.5-mini | 0.666 | 0.722 | 0.732 | 0.746 |

Note: GASP-threshold is the single two-pass gap feature with
its direction fixed a priori (training-free); GASP+base is the logistic classifier on the six
sensitivity features plus perplexity and length. GASP+base beats the perplexity+length baseline
by +0.117 [0.082, 0.153] on the 14B scorer.

## Transfer across benchmarks (14B scorer, span level)

| Benchmark | Span AUC |
|---|---|
| RAGTruth | 0.773 |
| TofuEval (summarization) | 0.711 |
| RAGBench (short-answer QA) | 0.634 |

Per task within RAGTruth (with source-level 95% CIs): question answering 0.800 [0.751, 0.861],
summarization 0.775 [0.706, 0.837], data-to-text 0.736 [0.697, 0.776]. The intervals overlap, so
no task is distinguishably strongest, but question answering is at least as strong as the rest —
so the RAGBench weakness is specific to its short factual answers and heavier truncation, not the
QA format. Two senses of transfer differ: GASP-threshold is a fixed detector that carries across
benchmarks unchanged (only its threshold re-calibrated), whereas GASP+base re-fits its classifier
on each benchmark's own development split.

## GASP vs advanced verifiers (span-level, best variant, 14B)

| Method | RAGTruth | TofuEval | RAGBench |
|---|---|---|---|
| chunk-NLI | 0.721 | 0.727 | 0.636 |
| ContextCite | 0.736 | 0.699 | 0.668 |
| AlignScore | 0.791 | 0.783 | 0.650 |
| MiniCheck | 0.831 | 0.788 | 0.647 |
| LLM judge (14B) | 0.853 | 0.881 | 0.714 |
| GASP-threshold (training-free) | 0.745 | 0.700 | 0.603 |
| **GASP+base** | **0.773** | **0.711** | **0.634** |

GASP+base beats chunk-NLI and ContextCite and matches the per-chunk trained verifiers; the
full-context fact-checker and the LLM judge are more accurate at markedly higher compute. The
training-free GASP-threshold (the single two-pass gap feature) trails GASP+base but still beats
the entailment and attribution baselines.

## Complementarity (adding GASP on top of a verifier), RAGTruth

Comparison against each verifier's OWN span-ranking AUC on the same test ids (`Δ vs raw`), plus a
perplexity+length control column (`+ppl,len`) and the direct paired difference of `+GASP` over
that control (`Δ vs +ppl,len`). `*` = paired interval excludes zero.

Both deltas are the observed test-set difference of the AUC columns with a source-level bootstrap
CI. Δ₁ = +GASP − alone; Δ₂ = +GASP − (+ppl,len). `*` = interval excludes zero.

| Verifier | alone | +GASP | +ppl,len | Δ₁ vs alone [95% CI] | Δ₂ vs +ppl,len [95% CI] |
|---|---|---|---|---|---|
| LLM judge | 0.853 | 0.844 | 0.798 | -0.009 [-0.024, +0.007] | +0.046 [+0.030, +0.063]* |
| MiniCheck (full ctx) | 0.831 | 0.832 | 0.823 | +0.000 [-0.012, +0.013] | +0.008 [-0.006, +0.022] |
| MiniCheck (per chunk) | 0.792 | 0.832 | 0.792 | +0.040 [+0.023, +0.058]* | +0.040 [+0.021, +0.058]* |
| AlignScore (per chunk) | 0.765 | 0.815 | 0.770 | +0.051 [+0.032, +0.070]* | +0.046 [+0.025, +0.066]* |
| ContextCite | 0.736 | 0.765 | 0.744 | +0.029 [+0.014, +0.045]* | +0.020 [+0.004, +0.036]* |
| chunk-NLI (per chunk) | 0.721 | 0.793 | 0.736 | +0.073 [+0.048, +0.097]* | +0.057 [+0.031, +0.083]* |

For the four weaker verifiers both deltas exclude zero: GASP raises each over its own AUC and
over the cheap-feature control. The two strongest separate the questions: adding GASP does not
raise the LLM judge or the full-context MiniCheck over its own AUC (Δ₁ spans zero); against the
control it still beats perplexity+length for the LLM judge (Δ₂ = +0.046*) but **not for the
full-context MiniCheck (Δ₂ = +0.008, ns)**. So GASP adds more than cheap features for five of six
verifiers, but **with the tested fusion and data we found no significant improvement on the two
strongest verifiers' own ranking.** On RAGBench no verifier is significantly improved; on TofuEval
adding GASP slightly lowers the LLM judge. GASP is thus best used as a cheap, training-free
standalone detector with attribution and as a complement to weaker verifiers, not as a booster for
a state-of-the-art verifier.

## Cost (forward-pass FLOPs proxy, GFLOPs/sentence)

The two-pass GASP-threshold on a 1.5B scorer (~429) is the cheapest configuration, using only the
full-context and no-context passes. Of the two verifiers that beat GASP+base, the LLM judge
(~20,376) is far more expensive, but the full-context MiniCheck (~717) is actually cheaper than
the 14B GASP+base (~4,236) — it trades a separately trained, supervised model for its accuracy,
not more compute. ContextCite (~26,103) is the most expensive. Read each variant's cost with its
own accuracy; do not pair the small-scorer cost with the larger-scorer accuracy. We report a
compute estimate, not wall-clock latency.

## Attribution quality (human study)

Three annotators graded the GASP-attributed chunk for 120 grounded sentences. It was judged
fully supporting in 58% of cases and fully or partly supporting in 82.5% (Fleiss' kappa 0.86 on
the four levels, 0.94 supported-vs-not), close to the automatic LLM-judge fully-or-partly rate
of 88.1%. The study covers grounded, attributed sentences only.

## Truncation audit (1800-token context, 256-token answer caps)

Negligible on RAGTruth (99.9% context / 98.2% answer retained; 13 hallucination spans excluded)
and TofuEval (fully retained); material on RAGBench (94.9% context retained on average, down to
11.7% in the extreme; 23.4% of cases partly truncated; 72 hallucination spans excluded), a
caveat on the already-weak RAGBench numbers.
