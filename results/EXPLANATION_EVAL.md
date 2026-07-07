# Explanation-quality proxy (Qwen2.5-1.5B, large NLI)

Grounded spans evaluated: 2273 (of 2586 total).

## Entailment of the drop-attributed chunk vs controls (grounded spans)
- attributed (max drop) chunk entails span: mean 0.495
- random chunk: mean 0.045
- highest lexical-overlap chunk: mean 0.065
- best possible chunk (oracle max entail): mean 0.556
- paired Wilcoxon attributed > random: p = 4.55e-261
- paired Wilcoxon attributed > lexical-overlap: p = 4.30e-247

## Attribution agreement with NLI-chosen support (grounded spans)
- drop-attributed chunk == NLI-max-entail chunk: 69.3%  (chance ~ 1/K = 20%)

## Best-chunk entailment separates grounded from hallucinated
- grounded best-entail 0.556 vs hallucinated 0.281, Mann-Whitney p = 1.73e-24

## Response-level aggregation (399 responses with grounded spans)
- attributed 0.499 vs random 0.046 vs lexical 0.068
- response-level agreement mean 68.8%
- paired Wilcoxon (per response) attributed > random: p = 5.69e-65
- paired Wilcoxon (per response) attributed > lexical: p = 3.07e-63
