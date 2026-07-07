# Attractor controls (Qwen2.5-1.5B)

Sample: 200 responses; points per trajectory median 156. Grassberger-Procaccia: Euclidean metric, scaling region 5th-60th percentile of pairwise distances, 15 radii.

## Correlation dimension by layer (finite and stable => low-dimensional attractor)
- early (layer 7): 8.33
- mid (layer 14): 9.84
- late (layer 21): 8.00
- no-context trajectory (mid): 9.34
- iid-Gaussian control (1536-dim, 156 pts): 74.26  (must be high)

## Trajectory movement under chunk removal (mid layer), response-level means
- most-influential (by likelihood): 8.370
- random chunk: 3.275
- length-matched chunk: 3.671
- highest lexical-overlap chunk: 4.175
- first chunk: 4.068

## Paired response-level tests (Wilcoxon signed-rank)
- paired Wilcoxon move_top > move_rand: p = 7.13e-33
- paired Wilcoxon move_top > move_len: p = 4.98e-32
- paired Wilcoxon move_top > move_lex: p = 1.97e-28
- paired Wilcoxon move_top > move_first: p = 3.96e-26

Spearman(move_top, likelihood drop_max), response-level: 0.705 (p=2.52e-31)
move_top grounded 8.554 vs hallucinated 8.186, Mann-Whitney p = 1.48e-01
