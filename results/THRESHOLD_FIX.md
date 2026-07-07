# Threshold-only, corrected protocol (grouped CV, per-fold standardization)

## Qwen2.5-0.5B-Instruct_K5
- trained GASP (grouped CV):            0.635  [0.605, 0.663]
- threshold-only (grouped CV, per-fold): 0.672  [0.641, 0.701]
- threshold-only (OLD: global std, no CV, whole data): 0.672

## Qwen2.5-1.5B-Instruct_K5
- trained GASP (grouped CV):            0.645  [0.614, 0.678]
- threshold-only (grouped CV, per-fold): 0.673  [0.645, 0.700]
- threshold-only (OLD: global std, no CV, whole data): 0.673

## SmolLM2-1.7B-Instruct_K5
- trained GASP (grouped CV):            0.657  [0.624, 0.688]
- threshold-only (grouped CV, per-fold): 0.681  [0.655, 0.708]
- threshold-only (OLD: global std, no CV, whole data): 0.681

