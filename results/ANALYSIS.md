# GASP analysis


## Qwen2.5-1.5B-Instruct_K5  (resp=400, sent=2586, halluc sent=313)

### Response-level AUC with 95% bootstrap CI
- perplexity: 0.624  [0.572, 0.675]
- length: 0.553  [0.497, 0.609]
- grounding (GASP): 0.726  [0.676, 0.776]
- grounding + baselines: 0.713  [0.658, 0.760]

#### Response ablation (single features)
- gap: 0.680  [0.622, 0.730]
- jsd_noctx: 0.727  [0.676, 0.779]
- drop_max: 0.620  [0.563, 0.677]
- jsd_loo_max: 0.539  [0.480, 0.592]

#### Response ablation (groups)
- no-context only: 0.716  [0.667, 0.766]
- leave-one-out only: 0.583  [0.526, 0.639]
- likelihood only: 0.695  [0.645, 0.747]
- divergence only: 0.739  [0.689, 0.786]

Response grounding vs perplexity: AUC diff = +0.101 [+0.032, +0.167], bootstrap p = 0.002

### Sentence (grouped)-level AUC with 95% bootstrap CI
- perplexity: 0.565  [0.532, 0.599]
- length: 0.548  [0.512, 0.582]
- grounding (GASP): 0.645  [0.615, 0.675]
- grounding + baselines: 0.678  [0.648, 0.707]

#### Sentence (grouped) ablation (single features)
- gap: 0.620  [0.588, 0.651]
- jsd_noctx: 0.616  [0.583, 0.645]
- drop_max: 0.600  [0.568, 0.632]
- jsd_loo_max: 0.595  [0.563, 0.625]

#### Sentence (grouped) ablation (groups)
- no-context only: 0.638  [0.610, 0.666]
- leave-one-out only: 0.616  [0.582, 0.648]
- likelihood only: 0.636  [0.607, 0.664]
- divergence only: 0.640  [0.609, 0.668]

Sentence (grouped) grounding vs perplexity: AUC diff = +0.082 [+0.040, +0.126], bootstrap p = 0.000

### Aggregation study (sentence, grouped): grounding with different leave-one-out aggregation
- aggregation = max: 0.645  [0.615, 0.675]
- aggregation = mean: 0.625  [0.595, 0.655]
- aggregation = top-2 sum: 0.638  [0.606, 0.669]

### Threshold-only variant (no trained classifier), sentence level
- raw drop_max AUC: 0.349
- raw jsd_loo_max AUC: 0.398
- raw gap AUC: 0.303
- standardized sum of the four features AUC: 0.327

## Qwen2.5-0.5B-Instruct_K5  (resp=400, sent=2586, halluc sent=313)

### Response-level AUC with 95% bootstrap CI
- perplexity: 0.581  [0.524, 0.638]
- length: 0.553  [0.496, 0.611]
- grounding (GASP): 0.706  [0.654, 0.757]
- grounding + baselines: 0.731  [0.679, 0.779]

#### Response ablation (single features)
- gap: 0.712  [0.659, 0.762]
- jsd_noctx: 0.766  [0.719, 0.811]
- drop_max: 0.624  [0.568, 0.679]
- jsd_loo_max: 0.579  [0.524, 0.630]

#### Response ablation (groups)
- no-context only: 0.731  [0.680, 0.777]
- leave-one-out only: 0.554  [0.497, 0.611]
- likelihood only: 0.670  [0.617, 0.724]
- divergence only: 0.716  [0.664, 0.767]

Response grounding vs perplexity: AUC diff = +0.125 [+0.052, +0.193], bootstrap p = 0.000

### Sentence (grouped)-level AUC with 95% bootstrap CI
- perplexity: 0.536  [0.506, 0.568]
- length: 0.547  [0.510, 0.583]
- grounding (GASP): 0.635  [0.608, 0.664]
- grounding + baselines: 0.676  [0.647, 0.703]

#### Sentence (grouped) ablation (single features)
- gap: 0.602  [0.574, 0.632]
- jsd_noctx: 0.602  [0.572, 0.633]
- drop_max: 0.605  [0.571, 0.635]
- jsd_loo_max: 0.594  [0.563, 0.625]

#### Sentence (grouped) ablation (groups)
- no-context only: 0.620  [0.591, 0.647]
- leave-one-out only: 0.606  [0.578, 0.633]
- likelihood only: 0.623  [0.592, 0.655]
- divergence only: 0.630  [0.598, 0.659]

Sentence (grouped) grounding vs perplexity: AUC diff = +0.099 [+0.057, +0.143], bootstrap p = 0.000

### Aggregation study (sentence, grouped): grounding with different leave-one-out aggregation
- aggregation = max: 0.635  [0.605, 0.665]
- aggregation = mean: 0.648  [0.618, 0.677]
- aggregation = top-2 sum: 0.659  [0.634, 0.686]

### Threshold-only variant (no trained classifier), sentence level
- raw drop_max AUC: 0.336
- raw jsd_loo_max AUC: 0.375
- raw gap AUC: 0.313
- standardized sum of the four features AUC: 0.328

## SmolLM2-1.7B-Instruct_K5  (resp=400, sent=2550, halluc sent=306)

### Response-level AUC with 95% bootstrap CI
- perplexity: 0.578  [0.522, 0.630]
- length: 0.539  [0.483, 0.593]
- grounding (GASP): 0.716  [0.667, 0.766]
- grounding + baselines: 0.717  [0.668, 0.765]

#### Response ablation (single features)
- gap: 0.675  [0.622, 0.729]
- jsd_noctx: 0.712  [0.658, 0.760]
- drop_max: 0.583  [0.525, 0.640]
- jsd_loo_max: 0.600  [0.544, 0.653]

#### Response ablation (groups)
- no-context only: 0.691  [0.639, 0.743]
- leave-one-out only: 0.607  [0.552, 0.663]
- likelihood only: 0.700  [0.649, 0.752]
- divergence only: 0.694  [0.641, 0.746]

Response grounding vs perplexity: AUC diff = +0.136 [+0.065, +0.207], bootstrap p = 0.001

### Sentence (grouped)-level AUC with 95% bootstrap CI
- perplexity: 0.615  [0.584, 0.646]
- length: 0.523  [0.489, 0.558]
- grounding (GASP): 0.657  [0.624, 0.687]
- grounding + baselines: 0.702  [0.673, 0.731]

#### Sentence (grouped) ablation (single features)
- gap: 0.661  [0.631, 0.691]
- jsd_noctx: 0.666  [0.640, 0.695]
- drop_max: 0.603  [0.569, 0.639]
- jsd_loo_max: 0.617  [0.587, 0.649]

#### Sentence (grouped) ablation (groups)
- no-context only: 0.652  [0.622, 0.683]
- leave-one-out only: 0.603  [0.571, 0.639]
- likelihood only: 0.656  [0.624, 0.687]
- divergence only: 0.655  [0.624, 0.685]

Sentence (grouped) grounding vs perplexity: AUC diff = +0.043 [+0.001, +0.083], bootstrap p = 0.022

### Aggregation study (sentence, grouped): grounding with different leave-one-out aggregation
- aggregation = max: 0.656  [0.623, 0.689]
- aggregation = mean: 0.662  [0.631, 0.692]
- aggregation = top-2 sum: 0.654  [0.622, 0.685]

### Threshold-only variant (no trained classifier), sentence level
- raw drop_max AUC: 0.327
- raw jsd_loo_max AUC: 0.363
- raw gap AUC: 0.306
- standardized sum of the four features AUC: 0.319

## Chunking sensitivity (Qwen2.5-1.5B, sentence level, grounding AUC with CI)
- K = 3: 0.670  [0.639, 0.699]
- K = 5: 0.645  [0.614, 0.673]
- K = 10: 0.640  [0.611, 0.670]
