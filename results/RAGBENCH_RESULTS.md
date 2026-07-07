# RAGBench third benchmark (multi-domain RAG, grounding detection)

## ragbench_Qwen2.5-0.5B-Instruct  (responses=797, sentences=3858, halluc_sent=778)
domains: pubmedqa:414, finqa:224, techqa:60, hotpotqa:45, covidqa:36, delucionqa:18

### Response-level AUC [95% CI]
- perplexity: 0.569 [0.528, 0.608]
- length: 0.549 [0.509, 0.592]
- grounding (GASP-trained): 0.573 [0.532, 0.613]
- grounding + baselines: 0.594 [0.555, 0.634]
- GASP-threshold (default): 0.461 [0.422, 0.501]
- grounding vs perplexity: diff +0.001 [-0.053, +0.059], p=0.490

### Sentence-level AUC [95% CI]
- perplexity: 0.613 [0.590, 0.634]
- length: 0.515 [0.492, 0.539]
- grounding (GASP-trained): 0.595 [0.572, 0.618]
- grounding + baselines: 0.668 [0.646, 0.690]
- GASP-threshold (default): 0.576 [0.540, 0.610]
- grounding vs perplexity: diff -0.018 [-0.046, +0.010], p=0.887

## ragbench_Qwen2.5-1.5B-Instruct  (responses=797, sentences=3858, halluc_sent=778)
domains: pubmedqa:414, finqa:224, techqa:60, hotpotqa:45, covidqa:36, delucionqa:18

### Response-level AUC [95% CI]
- perplexity: 0.566 [0.526, 0.604]
- length: 0.550 [0.511, 0.591]
- grounding (GASP-trained): 0.575 [0.536, 0.617]
- grounding + baselines: 0.609 [0.570, 0.648]
- GASP-threshold (default): 0.455 [0.419, 0.495]
- grounding vs perplexity: diff +0.011 [-0.044, +0.064], p=0.363

### Sentence-level AUC [95% CI]
- perplexity: 0.619 [0.599, 0.642]
- length: 0.516 [0.492, 0.540]
- grounding (GASP-trained): 0.596 [0.573, 0.618]
- grounding + baselines: 0.679 [0.659, 0.700]
- GASP-threshold (default): 0.573 [0.540, 0.607]
- grounding vs perplexity: diff -0.023 [-0.051, +0.005], p=0.945

## ragbench_SmolLM2-1.7B-Instruct: (missing)

