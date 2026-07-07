# TofuEval-MeetingBank second benchmark (grounding detection)

## tofueval_Qwen2.5-0.5B-Instruct  (summaries=884, sentences=2401, halluc=440)

### Response-level AUC [95% CI]
- perplexity: 0.521 [0.481, 0.562]
- length: 0.472 [0.431, 0.512]
- grounding (GASP-trained): 0.632 [0.594, 0.669]
- grounding + baselines: 0.659 [0.620, 0.697]
- GASP-threshold (default): 0.625 [0.584, 0.663]
- grounding vs perplexity: diff +0.108 [+0.054, +0.164], p=0.000

### Sentence-level AUC [95% CI]
- perplexity: 0.555 [0.525, 0.587]
- length: 0.515 [0.485, 0.544]
- grounding (GASP-trained): 0.656 [0.620, 0.691]
- grounding + baselines: 0.672 [0.636, 0.705]
- GASP-threshold (default): 0.693 [0.659, 0.726]
- grounding vs perplexity: diff +0.102 [+0.055, +0.143], p=0.000

## tofueval_Qwen2.5-1.5B-Instruct  (summaries=884, sentences=2401, halluc=440)

### Response-level AUC [95% CI]
- perplexity: 0.548 [0.509, 0.586]
- length: 0.473 [0.428, 0.514]
- grounding (GASP-trained): 0.627 [0.590, 0.665]
- grounding + baselines: 0.633 [0.594, 0.670]
- GASP-threshold (default): 0.623 [0.581, 0.663]
- grounding vs perplexity: diff +0.078 [+0.022, +0.129], p=0.005

### Sentence-level AUC [95% CI]
- perplexity: 0.516 [0.487, 0.545]
- length: 0.515 [0.484, 0.545]
- grounding (GASP-trained): 0.642 [0.607, 0.676]
- grounding + baselines: 0.675 [0.642, 0.709]
- GASP-threshold (default): 0.693 [0.661, 0.726]
- grounding vs perplexity: diff +0.126 [+0.079, +0.168], p=0.000

## tofueval_SmolLM2-1.7B-Instruct  (summaries=884, sentences=2401, halluc=440)

### Response-level AUC [95% CI]
- perplexity: 0.516 [0.479, 0.552]
- length: 0.503 [0.463, 0.548]
- grounding (GASP-trained): 0.641 [0.605, 0.680]
- grounding + baselines: 0.656 [0.614, 0.695]
- GASP-threshold (default): 0.614 [0.575, 0.653]
- grounding vs perplexity: diff +0.123 [+0.074, +0.179], p=0.000

### Sentence-level AUC [95% CI]
- perplexity: 0.524 [0.497, 0.553]
- length: 0.535 [0.503, 0.565]
- grounding (GASP-trained): 0.631 [0.593, 0.664]
- grounding + baselines: 0.680 [0.646, 0.714]
- GASP-threshold (default): 0.693 [0.658, 0.726]
- grounding vs perplexity: diff +0.107 [+0.066, +0.151], p=0.000

