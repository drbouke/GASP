# Truncation analysis (context 700 tok, answer 200 tok)

## Qwen2.5-0.5B-Instruct_K5  (n=400 responses, 376 annotated spans)
- contexts truncated at 700 tok: 167/400 = 41.8%
- context tokens: median 655, 90th pct 1097, max 2265
- answers truncated at 200 tok: 99/400 = 24.8%
- answer tokens: median 159, 90th pct 266, max 461
- annotated hallucination spans starting within kept answer: 346/376 = 92.0%

## Qwen2.5-1.5B-Instruct_K5  (n=400 responses, 376 annotated spans)
- contexts truncated at 700 tok: 167/400 = 41.8%
- context tokens: median 655, 90th pct 1097, max 2265
- answers truncated at 200 tok: 99/400 = 24.8%
- answer tokens: median 159, 90th pct 266, max 461
- annotated hallucination spans starting within kept answer: 346/376 = 92.0%

## SmolLM2-1.7B-Instruct_K5  (n=400 responses, 376 annotated spans)
- contexts truncated at 700 tok: 190/400 = 47.5%
- context tokens: median 686, 90th pct 1144, max 2331
- answers truncated at 200 tok: 112/400 = 28.0%
- answer tokens: median 163, 90th pct 278, max 479
- annotated hallucination spans starting within kept answer: 338/376 = 89.9%

