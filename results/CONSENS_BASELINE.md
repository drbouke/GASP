# Answer-level (ConSens-style) baseline versus sentence-level gap

Source-level 60/40 split, seed 0; paired source-level bootstrap (2000 resamples), two-sided 95% CI.
Produced by `pipeline/consens_baseline.py` from `canon_results/<tag>/sentence.csv`.

```
## RT_14B/sentence.csv
test: 1591 sentences, 240 sources, 213 positives
span level      sentence gap AUC 0.745 | answer gap (ConSens-style) AUC 0.690 | diff +0.057 [+0.027, +0.086] sig
response level  answer gap (ConSens-style) AUC 0.736 | lowest sentence gap AUC 0.814 | diff +0.077 [+0.033, +0.119] sig  (240 test answers, 120 positive)
## TE_14B/sentence.csv
test: 973 sentences, 20 sources, 187 positives
span level      sentence gap AUC 0.700 | answer gap (ConSens-style) AUC 0.668 | diff +0.033 [+0.004, +0.066] sig
response level  answer gap (ConSens-style) AUC 0.608 | lowest sentence gap AUC 0.697 | diff +0.089 [+0.037, +0.133] sig  (353 test answers, 139 positive)
## RB_14B/sentence.csv
test: 1194 sentences, 210 sources, 309 positives
span level      sentence gap AUC 0.603 | answer gap (ConSens-style) AUC 0.535 | diff +0.071 [+0.027, +0.127] sig
response level  answer gap (ConSens-style) AUC 0.604 | lowest sentence gap AUC 0.643 | diff +0.040 [-0.041, +0.124] ns  (245 test answers, 108 positive)
## RT_1p5B/sentence.csv
test: 1591 sentences, 240 sources, 213 positives
span level      sentence gap AUC 0.717 | answer gap (ConSens-style) AUC 0.665 | diff +0.053 [+0.022, +0.085] sig
response level  answer gap (ConSens-style) AUC 0.694 | lowest sentence gap AUC 0.725 | diff +0.031 [-0.015, +0.076] ns  (240 test answers, 120 positive)
## RT_7B/sentence.csv
test: 1591 sentences, 240 sources, 213 positives
span level      sentence gap AUC 0.733 | answer gap (ConSens-style) AUC 0.699 | diff +0.035 [+0.006, +0.066] sig
response level  answer gap (ConSens-style) AUC 0.780 | lowest sentence gap AUC 0.782 | diff +0.002 [-0.040, +0.047] ns  (240 test answers, 120 positive)
## RT_Llama8B/sentence.csv
test: 1596 sentences, 240 sources, 213 positives
span level      sentence gap AUC 0.721 | answer gap (ConSens-style) AUC 0.676 | diff +0.046 [+0.016, +0.078] sig
response level  answer gap (ConSens-style) AUC 0.733 | lowest sentence gap AUC 0.743 | diff +0.010 [-0.040, +0.062] ns  (240 test answers, 120 positive)
## RT_Phi35/sentence.csv
test: 1532 sentences, 240 sources, 200 positives
span level      sentence gap AUC 0.722 | answer gap (ConSens-style) AUC 0.693 | diff +0.029 [+0.000, +0.057] sig
response level  answer gap (ConSens-style) AUC 0.720 | lowest sentence gap AUC 0.741 | diff +0.020 [-0.020, +0.060] ns  (240 test answers, 119 positive)
## RT_SmolLM/sentence.csv
test: 1580 sentences, 240 sources, 208 positives
span level      sentence gap AUC 0.717 | answer gap (ConSens-style) AUC 0.666 | diff +0.052 [+0.020, +0.084] sig
response level  answer gap (ConSens-style) AUC 0.680 | lowest sentence gap AUC 0.653 | diff -0.028 [-0.073, +0.020] ns  (240 test answers, 120 positive)
```
