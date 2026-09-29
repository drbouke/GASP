# GASP Attribution Human Study Protocol

## Purpose
GASP flags each answer sentence and returns, for grounded sentences, the retrieved chunk whose
removal most lowers the sentence's likelihood as a candidate supporting passage. This study measures how often that returned chunk supports the sentence, as judged by human
annotators, and the paper reports it as a direct measure of localization quality alongside the
automatic grade.

## Materials
`attribution_annotation_sheet.csv` holds 120 items sampled from the RAGTruth grounded sentences scored
by the 14B model. Each row has:
- `sentence`: the answer sentence GASP marked as grounded.
- `attributed_chunk`: the retrieved passage GASP returned as its support.
- `annotator_grade`: to be filled with a single letter, A, B, C, or D.
- `notes`: optional free text.

## Task
For each item, read the sentence and the attributed chunk and decide how well the chunk supports the
sentence. Judge support, not truth in the world: the question is whether this passage backs the claim,
not whether the claim is correct in general. Enter exactly one letter in `annotator_grade`.

- **A (fully supports)**: the chunk states or directly entails the claim; a reader would accept the
  sentence on the strength of this passage alone.
- **B (partly supports)**: the chunk supports part of the claim or supports it with a gap that a
  reasonable reader would still need to bridge.
- **C (topically related)**: the chunk is about the same subject but does not support the specific
  claim.
- **D (does not support)**: the chunk is unrelated or contradicts the claim.

## Guidelines
- Judge only the sentence and the shown chunk. Do not use outside knowledge to fill gaps.
- Paraphrase counts. The chunk need not repeat the wording, only support the meaning.
- If the sentence bundles several facts, grade by whether the chunk supports its main asserted fact.
- When torn between two grades, choose the lower one.

## Annotators and quality control
- Three annotators each graded all 120 items independently, without conferring.
- Annotator identities are not released; each annotator's grades are one file in `Reports/`.

## Analysis
- Report the percentage of items graded A or B (supported) for the attributed chunk.
- Report inter-annotator agreement as mean pairwise exact agreement and Fleiss' kappa, on the
  four-level scale and on the collapsed supported-versus-not distinction, and the majority grade
  per item (`analyze_annotations.py`).
- The automatic LLM-judge grade in the paper was computed on a different sample, so the two rates
  are reported side by side and are not an estimate of their agreement.

## Reporting
Each annotator filled `annotator_grade` for every item; the three graded sheets are
`Reports/annotation1.csv` to `annotation3.csv`, released without the annotators' identities.
