# GASP Attribution Human Study Protocol

## Purpose
GASP flags each answer sentence and returns, for grounded sentences, the retrieved chunk whose
removal most lowers the sentence's likelihood as a candidate supporting passage. This study measures
how often that returned chunk is a genuine supporting passage, judged by human annotators, which the
paper reports as a direct measure of localization quality alongside the automatic proxy.

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
- Recruit at least three annotators who each grade all 120 items independently.
- Include the two authors' names in an annotator log; annotators must not confer during grading.
- Seed 10 hidden check items with an obvious A and an obvious D to catch inattentive grading; exclude
  an annotator whose check accuracy is below 80 percent.

## Analysis
- Report the percentage of items graded A or B (supported) for the attributed chunk.
- Report inter-annotator agreement with Krippendorff's alpha on the four-level scale, and the majority
  grade per item.
- Compare against the automatic LLM-judge grade in the paper; agreement between the human majority and
  the automatic grade bounds how far the automatic proxy can stand in for human judgment.
- A supported rate well above the 1/K random-chunk rate establishes that the returned chunk is a
  genuine supporting passage rather than an arbitrary one.

## Reporting
Fill `annotator_grade` for every item, save one CSV per annotator as
`annotation_<name>.csv`, and keep the annotator log alongside. The study reports these files, not
the individual annotators' identities.
