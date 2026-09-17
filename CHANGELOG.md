# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0]

### Added
- `GASP.detect_batch` for scoring many answers at once.
- Command-line interface: `gasp detect` (score a .jsonl/.csv file) and `gasp eval` (metrics).
- Evaluation metrics: `evaluate`, `roc_auc`, `pr_auc`, `threshold_metrics`.
- File I/O helpers: `read_items`, `write_records` (JSONL and CSV).
- `Detection.to_records`, `Detection.summary`, `SentenceResult.to_dict`.
- Options: `economical` two-pass mode, selectable `sensitivity_feature`, `device`, `dtype`.

## [0.1.0]

### Added
- Initial release: the `GASP` span-level grounding-sensitivity detector.
- Canonical sentence and chunk segmentation (`sentence_spans`, `chunk_spans`, `Case`).
- The `Scorer` that re-scores a fixed answer under full, no-context, and leave-one-out
  perturbations and reports per-sentence grounding features and the attributed chunk.
