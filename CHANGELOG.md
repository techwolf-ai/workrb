## v0.5.1 (2026-03-13)

### Feat

- NDCG metrics for paper
- enable row with dataset counts for results table example
- latex table results reporting for paper example
- ConteXTMatch query batching for out-of-memory on large ranking tasks
- make ESCO language support version dependent
- deduplication strategy for queries and targets to enable datasets with duplicate targets

### Fix

- latex table result set best results in bold for different model groups
- cast prediction matrices explicitly to float
- update dataset_ids in Task based on filtering non-supported ones
- linter fixes
- DatasetConfigNotSupported exceptions are introduced and skipped to solve dynamic dataset loading that can result in 0-length query or target tasks.
- default resolve duplicates in ranking and query

## v0.5.0 (2026-03-09)

### Refactor

- align task groups to paper (JOBSIM/SKILLSIM → Semantic Similarity + Candidate Ranking) (#45)

## v0.4.0 (2026-03-04)

### BREAKING CHANGE

- MetricsResult.language replaced by input_languages/output_languages

### Feat

- add lazy execution filtering and ExecutionMode enum
- add cross-lingual aggregation modes for per-language metrics
- freelancer project ranking
- add unicode normalization to lexical baseline preprocessing
- add lexical baselines for ranking

### Fix

- remove from example the dataset that uses ESCO 1.0.5 but defines UK as supported language
- add language field to MetricsResult for proper per-language aggregation
- solve issues in example files
- include lowercase setting in lexical baseline model names
- import SkillSkape

### Refactor

- use language-grouped averaging in per-task aggregation
- migrate freelancer task to dataset_id-based language mapping
- make language_aggregation_mode a non-optional parameter in evaluate()
- migrate freelancer project matching tasks to load_dataset API
- rename language_results to datasetid_results for consistency with dataset_id abstraction
- generalize dataset indexing from language-based to dataset_id-based

## v0.3.0 (2026-01-09)

### Feat

- SkillSkape dataset as ranking task
- Job title similarity as a ranking task

### Refactor

- move functions centered in run.py for public api to registry.py and results.py.
- rename evaluate.py to run.py to remove ambiguity with workrb.evaluate function

## v0.2.1 (2026-01-06)

### Fix

- README updated with ContextMatch and CurriculumMatchModel. CurriculumMatchModel added to pkg imports.

## v0.2.0 (2026-01-06)

### Feat

- **Context-Match**: Contribution of ConTeXTMatch model
- **skill-encoder**: curriculum skill encoder model for skill extraction tasks, following the work: https://ceur-ws.org/Vol-4046/ (paper 5)

### Fix

- usage example fixed
- wrong order attributes evaluate call in evaluate_multiple_models function (#17)

## v0.1.0 (2025-11-11)

### Fix

- first version 0.1.0 for release
