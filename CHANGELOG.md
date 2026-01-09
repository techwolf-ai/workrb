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
