# Agent Instructions

## Scope
- Applies to the whole repository.
- Follow more specific instructions if added later.
- Keep unrelated user changes intact.

## Project Context
- Python scripts for CAM6 LFL pre-processing.
- Scientific configuration lives in `aerosol*.yaml` and `bands.yaml`.
- Inputs and outputs may be large NetCDF or table files outside the repo.

## Working Style
- Keep edits small and behavior-focused.
- Preserve existing procedural script style unless asked to refactor.
- Avoid hard-coded local data paths; use arguments, YAML, or environment variables.

## Validation
- No package manager or test suite is declared.
- Syntax-check Python changes with `python -m compileall .`.
- For data-processing changes, run a focused script smoke test when sample data is available.

## Communication
- Keep responses brief.
- Sentence fragments are okay.
- Present points three at a time.
