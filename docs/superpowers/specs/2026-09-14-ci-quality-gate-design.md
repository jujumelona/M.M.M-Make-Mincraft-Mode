# CI Quality Gate Design

## Goal

Make `main-ci` the authoritative correctness and engineering-quality gate for the repository. A green result must mean that the complete test suite and production audits ran, and that a change did not introduce new structural complexity, dependency cycles, duplicated implementations, obvious serialized expensive work, or CI topology gaps.

## Architecture

The gate has four layers:

1. **Correctness/runtime:** execute the entire Python test suite, package/bootstrap checks, real Fabric integrity, root-cause audit, and model/runtime boundary tests.
2. **Static engineering quality:** keep Ruff, Vulture, compile/import checks, and add a repository-specific AST quality regression audit.
3. **Delta regression:** compare the checked-out tree with its first parent. Existing debt is reported, while newly introduced or worsened complexity, cycles, duplication, oversized units, and expensive serial call patterns fail the build. New code also has hard ceilings.
4. **CI self-validation:** verify authoritative jobs remain connected to the final `CI Gate`; local paths referenced by the authoritative workflow must exist; the full `tests` directory remains the test target rather than a hand-maintained allowlist.

## Quality signals

The repository-specific audit measures per Python file/function:

- cyclomatic complexity;
- function span and file line count;
- positional/keyword parameter count;
- import dependency graph and newly introduced cycles;
- exact duplicate function bodies across files;
- model/network/process/wait calls executed serially inside loops;
- repeated identical expensive calls;
- runtime client/model/executor construction in hot function/loop paths.

The gate compares these signals to the parent commit. It fails on regressions instead of requiring all historical debt to be fixed at once. For newly added functions/files, conservative hard ceilings prevent extreme complexity from entering the repository.

## Model/runtime validation

Actual inference quality and hardware-specific execution remain integration concerns, but model-output handling is deterministic CI territory. The complete test suite must include scripted model responses covering valid output plus malformed, truncated, schema-invalid, contradictory, timeout/exception, retry, and terminal-failure paths wherever those production boundaries exist. Recorded real-model failures can be added as deterministic fixtures and replayed by the same suite.

## Failure semantics

A diagnostic that cannot affect the final authoritative gate is not considered a gate. Every required job is an explicit dependency of `CI Gate`, and `CI Gate` fails unless every required dependency succeeds. Diagnostic artifacts are uploaded on success or failure.

## Constraints

- Work only on `main`; do not create feature branches.
- Do not hide failures with permissive fallbacks.
- Avoid arbitrary fixed project-specific performance numbers when a parent-delta regression check is possible.
- Environment-dependent GPU/native-model workflows remain separate integrations, but their deterministic boundaries must be represented in `main-ci`.
