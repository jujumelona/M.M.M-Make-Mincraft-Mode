# MMM Contract Index

Small-model agents should inspect this directory before changing stage hand-offs.

Rules:
- Reuse canonical contract types; do not redeclare look-alike dictionaries.
- Producers and consumers must import the same contract symbol.
- Add a new contract only when no canonical owner already exists.
- Prefer explicit dataclasses/enums/TypedDicts over untyped dict payloads.
- Keep one semantic owner per contract; compatibility names may re-export only.

Current canonical import surface:
- `TargetContract` -> `minecraft_mod_ai.contracts.target.TargetContract`

Additional execution/planning contracts must be wired only after existing owners are checked.
