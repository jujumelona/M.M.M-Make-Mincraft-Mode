# Observation Rule Compiler Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans task by task in this session.

**Goal:** Replace production feature callback source authoring with small typed rule decisions compiled by the host.

**Architecture:** Separate observation binding, typed value/effect compilation and Java callback assembly. Retain exact inspected symbols and existing compiler admission; persist accepted rule IR.

**Tech Stack:** Python, existing JavaOperations/Tree-sitter/jsonschema, inspected Fabric ABI, Java.

**Spec:** docs/superpowers/specs/2026-10-22-observation-rule-compiler-design.md

## Global Constraints

Main only; no prompt-specific templates; no new model/provider or runtime dependency; no free Java fallback in the rule route. Full Minecraft and actual 9B runtime certification require their real execution.

## Review Focus

* Changes shared between worlds/players must retain separate previous values.
* A false/null/zero previous observation must not look like missing history.
* Overloads, inherited symbols and static state calls retain exact inspected types.
* An OR trigger cannot silently become an unconditional periodic effect.
* Required state authority and legacy compiler repair cannot bypass the typed rule path.

### Task 1: Bind observation operators

Create `minecraft_mod_ai/host_rule_observation.py`, test `tests/test_host_rule_observation.py`.

Produces `bind_rule_observation(router, behavior, candidates) -> dict`.

- [ ] Test the captured calendar triggers with an inspected tick while rejecting native-hook substitution and uninspected IDs.
- [ ] Run failing tests, implement semantic observation selection and paged native selection, run passing tests.
- [ ] Commit and push the independently verified binding component.

### Task 2: Typed value and effect authoring

Create `minecraft_mod_ai/host_rule_program.py`, test `tests/test_host_rule_program.py`.

Produces a typed program builder with literal, parameter, symbol, arithmetic,
comparison, selection, effects and scoped block/iteration operations. Uses
`JavaOperations` to check the actual ABI rather than model-provided Java types.

- [ ] Test chained clock arithmetic, typed state writes, injection/type rejection, scopes, short circuit and distinct overloads.
- [ ] Run failing tests, implement the builder and paged small-model authoring, run passing tests.
- [ ] Commit and push.

### Task 3: Compile and integrate production rules

Create `minecraft_mod_ai/host_rule_feature.py`; modify
`minecraft_mod_ai/construction_host_source.py`; test
`tests/test_host_rule_feature.py`.

Produces `compose_rule_feature(router, spec, context, *, package, name, behaviors, state_access) -> str`.

- [ ] Test real construction routing, no source/repair fallback, rule IR receipts and compiled Java activation/dispatch behavior.
- [ ] Run failing tests, implement callbacks and keyed edge/change tracking, wire default production route, run passing tests.
- [ ] Run regression tests, lint and project suite; document actual results and runtime limits.
- [ ] Commit and push verified implementation and evidence to main.
