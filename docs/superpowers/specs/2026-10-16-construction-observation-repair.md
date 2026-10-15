# Construction action, observation, and local repair

The user requires a reliable small-model coding workflow rather than another
symptom-specific paging or token limit. Preserve typed Java operations, exact
API/source reuse, complete tasks, and bounded per-call context. Do not run tests
or model inference in this session; commit each completed change.

## Evidence and references

The production trace selected the unfinished current method repeatedly. The
existing author only checks lexical names during admission and runs the main
JavaOperations validator at completion. Forward-reference parents are saved
before their inputs exist, while the checkpoint only permits append operations.
Thus a later input can expose an invalid parent that the author cannot repair.

- Aider separates a proposal from its concrete edit:
  https://aider.chat/docs/usage/modes.html#architect-mode-and-the-editor-model
- smolagents records the result of each action and uses it on the next step:
  https://huggingface.co/docs/smolagents/main/en/conceptual_guides/react
- mini-SWE-agent formats environment outputs as observations in the action loop:
  https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/agents/default.py

These are workflow references, not evidence that an arbitrary 9B model can
correctly implement all gameplay requirements.

## Design

Reuse JavaOperations inference and access/type rules. Prepare immutable task and
classpath facts once per method. In partial mode, permit explicit holes, check
cycles, validate every closed dependency, and report the culpable node. Do not
certify body completion, returns, exceptions, or gameplay from partial admission.

The host records a rejected operation and its diagnostic as an observation. The
next request edits that slot or the earlier provisional parent that caused the
failure. The model never overwrites a whole class or replays all previous code.
Cancellation, transport, checkpoint corruption, and unclassified host faults
remain failures, not model-repair observations.

Existing nodes can only change through a compare-and-swap checkpoint revision.
The replacement retains its ID. Preserve independent roots and shared inputs;
prune only former inputs no longer reachable from the original roots. Persist
the revised graph and its author-state snapshot atomically. An older cursor
cannot replay after a graph change. Final JavaOperations gates remain mandatory.

Report graph nodes, closed validated nodes, and deferred nodes separately.
None of those counts proves an implemented gameplay effect or runtime success.

