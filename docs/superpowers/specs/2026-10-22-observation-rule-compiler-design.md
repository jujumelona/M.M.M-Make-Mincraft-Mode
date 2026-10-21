# Observation rules owned by the host

The user authorizes implementation on main, with commits and pushes as work is
verified. The goal is to make the production path tractable for a small local
model. No request-specific seasonal/space/shop templates are allowed.

## Evidence and root cause

Run `0dd57e57200e` contains 405 model requests. Its global Fabric event-selection
requests contain roughly 8,500 prompt tokens. Both the global menu and a second
owner menu return UNBOUND for a condition observable from a server/world tick.
The planner describes an application transition as though it required a native
Fabric event. The Java writer also accepts free Java expressions and asks the
model to discover control flow, local declarations and API composition. Smaller
source fragments have not removed those responsibilities.

## Design

Production request-feature entrypoints use a host rule compiler. A small semantic
choice first distinguishes observing world state from listening to a native
occurrence. The former lowers mechanically to an inspected Fabric tick ABI;
the latter searches bounded pages of inspected event IDs. No English keyword
table chooses a hook. Polling is never an automatic fallback for an unbound
native occurrence.

A model authors one typed value or one effect operation at a time. It chooses
existing value IDs, actual symbol IDs, operators and JSON literals. The host
owns value identities, exact receiver/argument types, overload casts, local
scope, declarations, all Java syntax and callback registration. Arbitrary Java
expressions and source repair are absent from this production route. Existing
JavaOperations type/access checks, Java literal escaping, Tree-sitter parsing
and target javac verification remain authoritative.

Rules have a boolean gate and an explicit activation policy: every matching
observation, a false-to-true edge, or a changed observed value. A changed value
has an explicit first-observation policy. Host tracking is keyed by the real
callback context (or an explicitly authored typed context value); it does not
share one global previous value across worlds/players. Gate evaluation and
tracking precede effects, and skipped effects do not produce a handled result.
Tracking is transient; persistent gameplay state continues to use the canonical
typed state API.

Effects support checked calls, field/array assignment, nested conditions and
iteration over inspected arrays/Iterable values. Required scalar state writes
must remain in the rule's state authority; calls/returns alone are not evidence
that every gameplay obligation was fulfilled. The rule IR is retained for
inspection. Unsupported capabilities are explicit and never replaced by a
different feature or a success receipt.

## Alternatives investigated

* JavaPoet (maintained Palantir fork) generates Java syntax; it does not lower
  application triggers or choose gameplay semantics.
* JavaParser parses/resolves Java; the repository already has Tree-sitter,
  bytecode symbols and javac. Adding a second parser does not remove model work.
* CEL Java separates typed pure expressions from effects. Adopt that boundary
  using the existing JavaOperations checker instead of adding a second runtime
  and Minecraft build dependencies.
* llama.cpp/Outlines constrain output shape. Existing JSON-schema decoding is
  retained; it cannot by itself establish that an event observes a domain rule.

Sources: https://docs.fabricmc.net/develop/events/ ;
https://github.com/palantir/javapoet ; https://github.com/javaparser/javaparser ;
https://github.com/cel-expr/cel-java ;
https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md

## Verification and limits

Replay the captured trigger/catalogue shapes with the new protocol. Exercise
numeric clock arithmetic, state changes, false gates, repeated ticks, distinct
contexts, zero-valued old observations, callback results, type mismatches,
injection and out-of-scope state. Compile and execute generated Java probes.
Prove that the real construction entrypoint selects this path and never calls
the legacy Java source writer/repairer for rules. Run relevant regressions and
the project suite, reporting failures and skips without claiming an external
9B/Fabric execution that did not happen.
