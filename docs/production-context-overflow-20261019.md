# File-author context overflow at 0ae8317

## Observed failure

The second production log archive targets Minecraft 26.2 / Fabric / Java 25.
Classpath inspection passed. The first source-author request then failed at the
live llama.cpp context guard: **49,422 input tokens exceeded the 32,768-token
runtime slot**. The archive contains 331 planner requests and no coder request;
this rejection happened before a coder completion journal was opened.

The API evidence change at 0ae8317 supplied useful inspected members but also
copied full bytecode-index records and large member-name catalogs into the
prompt. Compiler repairs appended complete previous source and API views on
every attempt, creating a second growth path.

## Changes pushed to main

- `aa7c4db`: project symbol rows to Java contracts and include overload counts
  only for displayed names. Index hashes, descriptors, debug receipts and
  redundant IDs remain in the authoritative host index.
- `9b0dea4`: preserve Java access/modifier flags, keep one durable inspected API
  view, and replace the failed-source message on each compiler repair. Merged
  API contracts survive successive repairs without duplicating entire views.
- Recover typed context pressure by reducing supplemental API observations.
  Preserve complete requirements, target, source, compiler diagnostic, referenced
  overloads, constructors and factories. Every retry must reduce serialized
  bytes; the unchanged exact live-token guard decides whether it fits.
- Context preflight rejections do not consume compiler repair attempts.
  Exhausted canonical recovery, unrelated backend failures and a prompt with no
  further safe size reduction propagate their original errors.
- Save rejected file-author prompts and size receipts under the durable trace
  directory, including failures that occur before a model completion journal.

## Verification and limits

Regression tests first reproduced metadata/catalog growth, context recovery,
repair-history growth, contracts lost on the second repair, modifier loss and
generic factories incorrectly removed during compaction. Those regressions pass.

The construction suite plus existing exact-context, finish-reason and context
policy/safety tests completed with **156 passed, 10 skipped**. The skipped tests
require a local JDK. Focused file-author/API tests completed with **48 passed,
1 skipped**. Ruff checks for F/E7/E9 and compilation checks passed.

Independent review confirmed both Important findings were fixed and reported no
remaining Important or Critical findings. CI for the first code commit passed
Direct Implementation Migration, Cross-stage Contract ABI and Partial CI. CI for
`9b0dea4` was queued or running when inspected; the full remote suite is not
reported as passed.

The rejected production prompt is absent from the original archive, so its exact
post-fix token count cannot be measured here. No live MiMo completion, Minecraft
26.2 build, GameTest or gameplay success is claimed. A production rerun on the
updated main branch is still needed to establish those results.
