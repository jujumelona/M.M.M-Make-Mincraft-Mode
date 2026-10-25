# Execution ownership: novel Minecraft gameplay (2026-10-26)

The original MiMo-V2.6 9B profile is retained for requirements/planning and
host-proven typed rules. It **does not implement arbitrary new runtime mechanics**.
Java source which requires missing event producers, new Minecraft API calls,
or native inventory/world mutations now has an explicit implementation owner.

## Behavior

1. Host inspects the target 26.2 symbol index, Fabric callbacks, and immutable behavior.
2. Verified host rule / entity emitters can still produce source without a second AI.
3. When the feature requires `source_owned_java`, the compiler **requires**
   the independently configured `feature_coder` role before any source/model calls.
4. Only that role authors the inspected method selection, producer/UI ABI, bounded
   method bodies and compiler diagnostics. A JournaledConstructionRouter records
   its distinct role, not MiMo's original `planner`/`coder` label.
5. The host remains the only authority for source boundaries, API evidence,
   runtime mutation receipts, final Java compilation and required GameTest.
6. Missing implementation provider is a hard error. There is NO silent small-model
   fallback, no "unavailable" recovery, no guessed Fabric callback, and no
   success claim based on syntax compilation.

## External implementation worker configuration

Provide a supported OpenAI-compatible code model endpoint at runtime (for
example via a Colab secret-to-environment bridge). This worker is explicitly
separate from the model chosen for the rest of the pipeline:

```bash
export MMM_FEATURE_CODER_MODEL='your-capable-code-model'
export MMM_FEATURE_CODER_BASE_URL='https://your-openai-compatible-host/v1'
export MMM_FEATURE_CODER_API_KEY='your-secret'
```

**Do not store API keys in GitHub or in notebooks.** All three variables must
be present together. The provider's availability, cost, model adequacy and
runtime credentials are not supplied by this repository. It is not a free
or locally hosted solution by default.

Changing the external model/endpoint invalidates host-source checkpoints and
journal bindings. Credentials are not included in checkpoint fingerprints.

## Proof boundary

The source-owned path does **not** establish that every possible Minecraft mod
can be generated. Even a distinct code worker can fail to implement novel
features correctly. Production acceptance still requires target compilation and
behavior-specific in-game verification. Gate results must never be represented
as gameplay proof when runtime tests have not run.

The next host-native extension is to turn canonical semantic operations into
version-bound transactional game primitives, independent of free-form Java
generation. That is separate from enabling the external coder.
