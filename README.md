# M.M.M Make Mincraft Mode

[한국어](README_KO.md)

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jujumelona/M.M.M-Make-Mincraft-Mode/blob/main/M.M.M_Make_Mincraft_Mode_Colab.ipynb)

M.M.M turns a natural-language request into a Minecraft Java 1.20.1 Fabric game plan, lets you revise that plan in conversation, and then builds the project. It does not add a boss, arena, village, field, or dungeon unless your request needs one.

For each request, the central agent reclassifies the required systems, code, libraries, images, 3D, animation, audio, licensing, and test evidence, then searches exact-version RAG and compatible open-source/media candidates. It does not install or copy a search result before its origin license and immutable file hash are verified.

If the request needs AI or voice, M.M.M dynamically searches current runtimes and model catalogs instead of fixing one product in the engine. It separates inference, ASR, VAD, TTS, translation, transport, and optional voice adaptation, then checks the exact Minecraft/Fabric/Java boundary, immutable model revision, code/model/data licenses, hardware and latency measurements, privacy, and fallback. The approved plan adds one bounded, token-authenticated asynchronous localhost bridge for the requested executable capabilities; it does not bundle a model before its gates pass or let model output mutate the world directly. Voice adaptation stays disabled unless the speaker is authorized and explicit consent, provenance, revocation, and deletion all pass.

## Google Colab

1. Click the Colab badge and choose a GPU runtime.
2. Enter `PROMPT`, then run the cells in order.
3. Use the optional revision cell to change the plan in plain language.
4. Run **Build this plan**, then download the result.

The notebook does not require an engine ZIP. On every setup run it clones or fast-forwards GitHub `main` and prints the exact commit it used.

- New mod: leave `PATCH_EXISTING=False`. Nothing is uploaded.
- Modify an existing mod: set `PATCH_EXISTING=True`, then upload one source/release ZIP that you own or may modify. It must contain source code and a Gradle project.
- A JAR by itself can be inspected, but it is not presented as editable source.

Google Drive storage is enabled by default, so rerunning the same `RUN_NAME` resumes completed work instead of failing because the folder already exists.

## Local or remote models

`MODEL_PROFILE="t4_local"` runs the model roles on the Colab GPU. Set it to `remote_quality` and fill in the HTTPS API address and model fields to use OpenAI-compatible remote endpoints; the notebook asks for the API key without saving it in the notebook.

For local Python:

```python
from minecraft_mod_ai import CompleteModAISession

session = CompleteModAISession(output_root="mmm-output")
plan = session.plan("Make a seasonal farming and cooking mod.")
print(plan.message)
plan = session.revise("Remove combat and add a winter greenhouse.")
result = session.build(plan, source_only=True)
print(result.release_zip)
```

## Codex plugin

The optional plugin bundle is in
[`plugins/mmm-minecraft-mod-ai`](plugins/mmm-minecraft-mod-ai). It packages the
conversational entry skill and the stage-specific M.M.M MCP server configuration.
Colab and Python usage do not require installing this plugin.

## Scale

There is no fixed product-wide cap on feature count, module count, or total world scope. Large plans become paged modules, bounded artifact shards, and resumable checkpointed work instead of one oversized prompt or file.

This does not mean infinite hardware. Minecraft/Java formats, GPU and RAM, disk space, model APIs, and Colab session quotas still impose real limits. M.M.M keeps those as per-task safety/resource boundaries and continues the project through additional shards and sessions.

## License

[MIT](LICENSE)
