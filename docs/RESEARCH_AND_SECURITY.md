# Research, RAG, and MCP boundary

This repository implements a small, verified Fabric 1.20.1 slice of the larger
architecture PDF. It does not claim to be a general autonomous Minecraft
developer or a production MCP server.

## Decisions implemented in code

- `PlatformLock` pins Minecraft, Java, Yarn, Loader, Fabric API, Loom, and
  Gradle. "Latest" means the latest engine commit from this repository, not an
  unreviewed change to the Minecraft compatibility target.
- Technical evidence is selected from a code-owned, HTTPS-only official-source
  catalog. Retrieval results are data, never executable instructions.
- The ordered evidence snapshot, the fixed tool-capability manifest, and an
  optional imported-project snapshot are bound into the proposal approval hash.
  A change to any of them invalidates the approval.
- The capability manifest uses MCP 2025-11-25 risk vocabulary and strict JSON
  schemas as an interoperability design aid. The local broker remains the
  authority; annotations are not permissions.
- Research/plan operations cannot write files. The execution broker accepts only
  a validated, approved `Proposal` and a closed action enum.
- Existing-mod ZIP files are inspected without executing Gradle wrappers,
  scripts, JARs, or other archive content. JAR-only inputs are inventory inputs,
  not editable source.
- RAG does not replace compilation, Fabric GameTest, JAR validation, or human
  approval.

## Primary sources

Minecraft and Fabric:

- [Fabric project development](https://docs.fabricmc.net/develop/)
- [Fabric project structure](https://docs.fabricmc.net/develop/getting-started/project-structure)
- [Fabric build documentation](https://docs.fabricmc.net/develop/getting-started/building-a-mod)
- [Fabric data generation](https://docs.fabricmc.net/develop/data-generation/setup)
- [Fabric automated testing](https://docs.fabricmc.net/develop/automatic-testing)
- [fabric.mod.json specification](https://docs.fabricmc.net/develop/loader/fabric-mod-json)
- [Fabric Meta API](https://meta.fabricmc.net/)
- [Fabric API official Maven repository](https://maven.fabricmc.net/net/fabricmc/fabric-api/fabric-api/)
- [Yarn 1.20.1+build.1 Javadoc](https://maven.fabricmc.net/docs/yarn-1.20.1%2Bbuild.1/)

MCP:

- [MCP versioning](https://modelcontextprotocol.io/docs/learn/versioning)
- [MCP 2025-11-25 tools](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)
- [MCP lifecycle and capability negotiation](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle)
- [MCP security best practices](https://modelcontextprotocol.io/docs/tutorials/security/security_best_practices)

RAG and prompt-injection research:

- [Retrieval-Augmented Generation, NeurIPS 2020](https://papers.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)
- [ALCE citation evaluation, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.398/)
- [AgentDojo, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract-Datasets_and_Benchmarks_Track.html)
- [PoisonedRAG, USENIX Security 2025](https://www.usenix.org/conference/usenixsecurity25/presentation/zou-poisonedrag)
- [StruQ, USENIX Security 2025](https://www.usenix.org/conference/usenixsecurity25/presentation/chen-sizhe)

## Next implementation gates

The following PDF goals remain future work and must not be advertised as
complete:

- Fabric Datagen providers and repeat-run output-hash stability.
- Structure NBT generation plus rotated arena GameTests.
- A stable `ModelIR` and Blockbench codec round-trip instead of treating
  `.bbmodel` as the long-term canonical format.
- Minimal unified-diff repair of arbitrary imported source projects with
  invariant preservation and full old-plus-new regression suites.
- A real MCP server transport, protocol negotiation, resource authorization,
  OAuth, and conformance tests.
