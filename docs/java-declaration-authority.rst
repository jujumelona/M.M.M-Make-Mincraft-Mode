Java declaration authority
==========================

Failure boundary
----------------

Production-logs (38), running main ``4a63835``, passed construction and source
handoff, then failed ``MEMBER_NOT_IN_INSPECTED_CLASSPATH`` in the final symbol
resolver. Its actual 117-entry compile classpath contains the generated
project's ``build/classes/java/main`` output alongside Minecraft/Fabric jars.

The constructor already reconciled those old project classes with current
materialized sources. The final resolver did not: after javac selected a
current source declaration, it compared that member against the older class
file and replaced current source-side annotations with the old binary side.

The uploaded error omitted the member identity. An exact-version Fabric probe
reproduced the same failed boundary using an old project ``AuthoredProgram``
class and its current state-writer source. Full javac compilation passed;
the old final resolver then rejected
``proof/AuthoredProgram#writeState_abi_probe_count(I)V``. This is a reconstruction
of the validator defect, rather than a complete replay of the live model run.

Authority model
---------------

The Compiler Trees bridge reports the actual declaring source and reference
location for each selected symbol. That source identity must belong to an
explicit compiler input; implicit discovery of unrecorded classpath Java
sources is disabled.

For a referenced source declaration, a matching binary owner may be superseded
only when every binary location is that same project's real build/classes
output. Construction and final attribution share this ownership rule. JAR
files, another project's classes, and build symlinks escaping that output
remain protected. The current source supplies its own member ABI and side.
Binary declarations retain exact descriptor, kind, static-status and side
checks. Membership failures include the owner, descriptor and reference
location.

Public javac and javaparser compatibility functions accept keyword-only
``project_root``; the existing jdtls compatibility function forwards its
workspace root. Callers recompiling their own existing output must provide
that ownership context. Unknown-project source/binary collisions remain strict.

The scanner still emits references rather than an independent inventory of
every class declaration. An entirely unreferenced owner collision is an older
limitation outside this membership repair.

Verification
------------

Real javac tests cover old primary/companion ABIs, both directions of side
annotation drift, removed methods, immutable external JARs, another project's
output, a JAR stored beneath build/classes, escaped build symlinks, unrecorded
classpath sources, diagnostic identity, and both public compatibility routes.
The final focused group passed 78 tests. Ruff, compileall, diff checks and the
architecture regression audit passed without gate changes.

Minecraft 26.2 and Fabric API 0.159.0+26.2 compiled 115 Java sources containing
112 common callback registrations. The stale-output probe additionally used
Loader 0.19.5, the same loader as the uploaded run: 164,408 Minecraft/Fabric
symbols, two generated producers, 11 compiled sources, nine final companions,
126 checked symbols and two successful real Fabric event invocations. Its
actor is a compiler probe, not a running Minecraft player.

The normal publisher regenerated all 43 immutable HOST contexts and version
evidence together. A second publication into an independent temporary folder
produced byte-identical files. Only code-authority hashes and context IDs
changed in bundled metadata.

The final complete local run passed 4,746 tests with seven skipped and three
existing failures: the parallel MCP scope-cache test and two deterministic
E2E tests blocked by Java Maven network access. The MCP file, checked with
catalog/context regressions separately, passed in a 41-test group. Hosted CI results are reported with the commit.
Live MiMo generation and Minecraft gameplay remain unverified.
