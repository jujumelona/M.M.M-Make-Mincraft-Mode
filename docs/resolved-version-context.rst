Host version snapshot boundary
==============================

The default Fabric provider now reads whole HOST bundles. It no longer assembles
independently discovered Loader, API, Loom and Gradle versions during selection.
``VersionRequest`` accepts AUTO or an exact PINNED Minecraft coordinate. AUTO uses
the catalog's explicit preferred context; PINNED either matches exactly or fails.
The public resolver and its runtime compatibility wrapper both use this boundary.
Existing project targets remain preserved unless migration is explicit. Conflicting
new pins are rejected instead of silently replacing either target.

Host integration
----------------

The installed package includes ``data/host_version_catalog.json`` and its official
source evidence in ``data/official_version_evidence.json``. No network access or
environment setting is needed during selection. To override the packaged catalog,
set ``MMM_VERSION_BUNDLE_CATALOG`` to a HOST-managed JSON file containing exactly:

* ``schema_version``: ``mmm/host-version-catalog-v1``
* ``auto_context_id``: one of the admitted bundle context IDs
* ``bundles``: a list of ``ResolvedVersionContext.to_dict()`` results

To construct a bundle in the HOST, use a validated ``TargetContract`` with
``host_facts_json`` containing these fields:

* ``host_revision``: immutable revision identifying the HOST evidence snapshot
* ``capabilities``: named booleans, including module kinds requested by planning
* ``api_symbols``: exact symbol bindings
* ``schemas``: artifact-template ID to JSON Schema
* ``artifact_rules``: template ID to ``template_sha256``, ``required_symbols`` and
  ``requires_capabilities``
* ``dependency_coordinates``: exact Maven/build coordinates
* ``repositories``: HTTPS repository URLs
* ``replacements``: explicitly verified symbol replacement bindings

``target.version_context.to_dict()`` produces the canonical serialized bundle.
Template hashes use SHA-256 over parsed template JSON encoded with sorted keys,
compact separators and finite values. Never mark a template admitted merely because
it parses: the HOST must establish that its APIs, schemas and target rules actually
apply to the selected version. This application checks the supplied admission;
it does not manufacture compatibility evidence.

Run ``python -m minecraft_mod_ai.host_version_catalog`` with the environment setting
to audit every catalog template against the installed package. This audits bundle
structure and template identity, not Minecraft runtime compatibility. Catalog content
is a trusted HOST input. Content hashes detect drift; they are not signatures or
independent proof that the HOST facts are correct.

The bundled metadata was researched from immutable commits of Fabric's official
version-specific example branches, Fabric Maven, Mojang's release manifest and
checksummed client JARs, and Gradle distribution checksums. It covers 43 releases
from 1.14.4 through 26.2. AUTO is explicitly pinned to 26.2. The four earlier 1.14
releases are recorded with separately researched Java, pack formats and published
Yarn coordinates. They have no current official example release branch, so these
partial facts are not promoted into executable bundles. This is a metadata catalog,
not a claim that every template works on every
version. API symbols, schemas, template admissions and replacements remain absent
until independently reviewed; their use fails with ``HOST_FACT_UNAVAILABLE``.

The official examples currently request Loom 1.17-SNAPSHOT. Research resolves this
to 1.17.20 only after matching both the published binary SHA-256 and dependency list
against the official recommendation. The downloaded fixed binary is also hashed.
There is no substitution with an unrelated latest Loom release. Minecraft Java
minimums remain version-specific (8/16/17/21/25); the separate
``dependency_coordinates.gradle_jvm_major`` records the build JVM minimum, which
can be higher than Minecraft's Java minimum. Do not launch modern Gradle on Java 8
merely because an old Minecraft target uses Java 8 bytecode.

Refresh explicitly with ``python -m minecraft_mod_ai.research_version_catalog
--output minecraft_mod_ai/data`` and review the resulting diff. The command does not
run automatically at startup. Missing/invalid configured catalogs fail closed and
do not fall back to either live discovery or the packaged catalog.

Execution bindings
------------------

The context stores immutable canonical JSON; public views are copies and nested
read views are immutable. Target lock receipt hashes cover HOST facts. Serialization,
project lock persistence, research projection and target decision lowering preserve
and compare the context. User request modes are absent from the execution snapshot.

Artifact expansion checks the host-admitted template and binds jobs to context IDs.
Execution rejects conflicting host coordinates, templates changed since admission,
missing capabilities and missing symbols before materialization. JSON output uses
the HOST artifact schema. Receipts and published ports carry the same context.
Graph integration, persisted checkpoint reuse and context-aware source reuse reject
foreign contexts. Complete proposals bind module and asset identities to the lock's
context in ``_artifact_version_contexts``; the approval hash covers this manifest.

Compatibility boundaries and remaining validation
------------------------------------------------

Old serialized locks without ``host_facts_json`` remain readable for existing receipt
and migration tooling. Low-level standalone artifact APIs still accept unbound
fixtures; production complete planning passes ``spec.platform.version_context`` and
therefore rejects incomplete legacy locks. Context-aware source reuse is explicit:
callers must supply ``version_context`` and a matching project-index context ID.
There is no implicit cross-version migration rule.

This change does not certify all legacy/custom generators, every version-dependent
API in their bodies, or a repository-wide replacement of all criteria with executable
assertions. HOST admission of deterministic leaf templates prevents unreviewed leaf
changes; arbitrary custom code needs its existing compile/runtime gates and further
symbol-level verification. General model design and research are not a replacement
for HOST compatibility evidence.

Tests use explicitly synthetic HOST bundles. They establish software boundary
behavior, not Fabric/Gradle/GameTest success. The separate packaged-catalog tests
check real researched coordinates and their evidence bindings without network access.
This workspace currently has Java 17 on PATH; Gradle/GameTest and in-game checks
were not run. Complete API/schema/template admission and a matching build toolchain
are still required for a verified end-to-end generation run.
