Generated Java source handoff
=============================

Production failure
------------------

The production-logs (37) run used main ``8c173bd7`` and reached final Java
attribution after constructing its callback successfully. Attribution then
reported that the generated ``Occurrence.EVENT`` owner did not exist.

The dispatcher captured source context before the constructor created the
input producer. Construction compilation explicitly included those producer
files, but final symbol attribution received only the older context. A second
defect identified Java companions by basename, although independent generated
producer packages legitimately contain the same ``Occurrence.java`` filename.

Implementation
--------------

Each construction request now owns a ``JavaSourceCapture``. After successful
full compilation and source registration checks, the constructor records the
exact companion bytes accepted for that candidate. The final dispatcher
consumes that capture using the target path and source digest, then rechecks
the materialized bytes. Initial prerequisite hashes remain constraints: a
changed initial source, changed producer or different candidate cannot refresh
away its previous proof. Outer whitespace and CRLF normalization alone do not
change compilation-unit identity.

Full compilation and attribution share bounded project-source discovery and
snapshot reading. Discovery retains explicit producers, imports and referenced
package siblings; it never interprets arbitrary dotted field expressions as
package names. An earlier candidate on disk cannot replace the current source.
Existing source count, byte, path and symlink checks remain enforced.

Companion identities retain their project-relative namespace through the JDK
Compiler Trees bridge. Final side validation checks the candidate and every
companion. Each companion is checked under its own source set, and declarations
from client sources are classified CLIENT even without a side annotation.
Common code cannot hide a client-only call behind a generated helper.

The normal catalog publisher regenerates all 43 immutable HOST contexts and
official context identifiers together. No admission state or hash check is
manually overridden.

Verification
------------

The original missing-owner failure, basename collision, dotted-field shadowing,
and hidden client-reference failure each have real-JDK regressions observed
failing before the corresponding repair. The final focused group passed all
76 tests with JDK 25 on PATH. Additional tests bind captures to both candidate
and producer bytes and distinguish client companion callers from common ones.
Ruff, compileall, diff checks and the architecture regression audit passed.

Official Minecraft 26.2 / Fabric API 0.155.2+26.2 / Loader 0.18.4 validation
compiled 114 sources containing 111 discovered common callbacks. An additional
probe used the actual 164,380-symbol Minecraft/Fabric index to construct two
independent typed UI producers, compile 11 sources, hand nine companions to
final attribution, and check 126 side-sensitive symbols. Both registered real
Fabric event invocations reached their owned probe writer in a Java harness.
The harness supplies a compiler probe actor; it is not a running Minecraft
player or server.

The complete local suite completed with 4,732 passed, seven skipped and three
failures: the two deterministic full-pipeline tests could not download Maven
dependencies in the local Java environment, and the MCP scope-cache test
failed during the parallel run. The latter is checked separately. An earlier
parallel run also hit pytest-xdist's serialization of a private setuptools
warning class; the complete retry disabled that warning-capture plugin, not
any product assertions. These results do not certify full MiMo generation,
dedicated-server gameplay or transactional filesystem replacement.
