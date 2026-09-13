# Authoritative CI Design

Make `main-ci.yml` the fail-closed correctness gate for generic-runner CI. Every workflow must be explicitly classified, every required main-CI job must be included in the final gate, and environment-specific integrations must be explicitly separated from generic-runner correctness.