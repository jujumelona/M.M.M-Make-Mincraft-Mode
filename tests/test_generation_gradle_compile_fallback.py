from __future__ import annotations

from minecraft_mod_ai.generation_gradle_compile_fallback import (
    CompileVerificationReport,
    has_javac_source_diagnostics,
)


def test_javac_file_line_error_is_source_diagnostic() -> None:
    log = """
> Task :compileJava FAILED
/home/runner/project/src/main/java/demo/Example.java:17: error: cannot find symbol
        MissingThing value;
        ^
  symbol:   class MissingThing
1 error
"""

    assert has_javac_source_diagnostics(log) is True


def test_dependency_resolution_failure_is_not_source_diagnostic() -> None:
    log = """
> Task :compileJava FAILED
Could not resolve all files for configuration ':compileClasspath'.
> Could not resolve net.fabricmc:fabric-loader:0.16.0.
  > Could not GET 'https://maven.example.invalid/fabric-loader.pom'.
     > Connect timed out
"""

    assert has_javac_source_diagnostics(log) is False


def test_generic_gradle_failure_is_not_enough_to_authorize_source_repair() -> None:
    log = """
FAILURE: Build failed with an exception.
* What went wrong:
Execution failed for task ':compileJava'.
> Process 'Gradle Worker Daemon 1' finished with non-zero exit value 1
"""

    assert has_javac_source_diagnostics(log) is False


def test_compile_report_availability_is_explicit() -> None:
    source_fail = CompileVerificationReport(
        status="FAIL",
        gradle_version="9.1.0",
        commands=(),
        error="javac source errors",
        source_diagnostics=True,
    )
    infra_fail = CompileVerificationReport(
        status="UNAVAILABLE",
        gradle_version="9.1.0",
        commands=(),
        error="dependency resolution failed",
    )

    assert source_fail.passed is False
    assert source_fail.available is True
    assert infra_fail.passed is False
    assert infra_fail.available is False
    assert source_fail.to_dict()["verification_scope"] == "compileJava"
