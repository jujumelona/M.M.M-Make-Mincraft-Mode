import json
import os
import sys

from minecraft_mod_ai.runner import GradleRunner


def test_command_output_reaches_cell_before_child_finishes(monkeypatch, tmp_path):
    acknowledgment = tmp_path / "seen"

    class Cell:
        def write(self, value):
            if '"event":"gradle_command_output"' in value and "compiler detail" in value:
                acknowledgment.write_text("seen")
            return len(value)

        def flush(self):
            pass

    monkeypatch.setattr(sys, "stderr", Cell())
    monkeypatch.setenv("MMM_ROOT_CAUSE_TRACE_PATH", str(tmp_path / "trace.jsonl"))
    script = (
        "import pathlib,time,sys; print('compiler detail',flush=True); "
        f"p=pathlib.Path({str(acknowledgment)!r}); end=time.monotonic()+3; "
        "\nwhile not p.exists() and time.monotonic()<end: time.sleep(.01)"
        "\nsys.exit(0 if p.exists() else 7)"
    )
    result = GradleRunner(tmp_path, command_timeout_seconds=5)._run(
        name="build", executable=sys.executable, arguments=("-c", script),
        cwd=tmp_path, env=os.environ.copy(), log_path=tmp_path / "build.log",
    )
    assert result.exit_code == 0, "log appeared only after subprocess completion"
    assert "compiler detail" in (tmp_path / "build.log").read_text()


def test_command_timeout_keeps_output_and_original_limit(monkeypatch, tmp_path):
    monkeypatch.setenv("MMM_ROOT_CAUSE_TRACE_PATH", str(tmp_path / "trace.jsonl"))
    result = GradleRunner(tmp_path, command_timeout_seconds=1)._run(
        name="build", executable=sys.executable,
        arguments=("-c", "import time; print('dependency resolution stuck',flush=True); time.sleep(30)"),
        cwd=tmp_path, env=os.environ.copy(), log_path=tmp_path / "build.log",
    )
    assert result.timed_out
    assert result.exit_code == 124
    assert "dependency resolution stuck" in (tmp_path / "build.log").read_text()
    events = [json.loads(row) for row in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert any(row["event"] == "gradle_command_result" and row["result"] == "TIMEOUT" for row in events)
