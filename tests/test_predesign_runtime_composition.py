from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[1]


_FRESH_RUNTIME_PROBE = r"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

os.environ["MMM_CENTRAL_AI_WORKERS"] = "4"
os.environ["MMM_LLAMA_ACTIVE_PARALLEL"] = "4"
os.environ["MMM_LLAMA_RUNTIME_RECEIPT"] = json.dumps(
    {"schema_version": "mmm/llama-runtime-receipt-v1", "slots": 4},
    separators=(",", ":"),
)
probe_root = Path(os.environ["MMM_TEST_PROBE_ROOT"]).resolve()
os.environ["MMM_RESEARCH_CHECKPOINT_ROOT"] = str(probe_root / "checkpoints")
os.environ["MMM_RESEARCH_DOCUMENT_DIR"] = str(probe_root / "documents")

# Importing any package child in this fresh interpreter executes the real package
# bootstrap first. The assertions below therefore inspect the production install order,
# not a hand-built facsimile of the wrappers.
import minecraft_mod_ai.agentic_pre_design_rag as pre_design
import minecraft_mod_ai.agentic_research_game_design as agentic


def wrapper_chain(function):
    result = []
    seen = set()
    current = function
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        result.append(current)
        current = getattr(current, "__wrapped__", None)
    return result


chain = wrapper_chain(agentic.collect_pre_design_research)
forced_owner_indexes = [
    index
    for index, function in enumerate(chain)
    if function.__dict__.get(pre_design._MARKER) is function
]
parallel_owner_indexes = [
    index
    for index, function in enumerate(chain)
    if function.__dict__.get("_mmm_parallel_research_design_core_v1") is function
]
assert forced_owner_indexes, "the final forced-RAG wrapper has no identity owner"
assert parallel_owner_indexes, "the central parallel collector is missing"
outer_forced_index = forced_owner_indexes[0]
parallel_index = parallel_owner_indexes[0]
assert outer_forced_index < parallel_index, (
    "forced RAG must wrap central fan-out after the real bootstrap order: "
    f"forced={forced_owner_indexes}, parallel={parallel_owner_indexes}"
)
assert pre_design._effective_forced_collect_owner(
    agentic.collect_pre_design_research
), "the live outer forced-RAG owner is not effective"


lock = threading.Lock()
forced_calls = 0
worker_contexts = {}
worker_threads = {}
materialized_receipts = {}
model_calls = {}


def query_sha256(query: str) -> str:
    return "sha256:" + hashlib.sha256(query.encode("utf-8")).hexdigest()


def forced_bundle(_router, brief):
    global forced_calls
    with lock:
        forced_calls += 1
    return {
        "schema_version": "mmm/test-forced-project-rag-v1",
        "domains": [
            {
                "domain_id": domain["domain_id"],
                "queries": [
                    {"query": query, "query_sha256": query_sha256(query)}
                    for query in domain.get("queries", [])
                ],
            }
            for domain in brief.get("domains", [])
            if isinstance(domain, dict)
        ],
    }


pre_design._forced_rag_bundle = forced_bundle
agentic.retrieve_domain_evidence = lambda _brief: {"status": "ok"}
agentic.collect_technology_radar = lambda *_args, **_kwargs: {"status": "ok"}
agentic.collect_ecosystem_seed_bundle = lambda *_args, **_kwargs: {"status": "ok"}


def tiny_document(domain_id, evidence):
    receipt = evidence.get("forced_project_rag")
    assert isinstance(receipt, dict), f"{domain_id} lost its forced-RAG receipt"
    assert receipt.get("domain_id") == domain_id
    with lock:
        materialized_receipts[domain_id] = receipt

    raw_text = json.dumps(
        dict(evidence), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    document_sha256 = "sha256:" + hashlib.sha256(
        raw_text.encode("utf-8")
    ).hexdigest()
    directory = probe_root / "documents" / document_sha256.removeprefix("sha256:")
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / f"{domain_id}.json"
    pages_path = directory / f"{domain_id}.pages.jsonl"
    content = json.dumps({"domain_id": domain_id}, separators=(",", ":"))
    page = {
        "schema_version": pre_design._EVIDENCE_PAGE_SCHEMA,
        "domain_id": domain_id,
        "unit_id": "tiny-runtime-composition-probe",
        "part_index": 0,
        "part_count": 1,
        "content": content,
        "page_index": 0,
        "page_count": 1,
        "page_ref": f"{document_sha256}#page=1/1",
    }
    raw_path.write_text(raw_text, encoding="utf-8")
    pages_path.write_text(
        json.dumps(page, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": pre_design._EVIDENCE_DOCUMENT_SCHEMA,
        "domain_id": domain_id,
        "document_sha256": document_sha256,
        "raw_path": str(raw_path),
        "pages_path": str(pages_path),
        "page_count": 1,
        "page_chars": pre_design._EVIDENCE_PAGE_CHARS,
        "source_keys": sorted(str(key) for key in evidence),
    }


pre_design._materialize_domain_evidence_document = tiny_document

# Keep the fully installed worker path, adding only a probe around it. The central
# collector resolves this attribute inside each copied Context, so every domain must
# observe the request-local forced bundle even though it runs on an executor thread.
installed_worker = agentic._research_domain_with_agent


def observed_worker(
    router,
    *,
    prompt,
    domain,
    deterministic,
    trace_metadata,
):
    domain_id = domain["domain_id"]
    context_receipt = pre_design._FORCED_RAG_CONTEXT.get()
    with lock:
        worker_contexts[domain_id] = context_receipt
        names = worker_threads.setdefault(domain_id, [])
        names.append(threading.current_thread().name)
        attempt = len(names)
    if (
        domain_id == "mk_item_block"
        and attempt == 1
        and threading.current_thread().name.startswith("mmm_research_domain")
    ):
        raise RuntimeError("shared local router rejected concurrent request")
    return installed_worker(
        router,
        prompt=prompt,
        domain=domain,
        deterministic=deterministic,
        trace_metadata=trace_metadata,
    )


agentic._research_domain_with_agent = observed_worker


class ProbeRouter:
    profile = "fresh-runtime-composition"

    def __init__(self):
        config = SimpleNamespace(
            exclusive_gpu=True,
            provider="local",
            adapter="llama_cpp",
            model_id="probe-model",
            quantization="gguf",
            max_context=0,
            max_new_tokens=2048,
        )
        self.registry = SimpleNamespace(role=lambda _profile, _role: config)

    def generate_text(self, _role, messages, **_kwargs):
        user_message = next(
            message for message in reversed(messages) if message.get("role") == "user"
        )
        payload = json.loads(user_message["content"])
        domain_id = payload["domain"]["domain_id"]
        with lock:
            call = model_calls.get(domain_id, 0) + 1
            model_calls[domain_id] = call
        return json.dumps(
            {
                "research_note": {
                    "domain_id": domain_id,
                    "claims": [
                        {
                            "claim": f"validated {domain_id}",
                            "evidence_refs": ["probe:receipt"],
                        }
                    ],
                    "gaps": [],
                    "next_queries": [],
                    "sufficient": True,
                }
            }
        )


result = agentic.collect_pre_design_research(
    ProbeRouter(),
    "Add one custom item.",
)
coverage = result["minecraft_knowledge_route_coverage"]
coverage_by_domain = {
    item["domain_id"]: item["status"] for item in coverage["domains"]
}
notes = {item["domain_id"]: item for item in result["domain_notes"]}
brief_domain_ids = {
    item["domain_id"]
    for item in result["research_brief"]["domains"]
    if isinstance(item, dict)
}

assert forced_calls == 1, f"forced RAG executed {forced_calls} times"
assert set(worker_contexts) == brief_domain_ids
assert set(materialized_receipts) == brief_domain_ids
assert all(isinstance(value, dict) for value in worker_contexts.values())
assert all(
    names and names[0].startswith("mmm_research_domain")
    for names in worker_threads.values()
)
assert worker_threads["mk_item_block"] == [
    worker_threads["mk_item_block"][0],
    "MainThread",
]
assert all(
    len(names) == (2 if domain_id == "mk_item_block" else 1)
    for domain_id, names in worker_threads.items()
)
assert coverage["status"] == "PASS"
assert len(coverage_by_domain) == 7
assert all(status != "MISSING_FORCED_RAG_RECEIPT" for status in coverage_by_domain.values())
assert all(status == "ROUTES_EXECUTED" for status in coverage_by_domain.values())
assert model_calls["mk_item_block"] == 1
recovered_note = notes["mk_item_block"]
assert recovered_note["sufficient"] is True
assert recovered_note.get("worker_error") is not True
assert all(notes[domain_id]["sufficient"] is True for domain_id in coverage_by_domain)

print(
    "__MMM_RESULT__="
    + json.dumps(
        {
            "forced_calls": forced_calls,
            "forced_owner_index": outer_forced_index,
            "parallel_owner_index": parallel_index,
            "worker_domains": sorted(worker_contexts),
            "worker_threads": worker_threads,
            "coverage": coverage_by_domain,
            "recovery_calls": len(worker_threads["mk_item_block"]),
            "recovered_sufficient": recovered_note["sufficient"],
        },
        sort_keys=True,
    )
)
"""


def test_fresh_runtime_bootstrap_preserves_forced_rag_across_parallel_recovery(
    tmp_path: Path,
) -> None:
    env = dict(os.environ)
    env["MMM_TEST_PROBE_ROOT"] = str(tmp_path)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_FRESH_RUNTIME_PROBE)],
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, (
        "fresh runtime composition probe failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    result_line = next(
        (
            line.removeprefix("__MMM_RESULT__=")
            for line in completed.stdout.splitlines()
            if line.startswith("__MMM_RESULT__=")
        ),
        None,
    )
    assert result_line is not None, completed.stdout
    result = json.loads(result_line)
    assert result["forced_calls"] == 1
    assert result["forced_owner_index"] < result["parallel_owner_index"]
    assert len(result["coverage"]) == 7
    assert result["recovery_calls"] == 2
    assert result["worker_threads"]["mk_item_block"][-1] == "MainThread"
    assert result["recovered_sufficient"] is True
