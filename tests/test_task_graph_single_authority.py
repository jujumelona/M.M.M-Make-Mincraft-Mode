from __future__ import annotations

from minecraft_mod_ai import task_artifact_contract as contract


def _task(task_id: str, requirement: str, *, provides: list[str]) -> dict[str, object]:
    return {
        "task_id": task_id,
        "requirement_refs": [requirement],
        "consumes": ["target:frozen"],
        "provides": provides,
        "depends_on": [],
        "task_sha256": "",
    }


def test_requirement_causality_is_projected_into_consumes_before_dag_binding() -> None:
    catalog = {
        "requirements": [
            {
                "requirement_id": "req_resource",
                "depends_on": [],
                "provides": ["capability:resource_gathering"],
            },
            {
                "requirement_id": "req_currency",
                "depends_on": ["req_resource"],
                "provides": ["capability:currency_accumulation"],
            },
        ]
    }
    contract._capture_requirement_graph(catalog)

    bound = contract._project_requirement_dataflow(
        (
            _task(
                "task_resource",
                "req_resource",
                provides=["capability:resource_gathering"],
            ),
            _task(
                "task_currency",
                "req_currency",
                provides=["capability:currency_accumulation"],
            ),
        )
    )
    by_id = {task["task_id"]: task for task in bound}

    assert by_id["task_currency"]["consumes"] == [
        "target:frozen",
        "capability:resource_gathering",
    ]
    assert by_id["task_currency"]["depends_on"] == ["task_resource"]
    assert by_id["task_currency"]["dependency_reasons"] == {
        "task_resource": {
            "kind": "requirement_dataflow",
            "requirement_ref": "req_resource",
        }
    }


def test_retained_parent_is_root_dataflow_not_phantom_task_edge() -> None:
    catalog = {
        "requirements": [
            {
                "requirement_id": "req_resource",
                "depends_on": [],
                "provides": ["capability:resource_gathering"],
            },
            {
                "requirement_id": "req_currency",
                "depends_on": ["req_resource"],
                "provides": ["capability:currency_accumulation"],
            },
        ]
    }
    contract._capture_requirement_graph(catalog)

    bound = contract._project_requirement_dataflow(
        (
            _task(
                "task_currency",
                "req_currency",
                provides=["capability:currency_accumulation"],
            ),
        )
    )

    assert bound[0]["consumes"] == [
        "target:frozen",
        "capability:resource_gathering",
    ]
    assert bound[0]["depends_on"] == []
    assert bound[0]["dependency_reasons"] == {}
