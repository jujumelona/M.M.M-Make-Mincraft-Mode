from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"{label} target not found")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    work = Path("minecraft_mod_ai/work_graph.py")
    replace_once(
        work,
        """        if stage == 'content':
            # Deterministic extended-content aggregation is governed by the explicit
            # ScalePolicy. Environment-tuned pipeline widths belong to model-owned
            # stages; allowing them here can silently explode one bounded aggregate
            # into one work node per deterministic content module.
            return max(1, int(policy.java_shard_size))
""",
        """        if stage == 'content':
            return _pipeline_shard_size(
                'MMM_CONTENT_PIPELINE_SHARD_SIZE',
                1,
                max(1, int(policy.java_shard_size)),
            )
""",
        "content shard",
    )
    replace_once(
        work,
        """        if stage == 'entity':
            return _pipeline_shard_size(
                'MMM_ENTITY_PIPELINE_SHARD_SIZE',
                2,
                max(1, int(policy.entity_shard_size)),
            )
""",
        """        if stage == 'entity':
            return _pipeline_shard_size(
                'MMM_ENTITY_PIPELINE_SHARD_SIZE',
                1,
                max(1, int(policy.entity_shard_size)),
            )
""",
        "entity shard",
    )

    orch = Path("minecraft_mod_ai/complete_orchestrator.py")
    replace_once(
        orch,
        """        blockbench_receipts: list[dict[str, Any]] = []
        unresolved: list[str] = []
        asset_shards: list[dict[str, Any]] = []
        runtime_init_lock = threading.RLock()
""",
        """        blockbench_receipts: list[dict[str, Any]] = []
        unresolved: list[str] = []
        asset_shards: list[dict[str, Any]] = []
        review_futures: list[tuple[str, Future[dict[str, Any]]]] = []
        review_futures_lock = threading.Lock()
        runtime_init_lock = threading.RLock()
""",
        "review state",
    )
    replace_once(
        orch,
        """                        if options.run_blockbench and (not options.source_only):
                            blockbench_receipts.append(run_named_checkpoint(ledger, f'blockbench-review-{module.module_id}', stage='validate:blockbench', input_value={'graph_hash': work_plan.graph_hash, 'entity_receipt': entity_receipt}, action=lambda receipt=entity_receipt: self._blockbench_review(receipt, run_root), encode=lambda value: value, decode=lambda cached: cached, validate_cached=lambda cached: Path(str(cached.get('preview', ''))).is_file()))
                        elif options.run_blockbench:
""",
        """                        if options.run_blockbench and (not options.source_only):
                            review_future = review_pool.submit(
                                run_named_checkpoint,
                                ledger,
                                f'blockbench-review-{module.module_id}',
                                stage='validate:blockbench',
                                input_value={'graph_hash': work_plan.graph_hash, 'entity_receipt': entity_receipt},
                                action=lambda receipt=entity_receipt: self._blockbench_review(receipt, run_root),
                                encode=lambda value: value,
                                decode=lambda cached: cached,
                                validate_cached=lambda cached: Path(str(cached.get('preview', ''))).is_file(),
                            )
                            with review_futures_lock:
                                review_futures.append((module.module_id, review_future))
                        elif options.run_blockbench:
""",
        "inline blockbench",
    )
    replace_once(
        orch,
        """        commit_pool = ThreadPoolExecutor(max_workers=max(1, int(capacities['commit'])), thread_name_prefix='commit')
        node_futures: dict[str, Future[Any]] = {}
""",
        """        commit_pool = ThreadPoolExecutor(max_workers=max(1, int(capacities['commit'])), thread_name_prefix='commit')
        review_workers_raw = os.environ.get('MMM_BLOCKBENCH_REVIEW_WORKERS', '').strip()
        try:
            review_workers = int(review_workers_raw) if review_workers_raw else max(1, min(4, int(capacities['cpu_io'])))
        except ValueError:
            review_workers = max(1, min(4, int(capacities['cpu_io'])))
        review_pool = ThreadPoolExecutor(max_workers=max(1, review_workers), thread_name_prefix='blockbench_review')
        node_futures: dict[str, Future[Any]] = {}
""",
        "review pool",
    )
    replace_once(
        orch,
        """                break
        finally:
            cpu_pool.shutdown(wait=True, cancel_futures=True)
            llm_pool.shutdown(wait=True, cancel_futures=True)
            image_pool.shutdown(wait=True, cancel_futures=True)
            commit_pool.shutdown(wait=True, cancel_futures=True)
        asset_receipt = {'schema_version': 'mmm/complete-assets-sharded-v1', 'status': 'GENERATED', 'shard_count': len(asset_shards), 'asset_count': sum(len(item.get('assets', [])) for item in asset_shards), 'shards': asset_shards} if asset_shards else None
""",
        """                break
            review_results = [(module_id, future.result()) for module_id, future in tuple(review_futures)]
            blockbench_receipts.extend(receipt for _, receipt in sorted(review_results, key=lambda item: item[0]))
        finally:
            cpu_pool.shutdown(wait=True, cancel_futures=True)
            llm_pool.shutdown(wait=True, cancel_futures=True)
            image_pool.shutdown(wait=True, cancel_futures=True)
            commit_pool.shutdown(wait=True, cancel_futures=True)
            review_pool.shutdown(wait=True, cancel_futures=True)
        asset_receipt = {'schema_version': 'mmm/complete-assets-sharded-v1', 'status': 'GENERATED', 'shard_count': len(asset_shards), 'asset_count': sum(len(item.get('assets', [])) for item in asset_shards), 'shards': asset_shards} if asset_shards else None
""",
        "review join",
    )

    test = Path("tests/test_pipeline_parallel_bottlenecks.py")
    text = test.read_text(encoding="utf-8")
    if "from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator\n" not in text:
        text = text.replace(
            "from minecraft_mod_ai.complete_spec import ProductionModule\n",
            "from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator\nfrom minecraft_mod_ai.complete_spec import ProductionModule\n",
            1,
        )
    if "test_blockbench_review_uses_dedicated_parallel_lane" not in text:
        text += """


def test_blockbench_review_uses_dedicated_parallel_lane():
    source = inspect.getsource(CompleteProductionOrchestrator._execute_generation_work)
    assert "thread_name_prefix='blockbench_review'" in source
    assert 'review_pool.submit(' in source
    assert 'blockbench_receipts.append(run_named_checkpoint' not in source
    assert 'MMM_BLOCKBENCH_REVIEW_WORKERS' in source
"""
    test.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
