from __future__ import annotations
from typing import Any
from minecraft_mod_ai.mcp_tools import MMMToolService

class _DiscoveryStub:

    def __init__(self) -> None:
        self.search_call: dict[str, Any] = {}

    def search(self, provider: str, query: str, *, cursor: str, limit: int, target_profile: str) -> dict[str, Any]:
        self.search_call = {'provider': provider, 'query': query, 'cursor': cursor, 'limit': limit, 'target_profile': target_profile}
        return {'schema_version': 'test/page-v1', **self.search_call}

    def inspect_huggingface_model(self, repo_id: str) -> dict[str, Any]:
        return {'schema_version': 'test/hf-inspection-v1', 'repo_id': repo_id, 'revision_sha': 'a' * 40, 'download_performed': False}
