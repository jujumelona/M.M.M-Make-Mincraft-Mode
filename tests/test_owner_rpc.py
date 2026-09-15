import sys

import pytest


def test_rpc_keeps_process_and_correlates_responses(tmp_path):
    from minecraft_mod_ai.owner_rpc import OwnerRPC

    server = tmp_path / 'server.py'
    server.write_text('''import json, os, sys
for line in sys.stdin:
    request = json.loads(line)
    if request['method'] == 'close': break
    print(json.dumps({'id': request['id'], 'result': {'pid': os.getpid(), 'value': request['params']['value']}}), flush=True)
''')
    with OwnerRPC([sys.executable, '-u', str(server)]) as rpc:
        first = rpc.request('echo', {'value': 1}, timeout=5)
        second = rpc.request('echo', {'value': 2}, timeout=5)
        assert first['pid'] == second['pid']
        assert [first['value'], second['value']] == [1, 2]


def test_rpc_process_exit_wakes_pending_request(tmp_path):
    from minecraft_mod_ai.owner_rpc import OwnerRPC, OwnerRPCError

    server = tmp_path / 'exit.py'
    server.write_text("import sys\nsys.stdin.readline()\nprint('real failure', file=sys.stderr, flush=True)\nsys.exit(7)\n")
    with OwnerRPC([sys.executable, '-u', str(server)]) as rpc, pytest.raises(OwnerRPCError, match='closed|exit'):
        rpc.request('build', {}, timeout=5)


def test_rpc_timeout_retires_process_and_rejects_late_reply(tmp_path):
    from minecraft_mod_ai.owner_rpc import OwnerRPC, OwnerRPCError

    server = tmp_path / 'slow.py'
    server.write_text('import sys, time\nsys.stdin.readline()\ntime.sleep(60)\n')
    with OwnerRPC([sys.executable, '-u', str(server)]) as rpc:
        with pytest.raises(OwnerRPCError, match='timed out'):
            rpc.request('build', {}, timeout=0.2)
        with pytest.raises(OwnerRPCError, match='closed'):
            rpc.request('build', {}, timeout=1)


def test_rpc_timeout_preserves_backend_stderr(tmp_path):
    from minecraft_mod_ai.owner_rpc import OwnerRPC, OwnerRPCError

    server = tmp_path / 'stderr_timeout.py'
    server.write_text("import sys, time\nsys.stdin.readline()\nprint('Gradle model resolution blocked', file=sys.stderr, flush=True)\ntime.sleep(60)\n")
    with OwnerRPC([sys.executable, '-u', str(server)]) as rpc, pytest.raises(
        OwnerRPCError, match='Gradle model resolution blocked'
    ):
        rpc.request('resolve', {}, timeout=1)


def test_rpc_ignores_non_protocol_stdout_noise(tmp_path):
    from minecraft_mod_ai.owner_rpc import OwnerRPC

    server = tmp_path / 'noisy.py'
    server.write_text('''import json, sys
print('[0.006s][warning][os,thread] JVM startup noise', flush=True)
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({'id': request['id'], 'result': {'ok': True}}), flush=True)
''')
    with OwnerRPC([sys.executable, '-u', str(server)]) as rpc:
        assert rpc.request('build', {}, timeout=5) == {'ok': True}


def test_rpc_malformed_response_fails_closed(tmp_path):
    from minecraft_mod_ai.owner_rpc import OwnerRPC, OwnerRPCError

    server = tmp_path / 'malformed.py'
    server.write_text("import sys\nsys.stdin.readline()\nprint('{not json', flush=True)\nsys.stdin.read()\n")
    with OwnerRPC([sys.executable, '-u', str(server)]) as rpc, pytest.raises(OwnerRPCError, match='protocol|JSON'):
        rpc.request('build', {}, timeout=3)
