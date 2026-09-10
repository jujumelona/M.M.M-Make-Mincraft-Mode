from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from minecraft_mod_ai.artifact_ports import PortKind, PortRegistry, TypedPort


def test_concurrent_distinct_port_publication_is_lossless():
    registry = PortRegistry()

    def publish(index: int) -> None:
        registry.publish(
            TypedPort(
                name=f"port.{index}",
                port_kind=PortKind.GENERIC,
                target_type="Any",
                value=str(index),
            )
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(publish, range(200)))

    ports = registry.all_ports()
    assert len(ports) == 200
    assert all(ports[f"port.{i}"].value == str(i) for i in range(200))


def test_concurrent_identical_publication_is_idempotent():
    registry = PortRegistry()
    port = TypedPort(
        name="shared",
        port_kind=PortKind.GENERIC,
        target_type="Any",
        value="stable",
    )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: registry.publish(port), range(100)))

    assert registry.get("shared") == port
    assert len(registry.all_ports()) == 1
