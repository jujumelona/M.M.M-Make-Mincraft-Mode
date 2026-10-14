"""Read JVM class declarations directly from JAR bytes; never invent descriptors."""
from __future__ import annotations

from dataclasses import dataclass
import re
import struct
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from .implementation_identity import compute_content_hash


class ClassFormatError(ValueError):
    pass


_FORBIDDEN_MODIFIED_UTF8_BYTE = re.compile(rb"[\x00\xf0-\xff]")


def _decode_modified_utf8(data: bytes) -> str:
    """Decode JVM CONSTANT_Utf8 data without rejecting Java char values.

    Modified UTF-8 encodes NUL as C0 80 and every UTF-16 code unit in at
    most three bytes. Java strings may contain unpaired surrogate units;
    valid pairs are combined to match Python's Unicode class/path names.
    """
    forbidden = _FORBIDDEN_MODIFIED_UTF8_BYTE.search(data)
    if forbidden is not None:
        offset = forbidden.start()
        raise ClassFormatError(
            f"Invalid modified UTF-8 byte 0x{data[offset]:02x} at offset {offset}"
        )
    try:
        value = data.replace(b"\xc0\x80", b"\x00").decode("utf-8", "surrogatepass")
    except UnicodeDecodeError as exc:
        raise ClassFormatError("Invalid modified UTF-8 encoding: " + exc.reason) from exc
    return value.encode("utf-16-be", "surrogatepass").decode("utf-16-be", "surrogatepass")


class _Reader:
    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def take(self, n: int) -> bytes:
        end = self.pos + n
        if end > len(self.data):
            raise ClassFormatError("Truncated class file")
        value = self.data[self.pos:end]
        self.pos = end
        return value

    def u1(self):
        return self.take(1)[0]

    def u2(self):
        return struct.unpack(">H", self.take(2))[0]

    def u4(self):
        return struct.unpack(">I", self.take(4))[0]


@dataclass(frozen=True)
class ApiMember:
    owner: str
    name: str
    descriptor: str
    kind: str
    access: int
    side: str
    exceptions: tuple[str, ...] = ()

    @property
    def is_static(self):
        return bool(self.access & 8)


@dataclass(frozen=True)
class ApiClass:
    name: str
    superclass: str
    interfaces: tuple[str, ...]
    access: int
    side: str
    members: tuple[ApiMember, ...]
    content_hash: str
    enclosing_class: str = ''
    inner_access: int | None = None


def parse_class(data: bytes, *, default_side: str = "COMMON") -> ApiClass:
    if default_side not in {"COMMON", "CLIENT", "SERVER"}:
        raise ClassFormatError("Explicit classpath side required")
    r = _Reader(data)
    if r.u4() != 0xCAFEBABE:
        raise ClassFormatError("Invalid class magic")
    r.u2(), r.u2()
    cp: list = [None] * r.u2()
    i = 1
    while i < len(cp):
        tag = r.u1()
        if tag == 1:
            try:
                cp[i] = _decode_modified_utf8(r.take(r.u2()))
            except ClassFormatError as exc:
                raise ClassFormatError(f"Invalid CONSTANT_Utf8 #{i}: {exc}") from exc
        elif tag in {3, 4}:
            cp[i] = r.take(4)
        elif tag in {5, 6}:
            cp[i] = r.take(8)
            i += 1
        elif tag in {7, 8, 16, 19, 20}:
            cp[i] = r.u2()
        elif tag in {9, 10, 11, 12, 17, 18}:
            cp[i] = (r.u2(), r.u2())
        elif tag == 15:
            cp[i] = (r.u1(), r.u2())
        else:
            raise ClassFormatError(f"Unknown constant pool tag {tag}")
        i += 1

    def utf(index):
        if not 0 < index < len(cp) or not isinstance(cp[index], str):
            raise ClassFormatError("Invalid UTF8 reference")
        return cp[index]

    def cname(index):
        return utf(cp[index]) if index else ""

    def element(a):
        tag = chr(a.u1())
        if tag == "e":
            return (utf(a.u2()), utf(a.u2()))
        if tag in "BCDFIJSZsc":
            return cp[a.u2()]
        if tag == "@":
            return annotation(a)
        if tag == "[":
            return [element(a) for _ in range(a.u2())]
        raise ClassFormatError("Invalid annotation element")

    def annotation(a):
        name = utf(a.u2())
        values = {}
        for _ in range(a.u2()):
            key = utf(a.u2())
            values[key] = element(a)
        return name, values

    def attributes():
        annotations = []
        exceptions = []
        inner_classes = []
        for _ in range(r.u2()):
            name, payload = utf(r.u2()), r.take(r.u4())
            if name in {"RuntimeVisibleAnnotations", "RuntimeInvisibleAnnotations"}:
                a = _Reader(payload)
                annotations.extend(annotation(a) for _ in range(a.u2()))
                if a.pos != len(payload):
                    raise ClassFormatError("Trailing annotation data")
            elif name == 'Exceptions':
                a = _Reader(payload)
                exceptions.extend(cname(a.u2()) for _ in range(a.u2()))
                if a.pos != len(payload):
                    raise ClassFormatError('Trailing exception data')
            elif name == 'InnerClasses':
                a = _Reader(payload)
                for _ in range(a.u2()):
                    inner, outer, simple, flags = a.u2(), a.u2(), a.u2(), a.u2()
                    inner_classes.append((cname(inner), cname(outer), utf(simple) if simple else '', flags))
                if a.pos != len(payload):
                    raise ClassFormatError('Trailing inner class data')
        return annotations, tuple(exceptions), inner_classes

    def side(annotations, fallback):
        values = []
        for name, fields in annotations:
            if name in {"Lnet/fabricmc/api/Environment;", "Lnet/minecraftforge/api/distmarker/OnlyIn;",
                        "Lnet/neoforged/api/distmarker/OnlyIn;"}:
                value = fields.get("value")
                if not isinstance(value, tuple):
                    raise ClassFormatError("Invalid side annotation")
                mapped = {"CLIENT": "CLIENT", "SERVER": "SERVER", "DEDICATED_SERVER": "SERVER"}.get(value[1])
                if mapped is None:
                    raise ClassFormatError("Unknown environment")
                values.append(mapped)
        if len(set(values)) > 1:
            raise ClassFormatError("Conflicting environments")
        return values[0] if values else fallback

    access, name, superclass = r.u2(), cname(r.u2()), cname(r.u2())
    interfaces = tuple(cname(r.u2()) for _ in range(r.u2()))
    raw_members = []
    for kind in ("FIELD", "METHOD"):
        for _ in range(r.u2()):
            flags, member_name, descriptor = r.u2(), utf(r.u2()), utf(r.u2())
            annotations, exceptions, _ = attributes()
            raw_members.append((member_name, descriptor, kind, flags, annotations, exceptions))
    class_annotations, _, inner_classes = attributes()
    class_side = side(class_annotations, default_side)
    if r.pos != len(data):
        raise ClassFormatError("Trailing class bytes")
    members = tuple(ApiMember(name, n, d, "CONSTRUCTOR" if n == "<init>" else k, f, side(a, class_side), e)
                    for n, d, k, f, a, e in raw_members)
    own_inner = next((row for row in inner_classes if row[0] == name and row[1]), None)
    return ApiClass(name, superclass, interfaces, access, class_side, members, compute_content_hash(data),
                    own_inner[1] if own_inner else '', own_inner[3] if own_inner else None)


def inspect_jar(path: Path, *, java_version: int, default_side: str = "COMMON",
                require_classes: bool = True) -> dict[str, ApiClass]:
    """Read eligible declarations, keeping required API archives strict.

    A compile classpath can include resource/native/aggregate JARs which expose
    no Java declarations. Its callers opt out of the per-archive class minimum;
    malformed archives and class entries still fail inspection.
    """
    path, entry = Path(path), ""
    try:
        if default_side not in {"COMMON", "CLIENT", "SERVER"}:
            raise ClassFormatError("Explicit classpath side required")
        selected = {}
        with ZipFile(path) as jar:
            names = jar.namelist()
            if len(names) != len(set(names)):
                raise ClassFormatError("Duplicate JAR entries")
            entry = "META-INF/MANIFEST.MF" if "META-INF/MANIFEST.MF" in names else ""
            manifest = jar.read(entry).decode("utf-8") if entry else ""
            multi = "multi-release: true" in manifest.lower()
            for name in names:
                if not name.endswith(".class"):
                    continue
                entry = name
                version, logical = 0, name
                if name.startswith("META-INF/versions/"):
                    parts = name.split("/", 3)
                    if not multi:
                        continue
                    version, logical = int(parts[2]), parts[3]
                    if version > java_version:
                        continue
                if logical not in selected or selected[logical][0] < version:
                    selected[logical] = (version, name)
            result = {}
            for logical, (_, name) in sorted(selected.items()):
                entry = name
                item = parse_class(jar.read(name), default_side=default_side)
                if logical != item.name + ".class":
                    raise ClassFormatError("JAR path/class name mismatch")
                # Module descriptors are metadata, not callable classpath types.
                if logical != "module-info.class":
                    result[item.name] = item
        entry = ""
        if require_classes and not result:
            raise ClassFormatError("JAR has no classes")
        return result
    except (ValueError, BadZipFile) as exc:
        location = f"jar={path}" + (f" entry={entry}" if entry else "")
        raise ClassFormatError(f"{exc} ({location})") from exc
