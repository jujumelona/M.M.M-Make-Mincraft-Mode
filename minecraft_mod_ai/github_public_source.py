from __future__ import annotations

"""Public Git metadata for source inspection when the GitHub API is unavailable.

Only advertised references and actual Git objects supply source identities.  No
working tree is created and no repository code, hooks, credentials or inherited
Git configuration is used.  The temporary object database lives only for one
inspection; callers subsequently retrieve bytes by the returned commit/blob IDs.
"""

import hashlib
import json
import os
import re
import selectors
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .reuse_license import is_reusable_source_license

_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}")
_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_LICENSE_NAMES = frozenset({"license", "license.txt", "license.md", "copying", "copying.txt", "copying.md"})

# Exact standard bodies from the official SPDX License List.  Deliberately a
# subset of SPDX's matching rules: replaceable author names in disclaimers,
# additional terms, and unsupported licenses are not guessed.
# https://spdx.org/licenses/MIT.html
# https://spdx.org/licenses/BSD-2-Clause.html
# https://spdx.org/licenses/BSD-3-Clause.html
# https://spdx.org/licenses/ISC.html
_MIT_BODY = '''Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''
_BSD_PERMISSION = '''Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.'''
_BSD_THIRD_CLAUSE = '''3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.'''
_BSD_DISCLAIMER = '''THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.'''
_ISC_BODY = '''Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.'''
_TITLES = {
    "MIT": frozenset({"MIT License", "The MIT License (MIT)"}),
    "BSD-2-Clause": frozenset({'BSD 2-Clause "Simplified" License', "BSD 2-Clause License"}),
    "BSD-3-Clause": frozenset({'BSD 3-Clause "New" or "Revised" License', "BSD 3-Clause License"}),
    "ISC": frozenset({"ISC License", "The ISC License (ISC)"}),
}
_NOTICE_YEAR = r"[12][0-9]{3}(?:[ \t]*-[ \t]*[12][0-9]{3})?"
_COPYRIGHT = re.compile(
    r"Copyright(?: \([cC]\)| ©)?[ \t]+"
    + _NOTICE_YEAR + r"(?:[ \t]*,[ \t]*" + _NOTICE_YEAR + r")*[ \t]+(?P<holders>.+)"
)
# A holder is an explicit identifier, not an unrestricted prose placeholder.
# Unquoted multiword names intentionally remain unresolved.  Display names may
# be explicitly quoted, and lists must use a comma, ampersand, or literal 'and'.
_HOLDER_TOKEN = r"@?[A-Za-z0-9_][A-Za-z0-9_.+-]*(?:@[A-Za-z0-9][A-Za-z0-9.-]*)?"
_QUOTED_HOLDER = r'''(?:"[\w][\w .,&'@()+/-]*"|'[\w][\w .,&@()+/-]*')'''
_HOLDER = r"(?:" + _HOLDER_TOKEN + r"|" + _QUOTED_HOLDER + r")"
_HOLDERS = re.compile(_HOLDER + r"(?:(?:[ \t]*[,\x26][ \t]*|[ \t]+and[ \t]+)" + _HOLDER + r")*\.?" )


def _normalize_license(text: str) -> str:
    return re.sub(r"[ \t\r\n\f\v]+", " ", text).strip()


def _copyright_only_notice(line: str) -> bool:
    match = _COPYRIGHT.fullmatch(line)
    return bool(match is not None and _HOLDERS.fullmatch(match["holders"]))


def _standard_license(raw: bytes, *, rejection_reason: list[str] | None = None) -> str | None:
    """Accept a complete standard body with only a title/copyright preamble.

    Whitespace is normalized; substantive words and punctuation are preserved.
    A copyright placeholder is not a wildcard for arbitrary introductory terms.
    """
    try:
        text = raw.decode("utf-8-sig", errors="strict")
    except UnicodeError:
        if rejection_reason is not None:
            rejection_reason.append("license_text_not_utf8")
        return None
    if "\x00" in text:
        if rejection_reason is not None:
            rejection_reason.append("license_text_contains_nul")
        return None
    lines = text.splitlines()
    ambiguous_notice = False
    for license_id, titles in _TITLES.items():
        candidate = list(lines)
        while candidate and not candidate[0].strip():
            candidate.pop(0)
        if candidate and candidate[0].strip() in titles:
            candidate.pop(0)
        saw_copyright = False
        while candidate:
            line = candidate[0].strip()
            if not line:
                candidate.pop(0)
                continue
            if _copyright_only_notice(line):
                saw_copyright = True
                candidate.pop(0)
                continue
            if saw_copyright and line == "All rights reserved.":
                candidate.pop(0)
                continue
            if line.startswith("Copyright"):
                ambiguous_notice = True
            break
        if not saw_copyright:
            continue
        body = _normalize_license("\n".join(candidate))
        if license_id == "MIT":
            variants = {
                _normalize_license(_MIT_BODY),
                _normalize_license(_MIT_BODY).replace("limitation the rights", "limitation on the rights"),
            }
            variants |= {value.replace("permission notice shall", "permission notice (including the next paragraph) shall") for value in tuple(variants)}
        elif license_id == "BSD-2-Clause":
            variants = {_normalize_license(_BSD_PERMISSION + "\n" + _BSD_DISCLAIMER)}
        elif license_id == "BSD-3-Clause":
            variants = {_normalize_license(_BSD_PERMISSION + "\n" + _BSD_THIRD_CLAUSE + "\n" + _BSD_DISCLAIMER)}
        else:
            variants = {_normalize_license(_ISC_BODY), _normalize_license(_ISC_BODY).replace("and/or distribute", "and distribute")}
        if body in variants and is_reusable_source_license(license_id):
            return license_id
    if rejection_reason is not None:
        rejection_reason.append("copyright_notice_not_unambiguous" if ambiguous_notice else "license_notice_or_body_not_supported")
    return None


class PublicGitSourceError(RuntimeError):
    def __init__(self, message: str, *, kind: str = "invalid_metadata") -> None:
        super().__init__(message)
        self.kind = kind
        self.hostname = "github.com"
        # Subprocess exit codes/timeouts are repository-scoped observations;
        # neither establishes independent evidence of a host-wide outage.
        self.request_failure = None


def public_git_available() -> bool:
    """Capability check only; it does not invoke Git or probe the network."""
    return False  # Public git donor inspection prohibited by CurseForge-only policy.


def _disk_bytes(root: Path, budget: int) -> int:
    total = 0
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in (*directories, *files):
            path = Path(parent) / name
            try:
                stat = path.lstat()
            except FileNotFoundError:
                continue
            if path.is_symlink():
                raise PublicGitSourceError("Public Git metadata created an unexpected symlink.")
            total += stat.st_size
            if total > budget:
                raise PublicGitSourceError("Public Git metadata exceeded its temporary disk byte budget.", kind="budget")
    return total


def _terminate(process: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "posix":
            # A Git parent may have exited while a remote helper still owns its
            # output pipes.  Terminate the isolated group in that case as well.
            os.killpg(process.pid, signal.SIGKILL)
        elif process.poll() is None:
            process.kill()
    except ProcessLookupError:
        pass
    process.wait()


def _run_git(
    arguments: list[str], *, executable: str, root: Path, repository_url: str,
    output_budget: int, disk_budget: int, timeout_seconds: float,
    network: bool = False,
) -> bytes:
    """Bound stdout/stderr and temporary disk while a sanitized Git runs."""
    home = root / "home"
    empty = root / "empty"
    command = [executable]
    for setting in (
        "credential.helper=", "credential.interactive=never", "core.askPass=",
        f"core.hooksPath={empty}", "core.fsmonitor=false", "core.pager=cat",
        "http.followRedirects=false", "protocol.allow=never", "protocol.https.allow=always",
        "protocol.file.allow=never", "submodule.recurse=false", "fetch.recurseSubmodules=false",
        "fetch.fsckObjects=true", "transfer.fsckObjects=true", "gc.auto=0",
        "maintenance.auto=false", "fetch.writeCommitGraph=false", "init.defaultBranch=mmm-inspection",
        f"init.templateDir={empty}", f"remote.origin.url={repository_url}",
        "remote.origin.promisor=true", "remote.origin.partialclonefilter=blob:none",
    ):
        command.extend(["-c", setting])
    command.extend(arguments)
    environment = {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": str(home), "XDG_CONFIG_HOME": str(home), "TMPDIR": str(root),
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull, "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "", "SSH_ASKPASS": "", "GCM_INTERACTIVE": "Never",
        "GIT_ATTR_NOSYSTEM": "1", "GIT_LFS_SKIP_SMUDGE": "1",
        "LC_ALL": "C", "LANG": "C",
    }
    for name in ("SYSTEMROOT", "WINDIR"):
        if name in os.environ:
            environment[name] = os.environ[name]
    try:
        process = subprocess.Popen(
            command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
    except OSError as exc:
        raise PublicGitSourceError("Public Git executable could not start.", kind="unavailable") from exc
    stdout = bytearray()
    output_size = 0
    deadline = time.monotonic() + timeout_seconds
    try:
        with selectors.DefaultSelector() as selector:
            assert process.stdout is not None and process.stderr is not None
            selector.register(process.stdout, selectors.EVENT_READ, True)
            selector.register(process.stderr, selectors.EVENT_READ, False)
            while selector.get_map():
                if time.monotonic() >= deadline:
                    raise PublicGitSourceError("Public Git inspection exceeded its command timeout.", kind="timeout" if network else "budget")
                _disk_bytes(root, disk_budget)
                for key, _events in selector.select(timeout=min(0.1, max(0.0, deadline - time.monotonic()))):
                    chunk = os.read(key.fileobj.fileno(), 64 * 1024)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    output_size += len(chunk)
                    if output_size > output_budget:
                        raise PublicGitSourceError("Public Git command exceeded its response byte budget.", kind="budget")
                    if key.data:
                        stdout.extend(chunk)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PublicGitSourceError("Public Git inspection exceeded its command timeout.", kind="timeout" if network else "budget")
            try:
                returncode = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise PublicGitSourceError("Public Git inspection exceeded its command timeout.", kind="timeout" if network else "budget") from exc
        _disk_bytes(root, disk_budget)
        if returncode:
            # Never return stderr, which may contain host/environment details.
            # A Git exit code does not identify an HTTP status or distinguish a
            # missing/private repository from a whole-host transport outage.
            raise PublicGitSourceError("Public Git inspection command failed.", kind="command_failed" if network else "invalid_metadata")
        return bytes(stdout)
    finally:
        _terminate(process)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


def _safe_path(path: str) -> bool:
    return bool(path and not path.startswith("/") and "\\" not in path
                and not re.match(r"^[A-Za-z]:", path)
                and all(part not in {"", ".", ".."} and part.casefold() != ".git" for part in path.split("/"))
                and not any(ord(char) < 32 or ord(char) == 127 for char in path))


def _advertised_branch(raw: bytes, minecraft_version: str, loader: str) -> tuple[str, str, str]:
    try:
        lines = raw.decode("utf-8", errors="strict").splitlines()
    except UnicodeError as exc:
        raise PublicGitSourceError("Public Git references are not valid UTF-8.") from exc
    branches: dict[str, str] = {}
    default_branch = head = ""
    for line in lines:
        fields = line.split("\t")
        if len(fields) != 2:
            raise PublicGitSourceError("Public Git advertised malformed references.")
        value, ref = fields
        if ref == "HEAD" and value.startswith("ref: refs/heads/"):
            branch = value[len("ref: refs/heads/"):]
            if default_branch and default_branch != branch or not _safe_path(branch):
                raise PublicGitSourceError("Public Git advertised an invalid default branch.")
            default_branch = branch
        elif _OBJECT_ID.fullmatch(value):
            if ref == "HEAD":
                if head and head != value:
                    raise PublicGitSourceError("Public Git advertised conflicting HEAD identities.")
                head = value
            elif ref.startswith("refs/heads/"):
                branch = ref[len("refs/heads/"):]
                if not _safe_path(branch) or branch in branches and branches[branch] != value:
                    raise PublicGitSourceError("Public Git advertised invalid branch identities.")
                branches[branch] = value
            else:
                raise PublicGitSourceError("Public Git advertised an unexpected reference.")
        else:
            raise PublicGitSourceError("Public Git advertised a non-immutable reference.")
    if not default_branch or not head or branches.get(default_branch) != head:
        raise PublicGitSourceError("Public Git did not prove its advertised default branch and HEAD.")
    selected = default_branch
    if minecraft_version:
        version_name = re.compile(r"(?<![0-9.])" + re.escape(minecraft_version) + r"(?![0-9.]|-(?:pre|rc)[0-9])")
        if not version_name.search(default_branch):
            matches = [name for name in branches if version_name.search(name)]
            if matches:
                selected = max(matches, key=lambda name: (
                    loader.casefold() in name.casefold() if loader else False,
                    name == minecraft_version, -len(name), name,
                ))
    return default_branch, selected, branches[selected]


def _semantic_rejection(repository: str, reason: str, path: str = "") -> None:
    record = {"repository": repository, "reason": reason}
    if path:
        record["path"] = path[:512]
    print("SOURCE_REUSE_PUBLIC_GIT_REJECTED: " + json.dumps(record, ensure_ascii=True), flush=True)


def _tree_blobs(raw: bytes, object_length: int, *, repository: str = "") -> dict[str, str] | None:
    if raw and not raw.endswith(b"\x00"):
        raise PublicGitSourceError("Public Git returned an incomplete immutable tree.")
    blobs: dict[str, str] = {}
    for entry in raw.split(b"\x00"):
        if not entry:
            continue
        try:
            metadata, encoded_path = entry.split(b"\t", 1)
            mode, kind, encoded_sha = metadata.split(b" ")
            path = encoded_path.decode("utf-8", errors="strict")
            sha = encoded_sha.decode("ascii", errors="strict")
        except (ValueError, UnicodeError) as exc:
            raise PublicGitSourceError("Public Git returned malformed tree metadata.") from exc
        if not _safe_path(path) or path in blobs or not _OBJECT_ID.fullmatch(sha) or len(sha) != object_length:
            raise PublicGitSourceError("Public Git returned unsafe paths or invalid object identities.")
        # A snapshot with symlinks or gitlinks cannot claim a complete regular
        # source tree through archive/raw transports; leave it unresolved.
        if kind != b"blob" or mode not in {b"100644", b"100755"}:
            _semantic_rejection(repository, "unsupported_tree_entry", path)
            return None
        blobs[path] = sha
    return blobs


def _license_candidate(path: str) -> bool:
    parts = path.casefold().split("/")
    leaf = parts[-1]
    return leaf in _LICENSE_NAMES or leaf.startswith(("license-", "license.", "copying-", "copying.")) or "licenses" in parts[:-1]


def inspect_public_repository(
    repository: str, minecraft_version: str, loader: str,
    response_byte_budget: int, archive_byte_budget: int, single_blob_byte_budget: int,
    *, command_timeout_seconds: float = 30.0,
) -> dict[str, Any] | None:
    """Return authoritative pinned Git metadata and conservative license evidence.

    ``None`` means source/license semantics are unsupported or unresolved.  Typed
    errors distinguish transport/resource failures from a successful inspection.
    No source compatibility or build success is asserted by this function.
    """
    raise PublicGitSourceError("Public Git inspection disabled: CurseForge only.", kind="unavailable")
    if not _REPOSITORY.fullmatch(repository) or any(part in {".", ".."} for part in repository.split("/")):
        raise PublicGitSourceError("Invalid public GitHub repository identity.")
    if any(type(value) is not int or value <= 0 for value in (response_byte_budget, archive_byte_budget, single_blob_byte_budget)):
        raise PublicGitSourceError("Public Git inspection requires positive byte budgets.", kind="budget")
    if not isinstance(command_timeout_seconds, (int, float)) or not 0 < command_timeout_seconds < float("inf"):
        raise PublicGitSourceError("Public Git inspection requires a finite positive timeout.", kind="budget")
    executable = shutil.which("git")
    if executable is None:
        raise PublicGitSourceError("Public Git source inspection is unavailable without Git.", kind="unavailable")
    executable = str(Path(executable).resolve())
    repository_url = "https://github.com/" + repository + ".git"
    with tempfile.TemporaryDirectory(prefix="mmm-public-source-") as directory:
        root = Path(directory)
        (root / "home").mkdir()
        (root / "empty").mkdir()

        def git(arguments: list[str], *, budget: int = response_byte_budget, network: bool = False) -> bytes:
            return _run_git(arguments, executable=executable, root=root, repository_url=repository_url,
                            output_budget=budget, disk_budget=archive_byte_budget,
                            timeout_seconds=float(command_timeout_seconds), network=network)

        advertised = git(["ls-remote", "--symref", "--exit-code", repository_url, "HEAD", "refs/heads/*"], network=True)
        default_branch, selected_branch, commit_sha = _advertised_branch(advertised, minecraft_version, loader)
        bare = root / "objects.git"
        object_format = "sha1" if len(commit_sha) == 40 else "sha256"
        git(["init", "--bare", "--quiet", "--template=" + str(root / "empty"), "--object-format=" + object_format, str(bare)])
        prefix = ["--git-dir=" + str(bare)]
        git(prefix + ["fetch", "--quiet", "--depth=1", "--filter=blob:none", "--no-tags", "--no-recurse-submodules", "origin", commit_sha], network=True)
        actual_commit = git(prefix + ["rev-parse", "--verify", "FETCH_HEAD^{commit}"]).decode("ascii", errors="strict").strip()
        if actual_commit != commit_sha:
            raise PublicGitSourceError("Fetched public Git commit did not match the advertised immutable identity.")
        blobs = _tree_blobs(git(prefix + ["ls-tree", "-r", "-z", "--full-tree", commit_sha]), len(commit_sha), repository=repository)
        if blobs is None:
            return None
        if not blobs:
            _semantic_rejection(repository, "empty_source_tree")
            return None
        license_paths = sorted((path for path in blobs if _license_candidate(path)), key=lambda path: ("/" in path, path.casefold(), path))
        if not any("/" not in path and path.casefold() in _LICENSE_NAMES for path in license_paths):
            _semantic_rejection(repository, "missing_root_license")
            return None
        license_id = ""
        verified_blobs: dict[str, bytes] = {}
        license_files: dict[str, dict[str, str]] = {}
        total_license_bytes = 0
        for path in license_paths:
            sha = blobs[path]
            size_text = git(prefix + ["cat-file", "-s", sha], network=True).strip()
            try:
                size = int(size_text)
            except ValueError as exc:
                raise PublicGitSourceError("Pinned public Git license size was invalid.") from exc
            if size <= 0 or size > single_blob_byte_budget or total_license_bytes + size > response_byte_budget:
                raise PublicGitSourceError("Pinned public Git license exceeded its byte budget.", kind="budget")
            raw = git(prefix + ["cat-file", "blob", sha], budget=min(response_byte_budget, single_blob_byte_budget), network=True)
            identity = b"blob " + str(len(raw)).encode("ascii") + b"\x00" + raw
            actual_sha = (hashlib.sha1(identity).hexdigest() if len(sha) == 40 else hashlib.sha256(identity).hexdigest())
            if len(raw) != size or actual_sha != sha:
                raise PublicGitSourceError("Pinned public Git license blob identity did not verify.")
            rejection_reason: list[str] = []
            detected = _standard_license(raw, rejection_reason=rejection_reason)
            if detected is None:
                _semantic_rejection(repository, rejection_reason[0] if rejection_reason else "license_notice_or_body_not_supported", path)
                return None
            if license_id and detected != license_id:
                _semantic_rejection(repository, "conflicting_pinned_license_ids", path)
                return None
            license_id = detected
            total_license_bytes += len(raw)
            verified_blobs[sha] = raw
            license_files[path] = {"blob_sha": sha, "license_id": detected, "sha256": "sha256:" + hashlib.sha256(raw).hexdigest()}
        license_path = next(path for path in license_paths if "/" not in path and path.casefold() in _LICENSE_NAMES)
        license_blob_sha = blobs[license_path]
        return {
            "commit_sha": commit_sha, "default_branch": default_branch, "selected_branch": selected_branch,
            "source_url": "https://github.com/" + repository, "blobs": blobs,
            "license_id": license_id, "license_path": license_path, "license_blob_sha": license_blob_sha,
            "license_text_sha256": license_files[license_path]["sha256"],
            "license_url": "https://github.com/" + repository + "/blob/" + commit_sha + "/" + license_path,
            "verified_license_bytes": verified_blobs[license_blob_sha], "verified_blobs": verified_blobs,
            "license_files": license_files, "metadata_transport": "public_git",
            "compatibility": "unverified_until_pinned_metadata_and_build_pass",
        }


__all__ = ["PublicGitSourceError", "inspect_public_repository", "public_git_available"]
