"""Read public issuer documents, record their provenance, and replay exact bytes."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZIP_DEFLATED, ZipFile

import requests


class ResearchError(ValueError):
    """A report could not establish its issuer, source, or measurement."""


_ALLOWED = {"www.sec.gov", "data.sec.gov", "mtygroup.com", "www.mtygroup.com"}
_MAX_BYTES = 48 * 1024 * 1024
_SEC_LOCK = threading.Lock()
_last_request = 0.0


def _pace_sec():
    global _last_request
    with _SEC_LOCK:
        time.sleep(max(0, 0.22 - (time.monotonic() - _last_request)))
        _last_request = time.monotonic()


@dataclass(frozen=True)
class Source:
    url: str
    sha256: str
    retrieved_at: str
    size: int


class SourceClient:
    def __init__(self, replay: Path | None = None):
        self.started_at = datetime.now(UTC).isoformat()
        self.sources: dict[str, Source] = {}
        self._bytes: dict[str, bytes] = {}
        self._replay = replay is not None
        if replay is not None:
            with ZipFile(replay) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                self.started_at = manifest["started_at"]
                for item in manifest["sources"]:
                    source = Source(**item)
                    entry = archive.getinfo("documents/" + source.sha256)
                    if entry.file_size > _MAX_BYTES or entry.file_size != source.size:
                        raise ResearchError("Invalid document size in source archive.")
                    data = archive.read(entry)
                    if sha256(data).hexdigest() != source.sha256:
                        raise ResearchError("Source archive checksum mismatch.")
                    self.sources[source.url] = source
                    self._bytes[source.url] = data

    def get(self, url: str) -> bytes:
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.hostname not in _ALLOWED or parts.username or parts.port:
            raise ResearchError("Source is outside the configured issuer or regulator hosts.")
        if url in self._bytes:
            return self._bytes[url]
        if self._replay:
            raise ResearchError(f"The recorded run has no source for {url}.")
        if parts.hostname.endswith("sec.gov"):
            _pace_sec()
        agent = os.environ.get("SEC_USER_AGENT") or (
            "Axion Equity Research/1.0 (https://github.com/asrh-82/ai-stock-research-platform/issues)"
        )
        try:
            with requests.get(
                url, headers={"User-Agent": agent}, stream=True, timeout=(7, 30), allow_redirects=False
            ) as response:
                if response.status_code != 200:
                    raise ResearchError(
                        f"{parts.hostname} returned HTTP {response.status_code}. Try again later."
                    )
                data = bytearray()
                for chunk in response.iter_content(64 * 1024):
                    data.extend(chunk)
                    if len(data) > _MAX_BYTES:
                        raise ResearchError(f"The document from {parts.hostname} exceeds the download limit.")
        except requests.RequestException as exc:
            raise ResearchError(f"Could not retrieve {url}. Try again later.") from exc
        content = bytes(data)
        if not content:
            raise ResearchError(f"Empty response from {parts.hostname}.")
        self.sources[url] = Source(
            url, sha256(content).hexdigest(), datetime.now(UTC).isoformat(), len(content)
        )
        self._bytes[url] = content
        return content

    def json(self, url: str):
        try:
            return json.loads(self.get(url))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ResearchError(f"Expected a JSON document from {url}.") from exc

    def save(self, path: Path):
        """Publish the bundle only when every document and its manifest have been written."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
        try:
            with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
                archive.writestr(
                    "manifest.json",
                    json.dumps(
                        {
                            "started_at": self.started_at,
                            "sources": [asdict(s) for s in self.sources.values()],
                        },
                        indent=2,
                    ),
                )
                documents = {s.sha256: self._bytes[s.url] for s in self.sources.values()}
                for digest, data in documents.items():
                    archive.writestr("documents/" + digest, data)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
