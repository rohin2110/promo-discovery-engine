"""Safe file ingestion: uploads, ZIP archives (nested), size and path limits.

Nothing is extracted to disk. Archive members are read in memory with hard
limits so that zip bombs, path traversal, symlinks and oversized uploads are
rejected and *reported*, never silently dropped.
"""
from __future__ import annotations

import hashlib
import io
import logging
import stat
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from config.settings import AppConfig
from models.schemas import FileRecord, ParseIssue, Severity

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".docx", ".json", ".csv", ".srt", ".fountain"}
ARCHIVE_EXTENSIONS = {".zip"}
IGNORED_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}


@dataclass
class InputDocument:
    display_name: str  # unique, includes archive path, e.g. "data.zip/show/ep1.txt"
    filename: str  # basename
    data: bytes
    extension: str
    sha256: str

    @property
    def size(self) -> int:
        return len(self.data)


@dataclass
class IngestionResult:
    documents: list[InputDocument] = field(default_factory=list)
    records: list[FileRecord] = field(default_factory=list)
    issues: list[ParseIssue] = field(default_factory=list)


class _State:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.result = IngestionResult()
        self.seen_hashes: dict[str, str] = {}
        self.used_names: set[str] = set()
        self.total_uncompressed = 0
        self.file_count = 0

    def issue(self, severity: Severity, name: str, msg: str, code: str) -> None:
        self.result.issues.append(ParseIssue(severity=severity, source_filename=name, message=msg, code=code))
        getattr(log, "error" if severity == Severity.error else "warning" if severity == Severity.warning else "info")(
            "%s: %s", name, msg
        )

    def record(self, name: str, data: bytes | None, status: str, message: str = "", ext: str = "") -> None:
        self.result.records.append(
            FileRecord(
                display_name=name,
                size_bytes=len(data) if data else 0,
                sha256=hashlib.sha256(data).hexdigest() if data else "",
                extension=ext or PurePosixPath(name).suffix.lower(),
                status=status,  # type: ignore[arg-type]
                message=message,
            )
        )

    def unique_name(self, name: str) -> str:
        if name not in self.used_names:
            self.used_names.add(name)
            return name
        p = PurePosixPath(name)
        i = 2
        while True:
            cand = str(p.with_name(f"{p.stem} ({i}){p.suffix}"))
            if cand not in self.used_names:
                self.used_names.add(cand)
                return cand
            i += 1


def _is_unsafe_member(name: str) -> str | None:
    norm = name.replace("\\", "/")
    if norm.startswith("/") or (len(norm) > 1 and norm[1] == ":"):
        return "absolute path in archive"
    if any(part == ".." for part in PurePosixPath(norm).parts):
        return "path traversal ('..') in archive"
    if "\x00" in norm:
        return "null byte in archive member name"
    return None


def _add_file(state: _State, name: str, data: bytes) -> None:
    cfg = state.cfg
    base = PurePosixPath(name).name
    ext = PurePosixPath(base).suffix.lower()
    if base.lower() in IGNORED_NAMES or base.startswith("._") or "__MACOSX/" in name:
        state.record(name, data, "skipped", "system/metadata file ignored", ext)
        return
    state.file_count += 1
    if state.file_count > cfg.max_files:
        state.record(name, data, "rejected", f"file limit ({cfg.max_files}) exceeded")
        state.issue(Severity.error, name, f"file limit ({cfg.max_files}) exceeded; file not processed", "too_many_files")
        return
    if len(data) > cfg.max_file_mb * 1024 * 1024:
        state.record(name, data, "rejected", f"file exceeds {cfg.max_file_mb} MB limit")
        state.issue(Severity.error, name, f"file exceeds the {cfg.max_file_mb} MB per-file limit", "file_too_large")
        return
    if ext in ARCHIVE_EXTENSIONS:
        _ingest_zip(state, name, data, depth=name.count(".zip/") + 1)
        return
    if ext not in SUPPORTED_EXTENSIONS:
        state.record(name, data, "unsupported", f"unsupported file type '{ext or '(none)'}'")
        state.issue(
            Severity.error,
            name,
            f"unsupported file type '{ext or '(none)'}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS | ARCHIVE_EXTENSIONS))}",
            "unsupported_format",
        )
        return
    digest = hashlib.sha256(data).hexdigest()
    if digest in state.seen_hashes:
        first = state.seen_hashes[digest]
        state.record(name, data, "duplicate", f"identical content to '{first}'", ext)
        state.issue(Severity.warning, name, f"duplicate of '{first}' (identical content); skipped", "duplicate_file")
        return
    state.seen_hashes[digest] = name
    unique = state.unique_name(name)
    if unique != name:
        state.issue(Severity.warning, name, f"duplicate filename with different content; renamed to '{unique}'", "duplicate_name")
    state.result.documents.append(InputDocument(unique, base, data, ext, digest))
    state.record(unique, data, "parsed", "", ext)  # status updated after parsing


def _ingest_zip(state: _State, name: str, data: bytes, depth: int) -> None:
    cfg = state.cfg
    if depth > cfg.max_zip_depth:
        state.record(name, data, "rejected", f"nested archive depth > {cfg.max_zip_depth}")
        state.issue(Severity.error, name, f"nested ZIP depth exceeds {cfg.max_zip_depth}; not extracted", "zip_depth")
        return
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        state.record(name, data, "failed", f"invalid ZIP: {exc}")
        state.issue(Severity.error, name, f"cannot open ZIP archive: {exc}", "bad_zip")
        return
    state.record(name, data, "skipped", "archive expanded", ".zip")
    max_file = int(cfg.max_file_mb * 1024 * 1024)
    max_total = int(cfg.max_total_uncompressed_mb * 1024 * 1024)
    with zf:
        for info in zf.infolist():
            member = f"{name}/{info.filename}"
            if info.is_dir():
                continue
            reason = _is_unsafe_member(info.filename)
            mode = (info.external_attr >> 16) & 0o170000
            if not reason and mode and stat.S_ISLNK(mode):
                reason = "symbolic link in archive"
            if not reason and info.flag_bits & 0x1:
                reason = "encrypted archive member"
            if reason:
                state.record(member, None, "rejected", reason)
                state.issue(Severity.error, member, f"unsafe archive member rejected: {reason}", "unsafe_archive_path")
                continue
            if info.file_size > max_file:
                state.record(member, None, "rejected", f"member exceeds {cfg.max_file_mb} MB")
                state.issue(Severity.error, member, f"archive member exceeds {cfg.max_file_mb} MB limit", "file_too_large")
                continue
            ratio = info.file_size / max(info.compress_size, 1)
            if info.file_size > 1024 * 1024 and ratio > cfg.max_compression_ratio:
                state.record(member, None, "rejected", f"suspicious compression ratio {ratio:.0f}:1")
                state.issue(Severity.error, member, f"possible zip bomb (ratio {ratio:.0f}:1); rejected", "zip_bomb")
                continue
            if state.total_uncompressed + info.file_size > max_total:
                state.record(member, None, "rejected", "total uncompressed size limit reached")
                state.issue(
                    Severity.error, member,
                    f"total uncompressed size exceeds {cfg.max_total_uncompressed_mb} MB; remaining members not processed",
                    "archive_too_large",
                )
                break
            try:
                with zf.open(info) as fh:
                    content = fh.read(max_file + 1)
            except (RuntimeError, zipfile.BadZipFile, NotImplementedError, OSError) as exc:
                state.record(member, None, "failed", f"cannot read member: {exc}")
                state.issue(Severity.error, member, f"cannot read archive member: {exc}", "bad_zip_member")
                continue
            if len(content) > max_file:  # header lied about the size
                state.record(member, None, "rejected", "member larger than declared")
                state.issue(Severity.error, member, "archive member larger than declared size; rejected", "zip_bomb")
                continue
            state.total_uncompressed += len(content)
            _add_file(state, member, content)


def ingest_uploads(uploads: list[tuple[str, bytes]], cfg: AppConfig) -> IngestionResult:
    """Turn raw uploads (name, bytes) into validated in-memory documents."""
    state = _State(cfg)
    total = sum(len(b) for _, b in uploads)
    if total > cfg.max_upload_mb * 1024 * 1024:
        state.issue(Severity.error, "(upload)", f"upload of {total / 1e6:.1f} MB exceeds the {cfg.max_upload_mb} MB limit", "upload_too_large")
        for name, data in uploads:
            state.record(name, data, "rejected", "upload size limit exceeded")
        return state.result
    if not uploads:
        state.issue(Severity.error, "(upload)", "no files were uploaded", "empty_upload")
    for name, data in uploads:
        safe_name = PurePosixPath(name.replace("\\", "/")).name or "upload"
        _add_file(state, safe_name, data)
    if uploads and not state.result.documents:
        state.issue(Severity.error, "(upload)", "no supported screenplay files were found in the upload", "no_documents")
    return state.result
