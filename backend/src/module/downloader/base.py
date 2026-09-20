"""Download-client protocol and capability declarations.

`DownloadClient` (the facade) delegates every concrete operation to an object
that implements `DownloaderClient`. Different backends support different subsets
of the qBittorrent surface, so each concrete client advertises what it can do
via a `DownloaderCapabilities` class attribute; the facade consults it and skips
unsupported operations instead of blowing up on a missing method.
"""

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Protocol, runtime_checkable


def torrent_infohash(data: bytes) -> str | None:
    """从 .torrent 字节中提取 v1 infohash（bencoded ``info`` 字典的 SHA-1）。

    用途不止 qB 的重复上传检测：手写极简 bencode 游标（只需定位 info 值的
    字节区间，不需要完整解码），损坏/非 bencode 输入返回 None——manual
    import 的"上传 .torrent 文件"入口也靠它算出 hash，好在添加后按 hash
    查回这个种子。
    """

    def _skip(i: int) -> int:
        """返回从 i 开始的 bencode 元素的结束下标（exclusive）。"""
        c = data[i : i + 1]
        if c == b"i":
            return data.index(b"e", i) + 1
        if c in (b"l", b"d"):
            i += 1
            while data[i : i + 1] != b"e":
                i = _skip(i)
            return i + 1
        colon = data.index(b":", i)
        return colon + 1 + int(data[i:colon])

    try:
        if data[:1] != b"d":
            return None
        i = 1
        while data[i : i + 1] != b"e":
            colon = data.index(b":", i)
            key_len = int(data[i:colon])
            key = data[colon + 1 : colon + 1 + key_len]
            i = colon + 1 + key_len
            end = _skip(i)
            if key == b"info":
                if data[i : i + 1] != b"d":
                    return None
                return hashlib.sha1(data[i:end]).hexdigest()
            i = end
        return None
    except (ValueError, IndexError):
        return None


class AddResult(Enum):
    """Torrent-add result shared by concrete clients and the facade."""

    ADDED = "added"
    DUPLICATE = "duplicate"
    FAILED = "failed"


class RenameOutcome(str, Enum):
    """Downloader-independent outcome of a single file rename."""

    RENAMED = "renamed"
    ALREADY_APPLIED = "already_applied"
    DESTINATION_EXISTS = "destination_exists"
    RETRYABLE_FAILURE = "retryable_failure"


@dataclass(frozen=True)
class RenameResult:
    """Structured rename result preserved across the downloader facade.

    ``bool(result)`` deliberately keeps the old success/failure behaviour while
    callers migrate to inspecting :attr:`outcome`.  It must never make a
    collision or retryable failure truthy.
    """

    outcome: RenameOutcome
    detail: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.outcome in {
            RenameOutcome.RENAMED,
            RenameOutcome.ALREADY_APPLIED,
        }

    def __bool__(self) -> bool:
        return self.succeeded


@dataclass(frozen=True)
class DownloaderCapabilities:
    """What a concrete download client can do.

    can_query     -- torrents_info / torrents_files / get_torrents_by_tag
    can_rename    -- torrents_rename_file
    can_manage    -- delete / pause / resume / move / category / tags
    can_rss_rules -- qB-native RSS feeds + auto-download rules + prefs
    """

    can_query: bool
    can_rename: bool
    can_manage: bool
    can_rss_rules: bool


@runtime_checkable
class CoreDownloaderClient(Protocol):
    """The minimum every backend must implement: authenticate and add torrents."""

    capabilities: ClassVar[DownloaderCapabilities]

    async def auth(self, retry: int = 3) -> bool: ...

    async def logout(self) -> None: ...

    async def add_torrents(
        self, torrent_urls, torrent_files, save_path, category, tags=None
    ) -> AddResult: ...


@runtime_checkable
class DownloaderClient(Protocol):
    """The full async surface `DownloadClient` delegates to.

    A backend that satisfies this protocol supports every facade operation. A
    backend that only satisfies `CoreDownloaderClient` is limited to auth and
    adding torrents; the facade guards the rest with `capabilities`.
    """

    capabilities: ClassVar[DownloaderCapabilities]

    # Session lifecycle / connectivity
    async def auth(self, retry: int = 3) -> bool: ...

    async def logout(self) -> None: ...

    async def check_connection(self) -> str: ...

    # Preferences / setup
    async def prefs_init(self, prefs: dict) -> Any: ...

    async def get_app_prefs(self) -> dict: ...

    async def add_category(self, category: str) -> None: ...

    # Torrent lifecycle
    async def add_torrents(
        self, torrent_urls, torrent_files, save_path, category, tags=None
    ) -> AddResult: ...

    async def torrents_info(self, status_filter, category, tag=None) -> list[dict]: ...

    async def torrent_exists(self, torrent_hash: str) -> bool | None: ...

    async def torrents_files(self, torrent_hash: str) -> list[dict]: ...

    async def torrents_delete(self, hash, delete_files: bool = True) -> bool: ...

    async def torrents_pause(self, hashes: str) -> None: ...

    async def torrents_resume(self, hashes: str) -> None: ...

    async def torrents_rename_file(
        self, torrent_hash, old_path, new_path, verify: bool = True
    ) -> RenameResult: ...

    async def move_torrent(self, hashes, new_location) -> None: ...

    async def set_category(self, _hash, category) -> None: ...

    # Tagging
    async def add_tag(self, _hash, tag) -> None: ...

    # RSS auto-download rules
    async def rss_set_rule(self, rule_name, rule_def) -> None: ...
