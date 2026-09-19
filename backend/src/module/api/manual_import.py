import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from module.downloader import DownloadClient
from module.manager.manual_import import (
    FileMapping,
    apply_import,
    list_import_candidates,
    preview_import,
)
from module.security.api import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/manual-import", tags=["manual-import"])


class PreviewRequest(BaseModel):
    torrent_hash: str
    official_title: str
    season: int = 1


class FileMappingIn(BaseModel):
    source_path: str
    target_path: str
    episode: float | None = None
    parsed: bool = True
    kind: str = "media"


class ApplyRequest(BaseModel):
    torrent_hash: str
    target_folder: str
    mappings: list[FileMappingIn]


@router.get("/candidates", dependencies=[Depends(get_current_user)])
async def get_candidates():
    """Torrents qBittorrent has that no AutoBangumi subscription/search
    action ever tagged -- added directly through qBittorrent itself."""
    async with DownloadClient() as client:
        candidates = await list_import_candidates(client)
    return [c.__dict__ for c in candidates]


@router.post("/preview", dependencies=[Depends(get_current_user)])
async def post_preview(data: PreviewRequest):
    """Resolve the destination folder and propose a per-file rename+move
    without touching anything -- confirm via /apply once it looks right."""
    async with DownloadClient() as client:
        preview = await preview_import(
            client, data.torrent_hash, data.official_title, data.season
        )
    if preview is None:
        raise HTTPException(status_code=404, detail="Torrent not found")
    return {
        "torrent_hash": preview.torrent_hash,
        "official_title": preview.official_title,
        "year": preview.year,
        "tvdb_id": preview.tvdb_id,
        "id_source": preview.id_source,
        "target_folder": preview.target_folder,
        "mappings": [m.__dict__ for m in preview.mappings],
        "unparsed": preview.unparsed,
    }


@router.post("/apply", dependencies=[Depends(get_current_user)])
async def post_apply(data: ApplyRequest):
    """Apply a (possibly user-edited) preview: move the torrent's content to
    target_folder, then rename each mapped file within it."""
    mappings = [
        FileMapping(
            source_path=m.source_path,
            target_path=m.target_path,
            episode=m.episode,
            parsed=m.parsed,
            kind=m.kind,
        )
        for m in data.mappings
    ]
    async with DownloadClient() as client:
        results = await apply_import(
            client, data.torrent_hash, data.target_folder, mappings
        )
    return [
        {
            "source_path": mapping.source_path,
            "target_path": mapping.target_path,
            "succeeded": succeeded,
            "detail": detail,
        }
        for mapping, succeeded, detail in results
    ]
