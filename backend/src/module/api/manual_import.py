import logging
from dataclasses import asdict

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from module.downloader import DownloadClient
from module.manager.manual_import import (
    FileMapping,
    FolderFileMapping,
    apply_folder_import,
    apply_import,
    list_import_candidates,
    preview_folder_import,
    preview_import,
    upload_torrent,
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


class FolderPreviewRequest(BaseModel):
    path: str
    official_title: str
    season: int = 1


class FolderFileMappingIn(BaseModel):
    source_path: str
    target_path: str
    episode: float | None = None
    parsed: bool = True
    kind: str = "media"


class FolderApplyRequest(BaseModel):
    mappings: list[FolderFileMappingIn]


@router.get("/candidates", dependencies=[Depends(get_current_user)])
async def get_candidates():
    """Torrents qBittorrent has that no AutoBangumi subscription/search
    action ever tagged -- added directly through qBittorrent itself."""
    async with DownloadClient() as client:
        candidates = await list_import_candidates(client)
    return [asdict(c) for c in candidates]


@router.post("/upload", dependencies=[Depends(get_current_user)])
async def post_upload(file: UploadFile = File(...)):
    """Upload a .torrent file directly -- added to the downloader untagged
    (same as adding it through the downloader's own UI), returning the
    resulting candidate so the client can proceed straight into /preview,
    exactly like picking one from /candidates."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    async with DownloadClient() as client:
        candidate = await upload_torrent(client, data)
    if candidate is None:
        raise HTTPException(status_code=400, detail="Could not add this .torrent file")
    return asdict(candidate)


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
        "mappings": [asdict(m) for m in preview.mappings],
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


@router.post("/folder/preview", dependencies=[Depends(get_current_user)])
async def post_folder_preview(data: FolderPreviewRequest):
    """Same as /preview, but for a plain filesystem folder AutoBangumi's own
    process can see (e.g. a bind-mounted path), scanned directly instead of
    asking a download client for a torrent's file list."""
    preview = await preview_folder_import(data.path, data.official_title, data.season)
    if preview is None:
        raise HTTPException(status_code=404, detail="Folder not found")
    return {
        "source_root": preview.source_root,
        "official_title": preview.official_title,
        "year": preview.year,
        "tvdb_id": preview.tvdb_id,
        "id_source": preview.id_source,
        "target_folder": preview.target_folder,
        "mappings": [asdict(m) for m in preview.mappings],
        "unparsed": preview.unparsed,
    }


@router.post("/folder/apply", dependencies=[Depends(get_current_user)])
async def post_folder_apply(data: FolderApplyRequest):
    """Apply a (possibly user-edited) folder preview: move+rename each file
    directly on disk. No download client involved."""
    mappings = [
        FolderFileMapping(
            source_path=m.source_path,
            target_path=m.target_path,
            episode=m.episode,
            parsed=m.parsed,
            kind=m.kind,
        )
        for m in data.mappings
    ]
    results = await apply_folder_import(mappings)
    return [
        {
            "source_path": mapping.source_path,
            "target_path": mapping.target_path,
            "succeeded": succeeded,
            "detail": detail,
        }
        for mapping, succeeded, detail in results
    ]
