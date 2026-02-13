from datetime import datetime
import logging
import re
from typing import Annotated, Literal
from urllib.parse import parse_qs

from app.config import settings
from app.database import (
    Beatmap,
    Beatmapset,
    BeatmapsetModel,
    FavouriteBeatmapset,
    SearchBeatmapsetsResp,
    User,
)
from app.database.rhythia_beatmap import RhythiaBeatmap
from app.dependencies.beatmap_download import DownloadService
from app.dependencies.cache import BeatmapsetCacheService, UserCacheService
from app.dependencies.database import Database, Redis
from app.dependencies.fetcher import Fetcher
from app.dependencies.geoip import IPAddress, get_geoip_helper
from app.dependencies.storage import StorageService as StorageServiceDep
from app.dependencies.user import ClientUser, get_current_user
from app.helpers.asset_proxy_helper import asset_proxy_response
from app.models.beatmap import BeatmapRankStatus, SearchQueryModel
from app.models.beatmapset_upload import (
    BeatmapSetFile,
    PutBeatmapSetRequest,
    PutBeatmapSetResponse,
)
from app.models.score import GameMode, Rank
from app.service.beatmapset_cache_service import generate_hash
from app.service.rhythia_service import rhythia_service
from app.service.sspm_import_service import sspm_import_service
from app.utils import api_doc

from .router import router

logger = logging.getLogger(__name__)

# Create a separate router for Rhythia endpoints
from fastapi import APIRouter, Depends, File, Form, HTTPException, Path, Query, Security, UploadFile
from app.service.beatmapset_upload_service import BeatmapsetUploadService

rhythia_router = APIRouter(prefix="/rhythia", tags=["rhythia"])

@rhythia_router.get(
    "/token",
    name="Get Rhythia Token",
    description="Get a dummy token for Rhythia score submission.",
)
async def get_rhythia_token():
    logger.info("Rhythia token endpoint called")
    return {"id": 0}


import inspect
import io
import zipfile

from fastapi import (
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    Request,
    Security,
    UploadFile,
)
from fastapi.responses import RedirectResponse, StreamingResponse
import httpx
from httpx import HTTPError
from sqlmodel import select, col, exists


def _apply_status_override(sets: SearchBeatmapsetsResp) -> None:
    approved_val = BeatmapRankStatus.APPROVED.value
    approved_str = "approved"

    for beatmapset in sets.beatmapsets:
        # beatmapset is a dict (TypedDict)
        if beatmapset.get("ranked") not in (
            BeatmapRankStatus.RANKED,
            BeatmapRankStatus.APPROVED,
        ):
            beatmapset["ranked"] = approved_val
            beatmapset["status"] = approved_str

        for beatmap in beatmapset.get("beatmaps", []):
            if beatmap.get("ranked") not in (
                BeatmapRankStatus.RANKED,
                BeatmapRankStatus.APPROVED,
            ):
                beatmap["ranked"] = approved_val
                beatmap["status"] = approved_str


@router.get(
    "/beatmapsets/search",
    name="搜索谱面集",
    tags=["谱面集"],
    response_model=SearchBeatmapsetsResp,
)
@asset_proxy_response
async def search_beatmapset(
    query: Annotated[SearchQueryModel, Query()],
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: Annotated[User, Security(get_current_user, scopes=["public"])],
    fetcher: Fetcher,
    redis: Redis,
    cache_service: BeatmapsetCacheService,
    db: Database,
):
    params = parse_qs(qs=request.url.query, keep_blank_values=True)
    cursor = {}

    # 解析 cursor[field] 格式的参数
    for k, v in params.items():
        match = re.match(r"cursor\[(\w+)\]", k)
        if match:
            field_name = match.group(1)
            field_value = v[0] if v else None
            if field_value is not None:
                # 转换为适当的类型
                try:
                    if field_name in ["approved_date", "id"]:
                        cursor[field_name] = int(field_value)
                    else:
                        # 尝试转换为数字类型
                        try:
                            # 首先尝试转换为整数
                            cursor[field_name] = int(field_value)
                        except ValueError:
                            try:
                                # 然后尝试转换为浮点数
                                cursor[field_name] = float(field_value)
                            except ValueError:
                                # 最后保持字符串
                                cursor[field_name] = field_value
                except ValueError:
                    cursor[field_name] = field_value

    if query.is_local:
        # Search in local database
        stmt = select(Beatmapset).where(Beatmapset.is_local == True)
        if query.q:
            stmt = stmt.where(
                (col(Beatmapset.title).ilike(f"%{query.q}%")) |
                (col(Beatmapset.artist).ilike(f"%{query.q}%")) |
                (col(Beatmapset.creator).ilike(f"%{query.q}%"))
            )

        if query.m is not None:
            try:
                mode_enum = GameMode.from_int(query.m)
                stmt = stmt.where(exists(select(Beatmap.id).where(Beatmap.beatmapset_id == Beatmapset.id, Beatmap.mode == mode_enum)))
            except KeyError:
                # If mode is not recognized, return no results
                return SearchBeatmapsetsResp(total=0, beatmapsets=[], cursor_string=None)

        db_results = (await db.exec(stmt)).all()

        beatmapsets = []
        for s in db_results:
            # Convert to dict and ensure it matches SearchBeatmapsetsResp expectations
            # Using Beatmapset.transform instead of non-existent to_dict
            s_dict = await Beatmapset.transform(s, session=db)
            beatmapsets.append(s_dict)

        return SearchBeatmapsetsResp(
            total=len(beatmapsets),
            beatmapsets=beatmapsets, # type: ignore
            cursor_string=None
        )

    if (
        "recommended" in query.c
        or len(query.r) > 0
        or query.played
        or "follows" in query.c
        or "mine" in query.s
        or "favourites" in query.s
    ):
        # TODO: search locally
        return SearchBeatmapsetsResp(total=0, beatmapsets=[])

    # 生成查询和游标的哈希用于缓存
    query_hash = generate_hash(query.model_dump())
    cursor_hash = generate_hash(cursor)

    # 尝试从缓存获取搜索结果
    cached_result = await cache_service.get_search_from_cache(query_hash, cursor_hash)
    if cached_result:
        sets = SearchBeatmapsetsResp(**cached_result)
        if settings.enable_all_beatmap_leaderboard:
            _apply_status_override(sets)
        # merge Rhythia results for osu!space mode
        if query.m == 727:
            try:
                rh_res = await rhythia_service.search_beatmaps(
                    page=1,
                    text_filter=query.q or "",
                    status="RANKED",
                )
                beatmaps = rh_res.get("beatmaps", [])
                extra_sets = []
                for m in beatmaps[:10]:
                    map_id = int(m.get("id"))
                    name = m.get("name") or "Unknown"
                    authors = m.get("mappers") or []
                    authors_str = ", ".join(authors) if authors else (m.get("ownerUsername") or "Unknown")
                    title = name
                    artist = "Unknown"
                    if " - " in name:
                        parts = name.split(" - ", 1)
                        artist = parts[0].strip() or artist
                        title = parts[1].strip() or title
                    stars = float(m.get("starRating") or 0)
                    synthetic_id = 800000000 + map_id
                    extra_sets.append(
                        {
                            "id": synthetic_id,
                            "artist": artist,
                            "artist_unicode": artist,
                            "covers": {
                                "cover": f"{settings.web_url}static/default-cover.png",
                                "card": f"{settings.web_url}static/default-cover.png",
                                "list": f"{settings.web_url}static/default-cover.png",
                                "slimcover": f"{settings.web_url}static/default-cover.png",
                            },
                            "creator": authors_str,
                            "nsfw": False,
                            "preview_url": "",
                            "source": "rhythia",
                            "spotlight": False,
                            "title": title,
                            "title_unicode": title,
                            "track_id": None,
                            "user_id": 0,
                            "video": False,
                            "availability": {"more_information": None, "download_disabled": False},
                            "ranked": BeatmapRankStatus.APPROVED.value,
                            "status": "approved",
                            "beatmaps": [
                                {
                                    "id": synthetic_id,
                                    "mode": "osuspaceruleset",
                                    "version": "SSPM",
                                    "difficulty_rating": stars,
                                }
                            ],
                            "pack_tags": [],
                        }  # type: ignore
                    )
                sets.total += len(extra_sets)
                sets.beatmapsets.extend(extra_sets)
            except Exception as e:
                logger.warning(f"Error processing extra beatmapsets: {e}")
        return sets

    try:
        sets = await fetcher.search_beatmapset(query, cursor, redis)

        if settings.enable_all_beatmap_leaderboard:
            _apply_status_override(sets)

        # merge Rhythia results for osu!space mode
        if query.m == 727:
            try:
                rh_res = await rhythia_service.search_beatmaps(
                    page=1,
                    text_filter=query.q or "",
                    status="RANKED",
                )
                beatmaps = rh_res.get("beatmaps", [])
                extra_sets = []
                for m in beatmaps[:10]:
                    map_id = int(m.get("id"))
                    name = m.get("name") or "Unknown"
                    authors = m.get("mappers") or []
                    authors_str = ", ".join(authors) if authors else (m.get("ownerUsername") or "Unknown")
                    title = name
                    artist = "Unknown"
                    if " - " in name:
                        parts = name.split(" - ", 1)
                        artist = parts[0].strip() or artist
                        title = parts[1].strip() or title
                    stars = float(m.get("starRating") or 0)
                    synthetic_id = 800000000 + map_id
                    extra_sets.append(
                        {
                            "id": synthetic_id,
                            "artist": artist,
                            "artist_unicode": artist,
                            "covers": {
                                "cover": f"{settings.web_url}static/default-cover.png",
                                "card": f"{settings.web_url}static/default-cover.png",
                                "list": f"{settings.web_url}static/default-cover.png",
                                "slimcover": f"{settings.web_url}static/default-cover.png",
                            },
                            "creator": authors_str,
                            "nsfw": False,
                            "preview_url": "",
                            "source": "rhythia",
                            "spotlight": False,
                            "title": title,
                            "title_unicode": title,
                            "track_id": None,
                            "user_id": 0,
                            "video": False,
                            "availability": {"more_information": None, "download_disabled": False},
                            "ranked": BeatmapRankStatus.APPROVED.value,
                            "status": "approved",
                            "beatmaps": [
                                {
                                    "id": synthetic_id,
                                    "mode": "osuspaceruleset",
                                    "version": "SSPM",
                                    "difficulty_rating": stars,
                                }
                            ],
                            "pack_tags": [],
                        }  # type: ignore
                    )
                sets.total += len(extra_sets)
                sets.beatmapsets.extend(extra_sets)
            except Exception as e:
                logger.warning(f"Error processing extra beatmapsets: {e}")

        # 缓存搜索结果
        await cache_service.cache_search_result(query_hash, cursor_hash, sets.model_dump())
        return sets
    except HTTPError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get(
    "/beatmapsets/lookup",
    tags=["谱面集"],
    responses={200: api_doc("谱面集详细信息", BeatmapsetModel, BeatmapsetModel.BEATMAPSET_TRANSFORMER_INCLUDES)},
    name="查询谱面集 (通过谱面 ID)",
    description=("通过谱面 ID 查询所属谱面集。"),
)
@asset_proxy_response
async def lookup_beatmapset(
    db: Database,
    request: Request,
    beatmap_id: Annotated[int, Query(description="谱面 ID")],
    current_user: Annotated[User, Security(get_current_user, scopes=["public"])],
    fetcher: Fetcher,
    cache_service: BeatmapsetCacheService,
):
    # 先尝试从缓存获取
    cached_resp = await cache_service.get_beatmap_lookup_from_cache(beatmap_id)
    if cached_resp:
        return cached_resp

    try:
        beatmap = await Beatmap.get_or_fetch(db, fetcher, bid=beatmap_id)

        resp = await BeatmapsetModel.transform(
            beatmap.beatmapset, user=current_user, includes=BeatmapsetModel.API_INCLUDES
        )

        # 缓存结果
        await cache_service.cache_beatmap_lookup(beatmap_id, resp)
        return resp
    except HTTPError as exc:
        raise HTTPException(status_code=404, detail="Beatmap not found") from exc


@router.get(
    "/beatmapsets/{beatmapset_id}",
    tags=["谱面集"],
    responses={200: api_doc("谱面集详细信息", BeatmapsetModel, BeatmapsetModel.BEATMAPSET_TRANSFORMER_INCLUDES)},
    name="获取谱面集详情",
    description="获取单个谱面集详情。",
)
@asset_proxy_response
async def get_beatmapset(
    db: Database,
    request: Request,
    beatmapset_id: Annotated[int, Path(..., description="谱面集 ID")],
    current_user: Annotated[User, Security(get_current_user, scopes=["public"])],
    fetcher: Fetcher,
    cache_service: BeatmapsetCacheService,
):
    # 先尝试从缓存获取
    cached_resp = await cache_service.get_beatmapset_from_cache(beatmapset_id)
    if cached_resp:
        return cached_resp

    try:
        beatmapset = await Beatmapset.get_or_fetch(db, fetcher, beatmapset_id)
        await db.refresh(current_user)
        resp = await BeatmapsetModel.transform(beatmapset, includes=BeatmapsetModel.API_INCLUDES, user=current_user)

        # 缓存结果
        await cache_service.cache_beatmapset(resp)
        return resp
    except HTTPError as exc:
        raise HTTPException(status_code=404, detail="Beatmapset not found") from exc


@router.get(
    "/beatmapsets/{beatmapset_id}/download",
    tags=["谱面集"],
    name="下载谱面集",
    description="\n下载谱面集文件。基于请求IP地理位置智能分流，支持负载均衡和自动故障转移。中国IP使用Sayobot镜像，其他地区使用Nerinyan和OsuDirect镜像。",
)
async def download_beatmapset(
    client_ip: IPAddress,
    beatmapset_id: Annotated[int, Path(..., description="谱面集 ID")],
    current_user: ClientUser,
    download_service: DownloadService,
    storage: StorageServiceDep,
    no_video: Annotated[bool, Query(alias="noVideo", description="是否下载无视频版本")] = True,
):
    geoip_helper = get_geoip_helper()
    geo_info = geoip_helper.lookup(client_ip)
    country_code = geo_info.get("country_iso", "")

    # 优先使用IP地理位置判断，如果获取失败则回退到用户账户的国家代码
    is_china = country_code == "CN" or (not country_code and current_user.country_code == "CN")

    # 优先使用本地存储，如果文件存在
    file_path = f"beatmapsets/{beatmapset_id}.osz"
    if await storage.is_exists(file_path):
        from fastapi.responses import FileResponse
        from app.dependencies.storage import get_storage_service
        storage_service = get_storage_service()
        if hasattr(storage_service, "storage_path"):
            full_path = storage_service.storage_path / file_path
            return FileResponse(
                path=full_path,
                filename=f"{beatmapset_id}.osz",
                media_type="application/octet-stream",
            )

    try:
        # 使用负载均衡服务获取所有备选下载URL
        download_urls = download_service.get_download_urls(
            beatmapset_id=beatmapset_id, no_video=no_video, is_china=is_china
        )

        if not download_urls:
            raise HTTPException(status_code=503, detail="No download URLs available")

        # 为了实现真正的“重试”，我们采用 StreamingResponse 方式代理下载
        # 这样当一个镜像下载失败（如 404 或超时）时，我们可以自动尝试下一个镜像
        async def iterate_mirrors():
            async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
                for url in download_urls:
                    try:
                        logger.debug(f"Trying to download beatmapset {beatmapset_id} from {url}")
                        async with client.stream("GET", url) as response:
                            if response.status_code == 200:
                                # 检查是否是有效的 .osz 文件（简单检查 Content-Type 或内容前缀）
                                # 有些镜像在 404 时可能返回 HTML 页面但状态码是 200
                                content_type = response.headers.get("Content-Type", "")
                                if "text/html" in content_type:
                                    logger.warning(f"Mirror {url} returned HTML instead of .osz, trying next...")
                                    continue

                                async for chunk in response.aiter_bytes():
                                    yield chunk
                                return  # 成功完成下载，退出循环
                            else:
                                logger.warning(f"Mirror {url} failed with status {response.status_code}, trying next...")
                    except Exception as e:
                        logger.warning(f"Failed to download from mirror {url}: {e}, trying next...")
                        continue

                # 如果所有镜像都失败了
                logger.error(f"All download mirrors failed for beatmapset {beatmapset_id}")
                # 注意：在生成器中抛出异常可能无法被 FastAPI 正常捕获并返回特定的 HTTP 状态码
                # 但这里是最后的手段

        return StreamingResponse(
            iterate_mirrors(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{beatmapset_id}.osz"'},
        )
    except Exception as e:
        logger.error(f"Download service failed: {e}")
        # 如果负载均衡服务彻底失败，回退到原有重定向逻辑（作为最后保障）
        if is_china:
            return RedirectResponse(
                f"https://dl.sayobot.cn/beatmaps/download/{'novideo' if no_video else 'full'}/{beatmapset_id}"
            )
        else:
            return RedirectResponse(f"https://catboy.best/d/{beatmapset_id}{'n' if no_video else ''}")


@router.post(
    "/beatmapsets/{beatmapset_id}/favourites",
    tags=["谱面集"],
    name="收藏或取消收藏谱面集",
    description="\n收藏或取消收藏指定谱面集。",
)
async def favourite_beatmapset(
    db: Database,
    cache_service: UserCacheService,
    beatmapset_id: Annotated[int, Path(..., description="谱面集 ID")],
    action: Annotated[
        Literal["favourite", "unfavourite"],
        Form(description="操作类型：favourite 收藏 / unfavourite 取消收藏"),
    ],
    current_user: ClientUser,
):
    existing_favourite = (
        await db.exec(
            select(FavouriteBeatmapset).where(
                FavouriteBeatmapset.user_id == current_user.id,
                FavouriteBeatmapset.beatmapset_id == beatmapset_id,
            )
        )
    ).first()

    if (action == "favourite" and existing_favourite) or (action == "unfavourite" and not existing_favourite):
        return

    if action == "favourite":
        favourite = FavouriteBeatmapset(user_id=current_user.id, beatmapset_id=beatmapset_id)
        db.add(favourite)
    else:
        await db.delete(existing_favourite)
    await cache_service.invalidate_user_beatmapsets_cache(current_user.id)
    await db.commit()


@router.put(
    "/beatmapsets",
    response_model=PutBeatmapSetResponse,
    tags=["谱面集"],
    name="初始化谱面上传",
    description="初始化谱面集上传流程，返回谱面集 ID 和现有文件列表。",
)
async def initialize_beatmapset_upload(
    db: Database,
    storage: StorageServiceDep,
    req: PutBeatmapSetRequest,
    current_user: ClientUser,
):
    if req.beatmapset_id:
        beatmapset = await db.get(Beatmapset, req.beatmapset_id)
        if not beatmapset:
            # Create a temporary beatmapset with the provided ID
            beatmapset = Beatmapset(
                id=req.beatmapset_id,
                artist="Unknown",
                artist_unicode="Unknown",
                title="Unknown",
                title_unicode="Unknown",
                creator=current_user.username,
                user_id=current_user.id,
                video=False,
                is_local=True,
                submitted_date=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                beatmap_status=BeatmapRankStatus.PENDING if req.target == "Pending" else BeatmapRankStatus.WIP,
            )
            db.add(beatmapset)
            await db.flush()
        elif beatmapset.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="You do not own this beatmapset")

        # Delete beatmaps not in beatmaps_to_keep
        # If beatmaps_to_keep is empty, it means all existing beatmaps should be kept unless explicitly replaced?
        # Actually, osu!Game sends the full list of online IDs it wants to keep.
        if req.beatmaps_to_keep:
            from sqlmodel import col
            stmt = select(Beatmap).where(
                Beatmap.beatmapset_id == beatmapset.id,
                ~col(Beatmap.id).in_(req.beatmaps_to_keep)
            )
            to_delete = (await db.exec(stmt)).all()
            for b in to_delete:
                await db.delete(b)

        # Update status if target is provided
        if req.target == "Pending":
            beatmapset.beatmap_status = BeatmapRankStatus.PENDING
        else:
            beatmapset.beatmap_status = BeatmapRankStatus.WIP

        existing_files = await BeatmapsetUploadService.get_beatmapset_files(storage, beatmapset.id)

        # Allocate new beatmaps if needed
        beatmap_ids = []
        if req.beatmaps_to_create > 0:
            beatmap_ids = await BeatmapsetUploadService.allocate_beatmaps(
                db, beatmapset.id, current_user.id, req.beatmaps_to_create
            )

        await db.commit()
    else:
        beatmapset = Beatmapset(
            artist="Unknown",
            artist_unicode="Unknown",
            title="Unknown",
            title_unicode="Unknown",
            creator=current_user.username,
            user_id=current_user.id,
            video=False,
            is_local=True,
            submitted_date=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            beatmap_status=BeatmapRankStatus.PENDING if req.target == "Pending" else BeatmapRankStatus.WIP,
        )
        db.add(beatmapset)
        await db.flush()

        existing_files = []
        beatmap_ids = await BeatmapsetUploadService.allocate_beatmaps(
            db, beatmapset.id, current_user.id, req.beatmaps_to_create
        )
        await db.commit()
        await db.refresh(beatmapset)

    return PutBeatmapSetResponse(
        beatmapset_id=beatmapset.id,
        beatmap_ids=beatmap_ids,
        files=existing_files,
    )


@router.put(
    "/beatmapsets/{beatmapset_id}",
    tags=["谱面集"],
    name="上传完整谱面集",
    description="上传完整的 .osz 谱面集文件。",
    response_model=None,
)
async def upload_beatmapset_package(
    db: Database,
    storage: StorageServiceDep,
    cache_service: BeatmapsetCacheService,
    beatmapset_id: Annotated[int, Path(..., description="谱面集 ID")],
    beatmapArchive: Annotated[UploadFile, File(description="OSZ 文件")],
    current_user: ClientUser,
):
    beatmapset = await db.get(Beatmapset, beatmapset_id)
    if not beatmapset:
        raise HTTPException(status_code=404, detail="Beatmapset not found")
    if beatmapset.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="You do not own this beatmapset")

    content = await beatmapArchive.read()
    await storage.write_file(f"beatmapsets/{beatmapset_id}.osz", content)

    updated_ids = await BeatmapsetUploadService.process_beatmapset_package(db, storage, beatmapset_id)

    # Invalidate caches
    await cache_service.invalidate_beatmapset_cache(beatmapset_id)
    for bid in updated_ids:
        await cache_service.invalidate_beatmap_lookup_cache(bid)

    return {"status": "success"}


@router.patch(
    "/beatmapsets/{beatmapset_id}",
    tags=["谱面集"],
    name="增量更新谱面集",
    description="增量上传修改的文件或删除文件。",
    response_model=None,
)
async def patch_beatmapset_package(
    db: Database,
    storage: StorageServiceDep,
    cache_service: BeatmapsetCacheService,
    beatmapset_id: Annotated[int, Path(..., description="谱面集 ID")],
    current_user: Annotated[User, Security(get_current_user, scopes=["public"])],
    filesChanged: Annotated[list[UploadFile] | None, File()] = None,
    filesDeleted: Annotated[list[str] | None, Form()] = None,
):
    beatmapset = await db.get(Beatmapset, beatmapset_id)
    if not beatmapset:
        raise HTTPException(status_code=404, detail="Beatmapset not found")
    if beatmapset.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="You do not own this beatmapset")

    changed = []
    if filesChanged:
        for f in filesChanged:
            changed.append((f.filename, await f.read()))

    deleted = filesDeleted or []

    await BeatmapsetUploadService.patch_beatmapset_package(storage, beatmapset_id, changed, deleted)

    updated_ids = await BeatmapsetUploadService.process_beatmapset_package(db, storage, beatmapset_id)

    # Invalidate caches
    await cache_service.invalidate_beatmapset_cache(beatmapset_id)
    for bid in updated_ids:
        await cache_service.invalidate_beatmap_lookup_cache(bid)

    return {"status": "success"}

@rhythia_router.get(
    "/search",
    name="Search Rhythia Maps",
    description="Search for beatmaps on Rhythia servers.",
)
async def search_rhythia_maps(
    query: Annotated[str, Query(description="Search query")] = "",
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    status: Annotated[str, Query(description="Map status (RANKED, UNRANKED, etc.)")] = "RANKED",
):
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"search_rhythia_maps called with query='{query}', page={page}, status='{status}'")
    try:
        results = await rhythia_service.search_beatmaps(
            page=page,
            text_filter=query,
            status=status,
        )
        result_count = len(results) if isinstance(results, list) else "non-list"
        logger.info(f"search_rhythia_maps returning {result_count} results")
        if isinstance(results, dict):
            beatmaps = results.get("beatmaps", [])
            if "docs" in results:
                return results
            docs = []
            for bm in beatmaps:
                if not isinstance(bm, dict):
                    continue
                docs.append(
                    {
                        "id": bm.get("id"),
                        "beatmap": bm,
                    }
                )
            total = results.get("total", len(docs))
            return {
                "docs": docs,
                "total": total,
            }

        if isinstance(results, list):
            docs = []
            for bm in results:
                if not isinstance(bm, dict):
                    continue
                docs.append(
                    {
                        "id": bm.get("id"),
                        "beatmap": bm,
                    }
                )
            return {
                "docs": docs,
                "total": len(docs),
            }

        return results
    except Exception as e:
        logger.error(f"search_rhythia_maps error: {e}")
        raise


@rhythia_router.get(
    "/maps/{map_id}",
    name="Get Rhythia Map Details",
    description="Get details of a specific Rhythia beatmap.",
)
async def get_rhythia_map_details(map_id: int):
    try:
        details = await rhythia_service.get_beatmap_details(map_id)
        if not details or "beatmap" not in details:
             raise HTTPException(status_code=404, detail="Map not found")
        return details
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@rhythia_router.get(
    "/download/{map_id}",
    name="Download Rhythia Map",
    description="Download a Rhythia beatmap file. Redirects to the storage URL.",
)
async def download_rhythia_map(map_id: int):
    try:
        details = await rhythia_service.get_beatmap_details(map_id)
        if not details or "beatmap" not in details:
             raise HTTPException(status_code=404, detail="Map not found")

        beatmap_info = details["beatmap"]
        file_url = beatmap_info.get("beatmapFile")

        if not file_url:
            raise HTTPException(status_code=404, detail="Download URL not found for this map")

        return RedirectResponse(url=file_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/sspm/maps/{map_id}",
    name="Download stored SSPM map",
    description="Download a locally stored SSPM .osz/.osu file for a given map id.",
)
async def download_stored_sspm_map(
    map_id: int,
    storage: StorageServiceDep,
):
    # Prefer raw .osu for calculators; fallback to .osz archive
    exts = [".osu", ".osz"]
    for ext in exts:
        storage_key = f"sspm/{map_id}{ext}"
        try:
            content = await storage.read_file(storage_key)
            media_type = "application/octet-stream"
            return StreamingResponse(
                io.BytesIO(content),
                media_type=media_type,
                headers={
                    "Content-Disposition": f'attachment; filename="{map_id}{ext}"',
                },
            )
        except FileNotFoundError:
            continue
    raise HTTPException(status_code=404, detail="Stored SSPM file not found")

@router.post(
    "/maps/import",
    name="Import SSPM Map",
    description="Upload an .osu or .osz to register an SSPM map locally.",
)
async def import_sspm_map(
    db: Database,
    storage: StorageServiceDep,
    file: UploadFile = File(...),
):
    try:
        content = await file.read()
        filename = file.filename or "upload"

        def parse_osu(osu_text: str) -> dict:
            meta = {
                "title": "Unknown",
                "artist": "Unknown",
                "creator": "Unknown",
                "version": "Standard",
                "sspm_identifier": None
            }
            # Basic validation
            if "osuspaceruleset file format v1" not in osu_text:
                raise HTTPException(status_code=422, detail="Not an SSPM osu file")
            # Extract metadata section
            for line in osu_text.splitlines():
                if line.startswith("Title:"):
                    meta["title"] = line.split(":", 1)[1].strip()
                elif line.startswith("Artist:"):
                    meta["artist"] = line.split(":", 1)[1].strip()
                elif line.startswith("Creator:"):
                    meta["creator"] = line.split(":", 1)[1].strip()
                elif line.startswith("Version:"):
                    meta["version"] = line.split(":", 1)[1].strip()
                elif line.startswith("Tags:"):
                    # find 'sspm <identifier>'
                    m = re.search(r"sspm\s+([^\s]+)", line, re.IGNORECASE)
                    if m:
                        meta["sspm_identifier"] = m.group(1)
            if meta["sspm_identifier"] is None:
                raise HTTPException(status_code=422, detail="SSPM identifier not found in Tags")
            return meta

        osu_text: str | None = None
        if filename.lower().endswith(".osu"):
            try:
                osu_text = content.decode("utf-8", errors="ignore")
            except Exception:
                raise HTTPException(status_code=400, detail="Failed to decode .osu file")
        elif filename.lower().endswith(".osz"):
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    osu_names = [n for n in zf.namelist() if n.lower().endswith(".osu")]
                    if not osu_names:
                        raise HTTPException(status_code=422, detail=".osz does not contain .osu")
                    with zf.open(osu_names[0], "r") as fh:
                        osu_text = fh.read().decode("utf-8", errors="ignore")
            except zipfile.BadZipFile:
                raise HTTPException(status_code=400, detail="Invalid .osz archive")
        else:
            raise HTTPException(status_code=415, detail="Unsupported file type, expected .osu or .osz")

        meta = parse_osu(osu_text or "")
        identifier = meta["sspm_identifier"] or ""

        # numeric id if possible, otherwise generate deterministic integer from identifier
        try:
            map_id = int(identifier)
        except ValueError:
            import hashlib
            digest = hashlib.sha256(identifier.encode("utf-8")).digest()
            map_id = int.from_bytes(digest[:8], "big") & 0x7fffffff
            # ensure uniqueness if collision
            existing = await db.get(RhythiaBeatmap, map_id)
            if existing and existing.sspm_identifier != identifier:
                map_id ^= 0x9e3779b9  # mix with golden ratio constant

        # Upsert
        existing = await db.get(RhythiaBeatmap, map_id)
        if existing:
            existing.title = meta["title"]
            existing.artist = meta["artist"]
            existing.version = meta["version"]
            existing.is_sspm = True
            existing.sspm_identifier = identifier
            beatmap = existing
        else:
            beatmap = RhythiaBeatmap(
                id=map_id,
                title=meta["title"],
                artist=meta["artist"],
                version=meta["version"],
                mode="SPACE",
                is_sspm=True,
                sspm_identifier=identifier,
            )
            db.add(beatmap)

        await db.commit()
        await db.refresh(beatmap)

        ext = ".osz" if filename.lower().endswith(".osz") else ".osu"
        storage_key = f"sspm/{beatmap.id}{ext}"
        await storage.write_file(storage_key, content)

        return {
            "success": True,
            "beatmap_id": beatmap.id,
            "sspm_identifier": beatmap.sspm_identifier,
            "title": beatmap.title,
            "artist": beatmap.artist,
            "version": beatmap.version,
            "message": "SSPM map imported and registered"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@rhythia_router.post(
    "/maps/{map_id}/register",
    name="Register Rhythia Map",
    description="Register a Rhythia beatmap in the local database for leaderboard tracking.",
)
async def register_rhythia_map(
    map_id: int,
    db: Database,
    storage: StorageServiceDep,
    redis: Redis,
):
        details = await rhythia_service.get_beatmap_details(map_id)
        if not details or "beatmap" not in details:
            raise HTTPException(status_code=404, detail="Map not found on Rhythia")

        beatmap_data = details["beatmap"]

        if "map_id" in inspect.signature(rhythia_service.register_beatmap).parameters:
            registered_beatmap = await rhythia_service.register_beatmap(db, beatmap_data, map_id=map_id)
        else:
            beatmap_data["id"] = map_id
            registered_beatmap = await rhythia_service.register_beatmap(db, beatmap_data)

        # Ensure file downloaded & stored (idempotent)
        try:
            await sspm_import_service.ensure_import(db, storage, map_id, redis)
        except Exception as e:
            logger.warning(f"Failed to ensure SSPM file for {map_id}: {e}")

        return {
            "success": True,
            "beatmap_id": registered_beatmap.id,
            "message": f"Rhythia beatmap {map_id} registered successfully"
        }



@rhythia_router.get(
    "/maps/{map_id}/leaderboard",
    name="Get Rhythia Map Leaderboard",
    description="Get the leaderboard for a specific Rhythia beatmap.",
)
async def get_rhythia_leaderboard(
    map_id: int,
    db: Database,
    mode: Annotated[str, Query(description="Game mode (osu, taiko, fruits, mania)")] = "osu",
    limit: Annotated[int, Query(ge=1, le=100, description="Number of scores to return")] = 50,
):
    # Verify beatmap is registered
    beatmap = await db.get(RhythiaBeatmap, map_id)
    if not beatmap:
        # Auto-register if not found
        try:
            details = await rhythia_service.get_beatmap_details(map_id)
            if details and "beatmap" in details:
                beatmap_data = details["beatmap"]
                await rhythia_service.register_beatmap(db, beatmap_data, map_id=map_id)
            else:
                raise HTTPException(
                    status_code=404,
                    detail="Rhythia beatmap not registered in local database and not found on Rhythia"
                )
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Rhythia beatmap not registered and failed to fetch: {e!s}")

    # Get leaderboard
    scores = await rhythia_service.get_leaderboard(db, map_id, mode, limit)

    return {
        "beatmap_id": map_id,
        "mode": mode,
        "scores": scores,
        "total_scores": len(scores)
    }


from pydantic import BaseModel


class RhythiaScoreSubmission(BaseModel):
    user_id: int
    score: int
    max_combo: int
    accuracy: float
    rank: Rank  # Uses the Rank enum from app.models.score
    mode: GameMode  # Uses GameMode enum from app.models.score
    mode_int: int
    mods: list[str]
    statistics: dict[str, int]
    pp: float | None = None

@rhythia_router.post(
    "/maps/{map_id}/scores",
    name="Submit Rhythia Score",
    description="Submit a score for a Rhythia beatmap.",
)
async def submit_rhythia_score(
    map_id: int,
    db: Database,
    redis: Redis,
    fetcher: Fetcher,
    storage: StorageServiceDep,
    submission: RhythiaScoreSubmission,
):
    try:
        # Debug: Log the raw submission data
        logger.info("=== RHythia Score Submission Debug ===")
        logger.info(f"Raw submission: {submission}")
        logger.info(f"Raw submission model_dump: {submission.model_dump()}")
        logger.info(f"Submission user_id: {submission.user_id}, mode_int: {submission.mode_int}")
        logger.info(f"Submission mode: {submission.mode}, type: {type(submission.mode)}")

        # Validate submission data
        if submission.user_id == 0:
            logger.warning("Received submission with user_id=0, this may indicate client issue")
        if submission.mode_int == 0 and submission.mode == GameMode.SPACE:
            logger.warning("Received submission with mode_int=0 for SPACE mode, should be 727")
            # Auto-correct mode_int for SPACE mode
            submission.mode_int = 727
            logger.info("Auto-corrected mode_int to 727 for SPACE mode")

        score_data = submission.model_dump(exclude={"user_id"})
        logger.info(f"score_data after excluding user_id: {score_data}")

        # Ensure mode_int is included explicitly and mode is converted to string
        if "mode_int" not in score_data:
            score_data["mode_int"] = submission.mode_int
            logger.info(f"Added mode_int to score_data: {score_data['mode_int']}")

        # Convert GameMode enum to string (if not already converted)
        if "mode" in score_data and isinstance(score_data["mode"], GameMode):
            logger.info(f"Converting mode enum to string: {score_data['mode']} -> {score_data['mode']!s}")
            score_data["mode"] = str(score_data["mode"])

        # Map long ruleset names to short database-compatible names
        if score_data.get("mode") == "osuspaceruleset":
            score_data["mode"] = "space"
            logger.info("Mapped mode 'osuspaceruleset' to 'space' for database compatibility")

        # Add timestamps which are not in the submission model
        score_data["created_at"] = datetime.now()
        score_data["updated_at"] = datetime.now()

        # Debug: Log the final processed score data
        logger.info(f"Final processed score_data: {score_data}")
        logger.info(f"Calling submit_score with user_id: {submission.user_id}")

        # Ensure Rhythia beatmap exists before inserting score (FK requirement)
        try:
            details = await rhythia_service.get_beatmap_details(map_id)
            beatmap_info = details.get("beatmap") if isinstance(details, dict) else None
            if beatmap_info:
                await rhythia_service.register_beatmap(db, beatmap_info, map_id=map_id)
        except Exception as e:
            logger.warning(f"Failed to register Rhythia beatmap {map_id} prior to score submission: {e}")

        # Ensure SSPM file exists for PP/difficulty calculation and downloader
        try:
            await sspm_import_service.ensure_import(db, storage, map_id, redis)
        except Exception as e:
            logger.warning(f"Failed to ensure SSPM file for {map_id}: {e}")

        submitted_score = await rhythia_service.submit_score(db, map_id, submission.user_id, score_data)

        # Process PP calculation for the submitted score
        await rhythia_service.process_rhythia_score_pp(submitted_score, db, redis, fetcher)

        return {
            "success": True,
            "score_id": submitted_score.id,
            "message": f"Score submitted successfully for Rhythia beatmap {map_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@rhythia_router.get(
    "/users/{user_id}/best-scores",
    name="Get User Best Rhythia Scores",
    description="Get a user's best scores on Rhythia beatmaps.",
)
async def get_user_best_rhythia_scores(
    user_id: int,
    db: Database,
    mode: Annotated[str, Query(description="Game mode (osu, taiko, fruits, mania)")] = "osu",
    limit: Annotated[int, Query(ge=1, le=100, description="Number of scores to return")] = 50,

):
    scores = await rhythia_service.get_user_best_scores(db, user_id, mode, limit)
    return {
        "user_id": user_id,
        "mode": mode,
        "scores": scores,
        "total_scores": len(scores)
    }

# Include the Rhythia router in the main router
router.include_router(rhythia_router)
