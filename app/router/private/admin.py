from __future__ import annotations
from datetime import datetime, timedelta
from typing import Annotated, cast, Any

from app.database.auth import OAuthToken
from app.database.announcement import Announcement, AnnouncementCreate, AnnouncementUpdate, AnnouncementResponse, AnnouncementType
from app.database.audit_log import AuditLog, AuditLogCreate, AuditLogResponse, AuditActionType, TargetType
from app.database.beatmap import Beatmap, BannedBeatmaps
from app.database.beatmapset import Beatmapset
from app.database.chat import ChannelType, ChatChannel, ChatMessage, ChatMessageModel, MessageType
from app.database.client_log import ClientLog, ClientLogCreate, ClientLogResponse, ClientLogType
from app.database.rank_request import RankRequest, RankRequestCreate, RankRequestUpdate, RankRequestResponse, RankRequestStatus
from app.database.report import Report, ReportCreate, ReportUpdate, ReportResponse, ReportStatus, ReportType
from app.database.score import Score
from app.database.score_token import ScoreToken
from app.database.statistics import UserStatistics
from app.database.system_settings import SystemSetting, RecalculationTask, RecalculationTaskCreate, RecalculationTaskResponse, RecalculationStatus, RecalculationType
from app.database.user_login_log import UserLoginLog
from app.database.daily_challenge_model import DailyChallenge, DailyChallengeCreate, DailyChallengeUpdate, DailyChallengeResponse
from app.database.team import Team, TeamMember
from app.database.user import User
from app.database.user_account_history import UserAccountHistory, UserAccountHistoryType
from app.database.user_badge import UserBadge, UserBadgeCreate, UserBadgeUpdate, UserBadgeResponse
from app.database.verification import LoginSession, LoginSessionResp, TrustedDevice, TrustedDeviceResp
from app.const import BANCHOBOT_ID
from app.dependencies.database import Database, get_redis
from app.dependencies.client_verification import ClientVerificationService
from app.dependencies.geoip import GeoIPService, IPAddress
from app.dependencies.storage import StorageService
from app.dependencies.user import UserAndToken, get_client_user_and_token
from app.models.mods import APIMod, get_available_mods
from app.models.score import GameMode
from app.models.notification import ChannelMessage, GlobalAnnouncement, UserAchievementUnlock
from app.router.notification.server import server
from app.service.ranking_cache_service import get_ranking_cache_service
from app.service.recalculation_service import is_recalculation_running, has_pending_or_running_task, check_concurrent_limit, get_current_task_status
from app.tasks.recalculation_worker import process_pending_recalculation_tasks
from app.tasks.daily_challenge import create_daily_challenge_room
from app.utils import check_image, utcnow

from fastapi import APIRouter

router = APIRouter()

import json
import httpx
import hashlib
import time
from fastapi import File, Form, HTTPException, Query, Security
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy import or_ as sql_or
from sqlmodel import col, func, select
from app.log import log

logger = log("AdminAPI")

async def require_admin(session: Database, user_and_token: UserAndToken) -> User:
    """Helper function to check if user is admin"""
    current_user, _ = user_and_token
    # is_admin is a simple boolean field, no need to await
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


async def require_dev_or_admin(session: Database, user_and_token: UserAndToken) -> User:
    """Helper function to check if user is admin OR has dev role"""
    current_user, _ = user_and_token
    if current_user.is_admin:
        return current_user

    # Check dev role - is_dev is also a simple boolean field
    if current_user.is_dev:
        return current_user

    raise HTTPException(status_code=403, detail="Dev or admin access required")


async def sync_daily_challenge_to_redis(
    challenge_date: datetime.date,
    beatmap_id: int,
    ruleset_id: int,
    required_mods: str,
    allowed_mods: str,
    room_id: int | None = None,
) -> int | None:
    """Sync daily challenge data to Redis and optionally create a room for today's challenge.

    Returns the room_id if a room was created, None otherwise.
    """
    redis = get_redis()
    redis_key = f"daily_challenge:{challenge_date}"

    await redis.hset(
        redis_key,
        mapping={
            "beatmap": beatmap_id,
            "ruleset_id": ruleset_id,
            "required_mods": required_mods,
            "allowed_mods": allowed_mods,
        },
    )

    # Automatically create room if for today and no room_id provided
    if room_id is None and challenge_date == utcnow().date():
        now = utcnow()
        next_day = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        duration = max(0, min(1440, int((next_day - now).total_seconds() / 60)))

        required_mods_list = json.loads(required_mods)
        allowed_mods_list = json.loads(allowed_mods)

        room = await create_daily_challenge_room(
            beatmap=beatmap_id,
            ruleset_id=ruleset_id,
            duration=duration,
            required_mods=cast(list[APIMod], required_mods_list),
            allowed_mods=cast(list[APIMod], allowed_mods_list),
        )
        return room.id

    return None


async def get_beatmap_info_for_response(session: Database, beatmap_id: int) -> dict[str, Any] | None:
    """Get beatmap info for daily challenge response, with error handling for missing beatmaps."""
    beatmap = await session.get(Beatmap, beatmap_id)
    if not beatmap:
        return {
            "id": beatmap_id,
            "title": "Unknown (Beatmap not found)",
            "artist": "Unknown",
            "difficulty_rating": 0.0,
        }

    try:
        beatmapset = await beatmap.awaitable_attrs.beatmapset
        return {
            "id": beatmap.id,
            "title": beatmapset.title if beatmapset else "Unknown",
            "artist": beatmapset.artist if beatmapset else "Unknown",
            "difficulty_rating": beatmap.difficulty_rating,
        }
    except Exception:
        return {
            "id": beatmap.id,
            "title": "Unknown",
            "artist": "Unknown",
            "difficulty_rating": beatmap.difficulty_rating,
        }


async def user_to_dict(user: User, session: Database) -> dict:
    """Convert User object to dictionary for API response"""
    # Get basic model dump
    user_dict = user.model_dump(exclude_none=True)

    # Await OnDemand fields that might be needed
    try:
        user_dict["is_admin"] = await user.awaitable_attrs.is_admin
    except Exception:
        user_dict["is_admin"] = False

    try:
        user_dict["is_gmt"] = await user.awaitable_attrs.is_gmt
    except Exception:
        user_dict["is_gmt"] = False

    try:
        user_dict["is_qat"] = await user.awaitable_attrs.is_qat
    except Exception:
        user_dict["is_qat"] = False

    try:
        user_dict["is_dev"] = await user.awaitable_attrs.is_dev
    except Exception:
        user_dict["is_dev"] = False

    try:
        user_dict["is_restricted"] = await user.is_restricted(session)
    except Exception:
        user_dict["is_restricted"] = False

    # Handle badges - serialize datetime to ISO string
    try:
        # 1. Get badges from JSON field (legacy)
        legacy_badges = []
        json_badges = await user.awaitable_attrs.badges
        if json_badges:
            for badge in json_badges:
                badge_copy = dict(badge)
                if "awarded_at" in badge_copy and isinstance(badge_copy["awarded_at"], datetime):
                    badge_copy["awarded_at"] = badge_copy["awarded_at"].isoformat()
                legacy_badges.append(badge_copy)

        # 2. Get badges from user_badges table (new)
        db_badges = []
        user_badges_list = (
            await session.exec(
                select(UserBadge).where(UserBadge.user_id == user.id).order_by(col(UserBadge.awarded_at).desc())
            )
        ).all()
        if user_badges_list:
            for badge in user_badges_list:
                db_badges.append({
                    "id": badge.id,
                    "description": badge.description,
                    "image_url": badge.image_url,
                    "image@2x_url": badge.image_2x_url,
                    "url": badge.url,
                    "awarded_at": badge.awarded_at.isoformat() if isinstance(badge.awarded_at, datetime) else badge.awarded_at,
                    "user_id": badge.user_id
                })

        # Combine both, preferring DB badges.
        user_dict["badges"] = db_badges + legacy_badges
    except Exception:
        user_dict["badges"] = []

    return user_dict


class SessionsResp(BaseModel):
    total: int
    current: int = 0
    sessions: list[LoginSessionResp]


class AdminStatsResp(BaseModel):
    total_users: int
    online_users: int
    total_pp: float
    total_plays: int
    total_scores: int
    total_beatmaps: int
    blacklisted_beatmaps: int
    performance_server_status: str
    api_server_status: str


class AdminLoginLogItemResp(BaseModel):
    id: int
    user_id: int
    username: str | None = None
    ip_address: str
    user_agent: str | None = None
    login_time: datetime
    login_success: bool
    login_method: str
    client_label: str | None = None
    client_hash: str | None = None
    notes: str | None = None
    country_code: str | None = None
    country_name: str | None = None
    city_name: str | None = None
    organization: str | None = None


class AdminLoginLogListResp(BaseModel):
    total: int
    page: int
    per_page: int
    logs: list[AdminLoginLogItemResp]


class UnknownClientHashResp(BaseModel):
    hash: str
    count: int
    first_seen: str | None = None
    last_seen: str | None = None
    last_user_id: int | None = None
    last_user_agent: str | None = None
    last_detected_os: str | None = None
    last_source: str | None = None


class UnknownClientHashListResp(BaseModel):
    total: int
    page: int
    per_page: int
    hashes: list[UnknownClientHashResp]


class AssignClientHashReq(BaseModel):
    client_hash: str
    client_name: str
    version: str = ""
    os: str = ""
    remove_from_unknown: bool = True


async def _count_online_users(redis) -> int:
    """Count online users with set-first strategy and SCAN fallback."""
    try:
        online_set_key = "metadata:online_users_set"
        if await redis.exists(online_set_key):
            return int(await redis.scard(online_set_key))
    except Exception:
        pass

    try:
        cursor = 0
        online_count = 0
        max_iterations = 500
        iterations = 0
        while True:
            cursor, keys = await redis.scan(cursor, match="metadata:online:*", count=1000)
            online_count += len(keys)
            iterations += 1
            if cursor == 0 or iterations >= max_iterations:
                break
        return online_count
    except Exception:
        return 0


class UserUpdateRequest(BaseModel):
    username: str | None = None
    country_code: str | None = None
    is_qat: bool | None = None
    is_gmt: bool | None = None
    is_admin: bool | None = None
    is_dev: bool | None = None
    # Accept legacy payloads from older frontend builds (dict/str/list)
    badge: dict | str | list[dict] | None = None


class TrustScoreUpdateRequest(BaseModel):
    score: int = Field(ge=0, le=100)


class MarkSuspiciousRequest(BaseModel):
    reasons: list[str] = Field(default_factory=list)
    notes: str | None = None


class AddUserNoteRequest(BaseModel):
    note: str = Field(min_length=1)


class BeatmapBlacklistItem(BaseModel):
    id: int
    beatmapset_id: int
    beatmap_id: int
    beatmapset: dict | None = None


class BadgeCreateRequest(BaseModel):
    description: str
    image_url: str
    image_2x_url: str | None = None
    url: str | None = None
    awarded_at: str | None = None  # ISO format string


class BadgeUpdateRequest(BaseModel):
    description: str | None = None
    image_url: str | None = None
    image_2x_url: str | None = None
    url: str | None = None
    awarded_at: str | None = None  # ISO format string


class GlobalAnnouncementReq(BaseModel):
    title: str = "Server Announcement"
    message: str
    severity: str = "warning"
    also_send_pm: bool = True
    online_only: bool = True
    show_popup: bool = True  # Show as medal popup (for small announcements)
    sender_username: str | None = None
    sender_user_id: int | None = None


class GlobalAnnouncementResp(BaseModel):
    sent_to: int
    severity: str
    title: str
    online_only: bool
    sender_username: str


@router.get(
    "/admin/sessions",
    name="获取当前用户的登录会话列表",
    tags=["用户会话", "g0v0 API", "管理"],
    response_model=SessionsResp,
)
async def get_sessions(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
    geoip: GeoIPService,
):
    current_user, token = user_and_token
    current = 0

    sessions = (
        await session.exec(
            select(
                LoginSession,
            )
            .where(LoginSession.user_id == current_user.id, col(LoginSession.is_verified).is_(True))
            .order_by(col(LoginSession.created_at).desc())
        )
    ).all()
    resp = []
    for s in sessions:
        resp.append(LoginSessionResp.from_db(s, geoip))
        if s.token_id == token.id:
            current = s.id

    return SessionsResp(
        total=len(sessions),
        current=current,
        sessions=resp,
    )


@router.delete(
    "/admin/sessions/{session_id}",
    name="注销指定的登录会话",
    tags=["用户会话", "g0v0 API", "管理"],
    status_code=204,
)
async def delete_session(
    session: Database,
    session_id: int,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    current_user, token = user_and_token
    if session_id == token.id:
        raise HTTPException(status_code=400, detail="Cannot delete the current session")

    db_session = await session.get(LoginSession, session_id)
    if not db_session or db_session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")

    await session.delete(db_session)

    token = await session.get(OAuthToken, db_session.token_id or 0)
    if token:
        await session.delete(token)

    await session.commit()
    return


class TrustedDevicesResp(BaseModel):
    total: int
    current: int = 0
    devices: list[TrustedDeviceResp]


@router.get(
    "/admin/trusted-devices",
    name="获取当前用户的受信任设备列表",
    tags=["用户会话", "g0v0 API", "管理"],
    response_model=TrustedDevicesResp,
)
async def get_trusted_devices(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
    geoip: GeoIPService,
):
    current_user, token = user_and_token
    devices = (
        await session.exec(
            select(TrustedDevice)
            .where(TrustedDevice.user_id == current_user.id)
            .order_by(col(TrustedDevice.last_used_at).desc())
        )
    ).all()

    current_device_id = (
        await session.exec(
            select(TrustedDevice.id)
            .join(LoginSession, col(LoginSession.device_id) == TrustedDevice.id)
            .where(
                LoginSession.token_id == token.id,
                TrustedDevice.user_id == current_user.id,
            )
            .limit(1)
        )
    ).first()

    return TrustedDevicesResp(
        total=len(devices),
        current=current_device_id or 0,
        devices=[TrustedDeviceResp.from_db(device, geoip) for device in devices],
    )


@router.delete(
    "/admin/trusted-devices/{device_id}",
    name="移除受信任设备",
    tags=["用户会话", "g0v0 API", "管理"],
    status_code=204,
)
async def delete_trusted_device(
    session: Database,
    device_id: int,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    current_user, token = user_and_token
    device = await session.get(TrustedDevice, device_id)
    current_device_id = (
        await session.exec(
            select(TrustedDevice.id)
            .join(LoginSession, col(LoginSession.device_id) == TrustedDevice.id)
            .where(
                LoginSession.token_id == token.id,
                TrustedDevice.user_id == current_user.id,
            )
            .limit(1)
        )
    ).first()
    if device_id == current_device_id:
        raise HTTPException(status_code=400, detail="Cannot delete the current trusted device")

    if not device or device.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Trusted device not found")

    await session.delete(device)
    await session.commit()
    return


# ========== Admin Statistics ==========

@router.get(
    "/admin/stats",
    name="获取管理员统计数据",
    tags=["管理", "g0v0 API"],
    response_model=AdminStatsResp,
)
async def get_admin_stats(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Get admin statistics: total users, online users, total pp, total plays, total scores, beatmaps, and blacklisted beatmaps"""
    await require_admin(session, user_and_token)

    # Count total users
    total_users = (await session.exec(select(func.count()).select_from(User))).one()

    # Count online users
    redis = get_redis()
    online_users = await _count_online_users(redis)

    # Sum total PP
    total_pp = (await session.exec(select(func.sum(UserStatistics.pp)))).one() or 0.0

    # Sum total plays
    total_plays = (await session.exec(select(func.sum(UserStatistics.play_count)))).one() or 0

    # Count total scores
    total_scores = (await session.exec(select(func.count()).select_from(Score))).one()

    # Count total beatmaps (non-deleted)
    total_beatmaps = (await session.exec(select(func.count()).select_from(Beatmapset))).one()

    # Count blacklisted beatmaps (unique beatmapsets)
    blacklisted_beatmap_ids = (
        await session.exec(select(BannedBeatmaps.beatmap_id))
    ).all()
    # Get unique beatmapsets from banned beatmaps
    if blacklisted_beatmap_ids:
        unique_beatmapsets = (
            await session.exec(
                select(func.distinct(Beatmap.beatmapset_id))
                .where(col(Beatmap.id).in_(blacklisted_beatmap_ids))
            )
        ).all()
        blacklisted_beatmaps = len(unique_beatmapsets)
    else:
        blacklisted_beatmaps = 0

    # Check server status
    performance_server_status = "offline"
    perf_urls = (
        "http://performance-server:8080/health",
        "http://performance-server:8080/",
        "http://localhost:8080/health",
        "http://localhost:8080/",
    )
    try:
        async with httpx.AsyncClient() as client:
            for url in perf_urls:
                try:
                    resp = await client.get(url, timeout=1.5)
                    if resp.status_code < 500:
                        performance_server_status = "online"
                        break
                except Exception:
                    continue
    except Exception:
        pass

    api_server_status = "online"

    return AdminStatsResp(
        total_users=total_users,
        online_users=online_users,
        total_pp=total_pp,
        total_plays=total_plays,
        total_scores=total_scores,
        total_beatmaps=total_beatmaps,
        blacklisted_beatmaps=blacklisted_beatmaps,
        performance_server_status=performance_server_status,
        api_server_status=api_server_status,
    )


@router.get(
    "/admin/login-logs",
    name="Get login history logs",
    tags=["管理", "g0v0 API"],
    response_model=AdminLoginLogListResp,
)
async def get_admin_login_logs(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    search: str = Query(""),
    user_id: int | None = Query(None, ge=0),
    login_success: bool | None = Query(None),
    login_method: str | None = Query(None),
):
    await require_admin(session, user_and_token)

    conditions = []
    search_value = search.strip()

    if user_id is not None:
        conditions.append(col(UserLoginLog.user_id) == user_id)

    if login_success is not None:
        conditions.append(col(UserLoginLog.login_success) == login_success)

    if login_method:
        conditions.append(col(UserLoginLog.login_method).ilike(f"%{login_method.strip()}%"))

    if search_value:
        username_ids = (
            await session.exec(
                select(User.id).where(col(User.username).ilike(f"%{search_value}%")).limit(500)
            )
        ).all()

        text_condition = sql_or(
            col(UserLoginLog.ip_address).ilike(f"%{search_value}%"),
            col(UserLoginLog.user_agent).ilike(f"%{search_value}%"),
            col(UserLoginLog.client_label).ilike(f"%{search_value}%"),
            col(UserLoginLog.client_hash).ilike(f"%{search_value}%"),
            col(UserLoginLog.notes).ilike(f"%{search_value}%"),
            col(UserLoginLog.country_name).ilike(f"%{search_value}%"),
            col(UserLoginLog.city_name).ilike(f"%{search_value}%"),
            col(UserLoginLog.organization).ilike(f"%{search_value}%"),
            col(UserLoginLog.login_method).ilike(f"%{search_value}%"),
        )

        if search_value.isdigit():
            text_condition = sql_or(text_condition, col(UserLoginLog.user_id) == int(search_value))

        if username_ids:
            text_condition = sql_or(text_condition, col(UserLoginLog.user_id).in_(username_ids))

        conditions.append(text_condition)

    count_stmt = select(func.count()).select_from(UserLoginLog)
    data_stmt = select(UserLoginLog)
    if conditions:
        count_stmt = count_stmt.where(*conditions)
        data_stmt = data_stmt.where(*conditions)

    total = (await session.exec(count_stmt)).one()
    rows = (
        await session.exec(
            data_stmt.order_by(col(UserLoginLog.login_time).desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
        )
    ).all()

    user_ids = sorted({row.user_id for row in rows if row.user_id > 0})
    username_map: dict[int, str] = {}
    if user_ids:
        users = (
            await session.exec(
                select(User.id, User.username).where(col(User.id).in_(user_ids))
            )
        ).all()
        username_map = {uid: uname for uid, uname in users}

    logs = [
        AdminLoginLogItemResp(
            id=row.id or 0,
            user_id=row.user_id,
            username=username_map.get(row.user_id),
            ip_address=row.ip_address,
            user_agent=row.user_agent,
            login_time=row.login_time,
            login_success=row.login_success,
            login_method=row.login_method,
            client_label=row.client_label,
            client_hash=row.client_hash,
            notes=row.notes,
            country_code=row.country_code,
            country_name=row.country_name,
            city_name=row.city_name,
            organization=row.organization,
        )
        for row in rows
    ]

    return AdminLoginLogListResp(total=total, page=page, per_page=per_page, logs=logs)


@router.get(
    "/admin/client-hashes/unknown",
    name="Get unknown client hashes",
    tags=["管理", "g0v0 API"],
    response_model=UnknownClientHashListResp,
)
async def get_unknown_client_hashes(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
    verification_service: ClientVerificationService,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    search: str = Query(""),
):
    await require_admin(session, user_and_token)

    unknown = await verification_service.get_unknown_hashes()
    items: list[UnknownClientHashResp] = []
    search_value = search.strip().lower()

    for hash_value, data in unknown.items():
        entry = UnknownClientHashResp(
            hash=hash_value,
            count=int(data.get("count", 0) or 0),
            first_seen=str(data.get("first_seen")) if data.get("first_seen") else None,
            last_seen=str(data.get("last_seen")) if data.get("last_seen") else None,
            last_user_id=int(data["last_user_id"]) if data.get("last_user_id") is not None else None,
            last_user_agent=str(data.get("last_user_agent")) if data.get("last_user_agent") else None,
            last_detected_os=str(data.get("last_detected_os")) if data.get("last_detected_os") else None,
            last_source=str(data.get("last_source")) if data.get("last_source") else None,
        )
        if search_value:
            search_blob = " ".join(
                [
                    entry.hash,
                    entry.last_user_agent or "",
                    entry.last_source or "",
                    str(entry.last_user_id or ""),
                ]
            ).lower()
            if search_value not in search_blob:
                continue
        items.append(entry)

    items.sort(key=lambda x: (x.last_seen or "", x.count), reverse=True)
    total = len(items)
    start = (page - 1) * per_page
    end = start + per_page
    return UnknownClientHashListResp(total=total, page=page, per_page=per_page, hashes=items[start:end])


@router.post(
    "/admin/client-hashes/assign",
    name="Assign unknown client hash",
    tags=["管理", "g0v0 API"],
)
async def assign_unknown_client_hash(
    req: AssignClientHashReq,
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
    verification_service: ClientVerificationService,
):
    await require_admin(session, user_and_token)
    input_hash = req.client_hash.strip().lower()
    normalized_hash, ambiguous = await verification_service.resolve_hash_input(input_hash)
    if ambiguous:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Hash prefix is ambiguous; provide a longer hash.",
                "input_hash": input_hash,
                "candidates": ambiguous[:10],
            },
        )

    try:
        await verification_service.assign_hash_override(
            normalized_hash,
            client_name=req.client_name,
            version=req.version,
            os_name=req.os,
            remove_from_unknown=req.remove_from_unknown,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    resolved = await verification_service.validate_client_version(normalized_hash)
    resolved_name = (resolved.client_name or "").strip()
    resolved_version = (resolved.version or "").strip()
    resolved_os = (resolved.os or "").strip()
    resolved_label = " ".join(part for part in (resolved_name, resolved_version) if part).strip()
    if resolved_os:
        resolved_label = f"{resolved_label} ({resolved_os})" if resolved_label else resolved_os

    updated_login_logs = 0
    if resolved_label:
        login_rows = (
            await session.exec(
                select(UserLoginLog).where(
                    col(UserLoginLog.client_hash) == normalized_hash,
                )
            )
        ).all()
        for row in login_rows:
            if row.client_label != resolved_label:
                row.client_label = resolved_label
                session.add(row)
                updated_login_logs += 1

    updated_score_tokens = 0
    hash_prefix20 = normalized_hash[:20]
    hash_prefix12 = normalized_hash[:12]
    score_rows = (
        await session.exec(
            select(ScoreToken).where(
                sql_or(
                    col(ScoreToken.client_version).like(f"hash:{hash_prefix20}%"),
                    col(ScoreToken.client_version).like(f"%(hash:{hash_prefix12}%)%"),
                )
            )
        )
    ).all()
    for token_row in score_rows:
        current = (token_row.client_version or "").strip()
        if (
            current.startswith(f"hash:{hash_prefix20}")
            or f"(hash:{hash_prefix12})" in current
        ):
            token_row.client_version = resolved_label or current
            session.add(token_row)
            updated_score_tokens += 1

    if updated_login_logs or updated_score_tokens:
        await session.commit()

    return {
        "ok": True,
        "input_hash": input_hash,
        "hash": normalized_hash,
        "resolved_os": resolved_os or None,
        "updated_login_logs": updated_login_logs,
        "updated_score_tokens": updated_score_tokens,
    }


@router.post(
    "/admin/global-announcement",
    name="å‘é€å…¨æœå…¬å‘Š",
    tags=["ç®¡ç†", "g0v0 API", "é€šçŸ¥"],
    response_model=GlobalAnnouncementResp,
)
async def send_global_announcement(
    session: Database,
    req: GlobalAnnouncementReq,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Send a global in-app announcement, optionally mirrored as PM from a bot/admin account."""
    current_user = await require_admin(session, user_and_token)

    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message cannot be empty")

    severity = req.severity.lower()
    if severity not in {"info", "warning", "error"}:
        raise HTTPException(status_code=422, detail="severity must be one of: info, warning, error")

    sender: User | None = None
    if req.sender_user_id is not None:
        sender = await session.get(User, req.sender_user_id)
    elif req.sender_username:
        sender = (
            await session.exec(
                select(User).where(col(User.username) == req.sender_username.strip()).limit(1)
            )
        ).first()
    if sender is None:
        sender = await session.get(User, BANCHOBOT_ID)
    if sender is None:
        raise HTTPException(status_code=500, detail="Announcement sender user not found")
    sender_username = sender.username

    if req.online_only:
        connected_user_ids = [uid for uid, sockets in server.connect_client.items() if sockets]
        if not connected_user_ids:
            receivers: list[int] = []
        else:
            receivers = (
                await session.exec(
                    select(User.id).where(
                        col(User.id).in_(connected_user_ids),
                        User.id != BANCHOBOT_ID,
                        User.id != sender.id,
                        ~User.is_restricted_query(col(User.id)),
                    )
                )
            ).all()
    else:
        receivers = (
            await session.exec(
                select(User.id).where(
                    User.id != BANCHOBOT_ID,
                    User.id != sender.id,
                    ~User.is_restricted_query(col(User.id)),
                )
            )
        ).all()

    announcement = GlobalAnnouncement.init(
        source_user_id=current_user.id,
        title=req.title.strip() or "Server Announcement",
        message=message,
        severity=severity,  # pyright: ignore[reportArgumentType]
        receiver_ids=receivers,
    )
    await server.new_private_notification(announcement)

    # Also send as fake achievement unlock to trigger MedalOverlay popup
    # This is a hack to show announcements since osu!lazer only shows popups for achievements
    if req.show_popup:
        for user_id in receivers:
            # Use a unique achievement ID for each send (based on timestamp and user_id)
            fake_achievement_id = int(time.time_ns() % 1000000000) + user_id
            fake_achievement = UserAchievementUnlock(
                achievement_id=fake_achievement_id,
                achievement_mode="osu",  # Standard mode
                cover_url="https://a.g0v0.top/-/default-avatar.jpg",  # Use default avatar as cover
                slug="announcement",
                title=req.title.strip() or "Server Announcement",
                description=message[:150] if len(message) <= 150 else message[:147] + "...",  # Truncate for medal display
                user_id=user_id,
            )
            await server.new_private_notification(fake_achievement)

    if req.also_send_pm and receivers:
        targets = (
            await session.exec(
                select(User).where(
                    col(User.id).in_(receivers),
                )
            )
        ).all()

        for target in targets:
            channel = await ChatChannel.get_pm_channel(target.id, sender.id, session)
            if channel is None:
                user_min = min(target.id, sender.id)
                user_max = max(target.id, sender.id)
                channel = ChatChannel(
                    channel_name=f"pm_{user_min}_{user_max}",
                    description="Private message channel",
                    type=ChannelType.PM,
                )
                session.add(channel)
                await session.flush()
                await session.refresh(channel)

            await server.batch_join_channel([target, sender], channel)

            chat_msg = ChatMessage(
                channel_id=channel.channel_id,
                sender_id=sender.id,
                type=MessageType.PLAIN,
                content=f"[{announcement.title}] {message}",
            )
            session.add(chat_msg)
            await session.flush()
            await session.refresh(chat_msg)

            chat_resp = await ChatMessageModel.transform(chat_msg, includes=["sender"])
            await server.send_message_to_channel(chat_resp)
            pm_detail = ChannelMessage.init(
                message=chat_msg,
                user=sender,
                receiver=[target.id],
                channel_type=ChannelType.PM,
            )
            await server.new_private_notification(pm_detail)

        await session.commit()

    return GlobalAnnouncementResp(
        sent_to=len(receivers),
        severity=severity,
        title=announcement.title,
        online_only=req.online_only,
        sender_username=sender_username,
    )


# ========== User Management ==========

class UserListResponse(BaseModel):
    users: list[dict]
    total: int
    page: int
    per_page: int
    total_pages: int


@router.get(
    "/admin/users",
    name="获取所有用户列表",
    tags=["管理", "g0v0 API"],
    response_model=UserListResponse,
)
async def get_all_users(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    search: str = Query(""),
):
    """Get all users with pagination and search (admin only)"""
    await require_admin(session, user_and_token)

    conditions = []
    if search:
        search_term = f"%{search}%"
        # Username search (always works)
        conditions.append(col(User.username).ilike(search_term))

        # Try to parse search as user ID for exact match
        try:
            user_id = int(search)
            conditions.append(col(User.id) == user_id)
        except ValueError:
            pass

    # Build count and data queries
    count_stmt = select(func.count()).select_from(User)
    data_stmt = select(User).order_by(col(User.id))

    if conditions:
        count_stmt = count_stmt.where(*conditions)
        data_stmt = data_stmt.where(*conditions)

    # Get total count
    total = (await session.exec(count_stmt)).one()

    # Get paginated users
    users = (
        await session.exec(
            data_stmt.offset((page - 1) * limit).limit(limit)
        )
    ).all()

    # Convert to dict
    user_list = []
    for user in users:
        user_list.append(await user_to_dict(user, session))

    total_pages = max(1, (total + limit - 1) // limit)

    return UserListResponse(
        users=user_list,
        total=total,
        page=page,
        per_page=limit,
        total_pages=total_pages,
    )


@router.get(
    "/admin/users/{user_id}",
    name="获取指定用户信息",
    tags=["管理", "g0v0 API"],
)
async def get_user(
    session: Database,
    user_id: int,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Get a specific user by ID (admin only)"""
    await require_admin(session, user_and_token)

    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return await user_to_dict(user, session)


@router.patch(
    "/admin/users/{user_id}",
    name="更新用户信息",
    tags=["管理", "g0v0 API"],
)
async def update_user(
    session: Database,
    user_id: int,
    user_data: UserUpdateRequest,
    ip_address: IPAddress,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Update user information (admin only)"""
    current_user = await require_admin(session, user_and_token)

    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user_data.username is not None:
        normalized_username = user_data.username.strip()
        if not normalized_username:
            raise HTTPException(status_code=422, detail="username cannot be empty")

        # Avoid uniqueness crashes and return clean validation error.
        existing_user = (
            await session.exec(
                select(User.id).where(
                    col(User.username) == normalized_username,
                    User.id != user_id,
                ).limit(1)
            )
        ).first()
        if existing_user is not None:
            raise HTTPException(status_code=422, detail="username is already in use")

        if normalized_username != user.username:
            user.username = normalized_username

    if user_data.country_code is not None:
        normalized_country = user_data.country_code.strip().upper()
        user.country_code = normalized_country if normalized_country else None

    if user_data.is_qat is not None:
        user.is_qat = user_data.is_qat

    if user_data.is_gmt is not None:
        user.is_gmt = user_data.is_gmt

    if user_data.is_admin is not None:
        user.is_admin = user_data.is_admin

    if user_data.is_dev is not None:
        user.is_dev = user_data.is_dev

    if user_data.badge is not None:
        import json
        # Note: Badges are stored as JSON, so awarded_at must be an ISO string, not datetime
        # We use plain dicts here instead of Badge TypedDict because JSON storage requires strings
        if isinstance(user_data.badge, str):
            try:
                badge_dict = json.loads(user_data.badge)
                # Ensure awarded_at is an ISO string (not datetime) for JSON storage
                if "awarded_at" in badge_dict:
                    if isinstance(badge_dict["awarded_at"], datetime):
                        badge_dict["awarded_at"] = badge_dict["awarded_at"].isoformat()
                    elif not isinstance(badge_dict["awarded_at"], str):
                        badge_dict["awarded_at"] = datetime.now().isoformat()
                else:
                    badge_dict["awarded_at"] = datetime.now().isoformat()

                # Ensure image@2x_url is present (use image_url as fallback)
                if "image@2x_url" not in badge_dict:
                    badge_dict["image@2x_url"] = badge_dict.get("image_url", "")

                # Store as list of badge dicts (JSON-compatible format)
                # Note: We store as dict with string dates for JSON compatibility, not Badge TypedDict with datetime
                user.badges = cast(Any, [badge_dict])
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                # If parsing fails, create a simple badge structure
                # Note: We store as dict with string dates for JSON compatibility
                user.badges = cast(Any, [{
                    "awarded_at": datetime.now().isoformat(),
                    "description": "",
                    "image_url": user_data.badge if user_data.badge.startswith("http") else "",
                    "image@2x_url": user_data.badge if user_data.badge.startswith("http") else "",
                    "url": "",
                }])
        elif isinstance(user_data.badge, dict):
            # Convert awarded_at to ISO string if it's a datetime
            awarded_at_str = datetime.now().isoformat()
            if "awarded_at" in user_data.badge:
                if isinstance(user_data.badge["awarded_at"], str):
                    awarded_at_str = user_data.badge["awarded_at"]
                elif isinstance(user_data.badge["awarded_at"], datetime):
                    awarded_at_str = user_data.badge["awarded_at"].isoformat()
                else:
                    awarded_at_str = datetime.now().isoformat()

            badge_dict = {
                "awarded_at": awarded_at_str,  # Store as ISO string for JSON
                "description": user_data.badge.get("description", ""),
                "image_url": user_data.badge.get("icon_url") or user_data.badge.get("image_url", ""),
                "image@2x_url": user_data.badge.get("image@2x_url") or user_data.badge.get("icon_url") or user_data.badge.get("image_url", ""),
                "url": user_data.badge.get("url", ""),
            }
            # Note: We store as dict with string dates for JSON compatibility, not Badge TypedDict with datetime
            user.badges = cast(Any, [badge_dict])
        elif isinstance(user_data.badge, list):
            # Legacy frontend may send a list of badge dicts. Keep only JSON-safe dict entries.
            safe_badges: list[dict[str, Any]] = []
            for entry in user_data.badge:
                if not isinstance(entry, dict):
                    continue
                awarded_at = entry.get("awarded_at")
                if isinstance(awarded_at, datetime):
                    awarded_at = awarded_at.isoformat()
                elif not isinstance(awarded_at, str):
                    awarded_at = datetime.now().isoformat()

                safe_badges.append(
                    {
                        "awarded_at": awarded_at,
                        "description": entry.get("description", ""),
                        "image_url": entry.get("icon_url") or entry.get("image_url", ""),
                        "image@2x_url": entry.get("image@2x_url")
                        or entry.get("icon_url")
                        or entry.get("image_url", ""),
                        "url": entry.get("url", ""),
                    }
                )
            user.badges = cast(Any, safe_badges)
        else:
            user.badges = []

    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        error_text = str(exc.orig).lower() if exc.orig else str(exc).lower()
        if "username" in error_text and "duplicate" in error_text:
            raise HTTPException(status_code=422, detail="username is already in use")
        raise HTTPException(status_code=422, detail="invalid user update payload")
    await session.refresh(user)
    return await user_to_dict(user, session)


@router.post(
    "/admin/users/{user_id}/ban",
    name="封禁用户",
    tags=["管理", "g0v0 API"],
    status_code=204,
)
async def ban_user(
    session: Database,
    user_id: int,
    ip_address: IPAddress,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Ban a user (admin only)"""
    current_user = await require_admin(session, user_and_token)

    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot ban yourself")

    # Create restriction history
    restriction = UserAccountHistory(
        id=None,  # Will be auto-generated
        user_id=user_id,
        type=UserAccountHistoryType.RESTRICTION,
        description="Account restricted by admin",
        length=0,
        permanent=True,
    )
    session.add(restriction)

    # Create audit log
    audit_log = AuditLog(
        actor_id=current_user.id,
        actor_username=current_user.username,
        action_type=AuditActionType.USER_BAN,
        target_type=TargetType.USER,
        target_id=user.id,
        target_name=user.username,
        reason="Account restricted by admin",
        ip_address=ip_address,
        metadata={"permanent": True},
    )
    session.add(audit_log)
    await session.commit()


@router.post(
    "/admin/users/{user_id}/unban",
    name="解封用户",
    tags=["管理", "g0v0 API"],
    status_code=204,
)
async def unban_user(
    session: Database,
    user_id: int,
    ip_address: IPAddress,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Unban a user (admin only)"""
    current_user = await require_admin(session, user_and_token)

    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Remove active restrictions
    restrictions = (
        await session.exec(
            select(UserAccountHistory).where(
                UserAccountHistory.user_id == user_id,
                UserAccountHistory.type == UserAccountHistoryType.RESTRICTION,
            )
        )
    ).all()

    restriction_count = len(restrictions)
    for restriction in restrictions:
        await session.delete(restriction)

    # Create audit log
    audit_log = AuditLog(
        actor_id=current_user.id,
        actor_username=current_user.username,
        action_type=AuditActionType.USER_UNBAN,
        target_type=TargetType.USER,
        target_id=user.id,
        target_name=user.username,
        reason="Account unrestricted by admin",
        ip_address=ip_address,
        metadata={"restrictions_removed": restriction_count},
    )
    session.add(audit_log)
    await session.commit()


# ========== User History & Suspicious Activity ==========

@router.get(
    "/admin/users/{user_id}/history",
    name="获取用户历史记录",
    tags=["管理", "g0v0 API"],
)
async def get_user_history(
    session: Database,
    user_id: int,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Get user account history (admin only)"""
    await require_admin(session, user_and_token)

    history = (
        await session.exec(
            select(UserAccountHistory)
            .where(UserAccountHistory.user_id == user_id)
            .order_by(col(UserAccountHistory.timestamp).desc())
        )
    ).all()

    result = []
    for entry in history:
        result.append({
            "id": entry.id,
            "action": entry.type.value,
            "reason": entry.description,
            "created_at": entry.timestamp.isoformat() if entry.timestamp else None,
            "permanent": entry.permanent,
            "length": entry.length,
        })

    return result


@router.get(
    "/admin/users/{user_id}/suspicious",
    name="获取用户可疑活动信息",
    tags=["管理", "g0v0 API"],
)
async def get_user_suspicious_activity(
    session: Database,
    user_id: int,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Get user suspicious activity information (admin only)"""
    await require_admin(session, user_and_token)

    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get trust-related notes
    notes = (
        await session.exec(
            select(UserAccountHistory)
            .where(
                UserAccountHistory.user_id == user_id,
                UserAccountHistory.type == UserAccountHistoryType.NOTE
            )
            .order_by(col(UserAccountHistory.timestamp).desc())
        )
    ).all()

    # Find possible alt accounts by IP or other criteria
    # This is simplified - in production you'd check IPs, device fingerprints, etc.
    alt_accounts = []

    # Get the user's login logs
    login_logs = (
        await session.exec(
            select(UserLoginLog)
            .where(UserLoginLog.user_id == user_id)
            .order_by(col(UserLoginLog.login_time).desc())
            .limit(10)
        )
    ).all()

    unique_ips = {log.ip_address for log in login_logs if log.ip_address}

    # Find other users with same IP
    for ip in unique_ips:
        other_users = (
            await session.exec(
                select(UserLoginLog, User.username)
                .join(User, col(UserLoginLog.user_id) == User.id)
                .where(
                    UserLoginLog.ip_address == ip,
                    UserLoginLog.user_id != user_id
                )
                .distinct()
            )
        ).all()

        for log_entry, username in other_users:
            if not any(a["id"] == log_entry.user_id for a in alt_accounts):
                alt_accounts.append({
                    "id": log_entry.user_id,
                    "username": username or f"User {log_entry.user_id}",
                    "reason": f"Same IP: {ip}",
                })

    # Combine all notes into a single string
    all_notes = "\n".join([n.description or "" for n in notes])

    return {
        "trust_score": 100 - len([n for n in notes if "suspicious" in (n.description or "").lower()]) * 10,
        "is_suspicious": any("suspicious" in (n.description or "").lower() for n in notes),
        "suspicious_reasons": [n.description for n in notes if "suspicious" in (n.description or "").lower()],
        "alt_accounts": alt_accounts[:5],  # Limit to 5
        "notes": all_notes,
        "recent_ips": list(unique_ips)[:5],
    }


@router.patch(
    "/admin/users/{user_id}/trust-score",
    name="更新用户信任分数",
    tags=["管理", "g0v0 API"],
)
async def update_user_trust_score(
    session: Database,
    user_id: int,
    req: TrustScoreUpdateRequest,
    ip_address: IPAddress,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Update user trust score (admin only)"""
    current_user = await require_admin(session, user_and_token)

    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Create a note for the trust score change
    note = UserAccountHistory(
        user_id=user_id,
        type=UserAccountHistoryType.NOTE,
        description=f"Trust score updated to {req.score} by admin",
        length=0,
        permanent=False,
    )
    session.add(note)

    # Create audit log
    audit_log = AuditLog(
        actor_id=current_user.id,
        actor_username=current_user.username,
        action_type=AuditActionType.USER_UPDATE,
        target_type=TargetType.USER,
        target_id=user.id,
        target_name=user.username,
        reason=f"Trust score updated to {req.score}",
        ip_address=ip_address,
        metadata={"trust_score": req.score},
    )
    session.add(audit_log)
    await session.commit()

    return {"trust_score": req.score}


@router.post(
    "/admin/users/{user_id}/suspicious",
    name="标记用户为可疑",
    tags=["管理", "g0v0 API"],
)
async def mark_user_suspicious(
    session: Database,
    user_id: int,
    req: MarkSuspiciousRequest,
    ip_address: IPAddress,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Mark user as suspicious (admin only)"""
    current_user = await require_admin(session, user_and_token)

    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Add each reason as a note
    for reason in req.reasons:
        note = UserAccountHistory(
            user_id=user_id,
            type=UserAccountHistoryType.NOTE,
            description=f"[SUSPICIOUS] {reason}",
            length=0,
            permanent=False,
        )
        session.add(note)

    if req.notes:
        note = UserAccountHistory(
            user_id=user_id,
            type=UserAccountHistoryType.NOTE,
            description=f"[ADMIN NOTES] {req.notes}",
            length=0,
            permanent=False,
        )
        session.add(note)

    # Create audit log
    audit_log = AuditLog(
        actor_id=current_user.id,
        actor_username=current_user.username,
        action_type=AuditActionType.USER_BAN,  # Using BAN type for suspicious marking
        target_type=TargetType.USER,
        target_id=user.id,
        target_name=user.username,
        reason=f"User marked as suspicious: {', '.join(req.reasons)}",
        ip_address=ip_address,
        metadata={"suspicious_reasons": req.reasons, "notes": req.notes},
    )
    session.add(audit_log)
    await session.commit()

    return {"marked_as_suspicious": True, "reasons": req.reasons}


@router.delete(
    "/admin/users/{user_id}/suspicious",
    name="取消用户可疑标记",
    tags=["管理", "g0v0 API"],
    status_code=204,
)
async def unmark_user_suspicious(
    session: Database,
    user_id: int,
    ip_address: IPAddress,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Remove suspicious mark from user (admin only)"""
    current_user = await require_admin(session, user_and_token)

    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Add removal note
    note = UserAccountHistory(
        user_id=user_id,
        type=UserAccountHistoryType.NOTE,
        description="[SUSPICIOUS] Status cleared by admin",
        length=0,
        permanent=False,
    )
    session.add(note)

    # Create audit log
    audit_log = AuditLog(
        actor_id=current_user.id,
        actor_username=current_user.username,
        action_type=AuditActionType.USER_UNBAN,
        target_type=TargetType.USER,
        target_id=user.id,
        target_name=user.username,
        reason="Suspicious status cleared",
        ip_address=ip_address,
        metadata={"cleared_suspicious": True},
    )
    session.add(audit_log)
    await session.commit()


@router.post(
    "/admin/users/{user_id}/reset-password",
    name="重置用户密码",
    tags=["管理", "g0v0 API"],
)
async def reset_user_password(
    session: Database,
    user_id: int,
    ip_address: IPAddress,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Reset user password and send email (admin only)"""
    current_user = await require_admin(session, user_and_token)

    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Generate a password reset token
    import secrets
    token = secrets.token_urlsafe(32)

    # Store token in Redis with expiration
    redis = get_redis()
    await redis.setex(f"password_reset:{user.id}", 3600, token)  # 1 hour

    # TODO: Send email with reset link
    # For now, just return the token
    logger.info(f"Password reset requested for user {user.username} by admin {current_user.username}")

    # Create audit log
    audit_log = AuditLog(
        actor_id=current_user.id,
        actor_username=current_user.username,
        action_type=AuditActionType.USER_UPDATE,
        target_type=TargetType.USER,
        target_id=user.id,
        target_name=user.username,
        reason="Password reset requested by admin",
        ip_address=ip_address,
        metadata={"token_generated": True},
    )
    session.add(audit_log)
    await session.commit()

    return {"message": "Password reset email queued", "token": token}


@router.post(
    "/admin/users/{user_id}/resend-verification",
    name="重新发送验证邮件",
    tags=["管理", "g0v0 API"],
)
async def resend_verification_email(
    session: Database,
    user_id: int,
    ip_address: IPAddress,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Resend email verification (admin only)"""
    current_user = await require_admin(session, user_and_token)

    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # TODO: Trigger email verification flow
    logger.info(f"Verification email resent for user {user.username} by admin {current_user.username}")

    # Create audit log
    audit_log = AuditLog(
        actor_id=current_user.id,
        actor_username=current_user.username,
        action_type=AuditActionType.USER_UPDATE,
        target_type=TargetType.USER,
        target_id=user.id,
        target_name=user.username,
        reason="Verification email resent by admin",
        ip_address=ip_address,
        metadata={"action": "resend_verification"},
    )
    session.add(audit_log)
    await session.commit()

    return {"message": "Verification email queued"}


@router.delete(
    "/admin/users/{user_id}",
    name="删除用户账户",
    tags=["管理", "g0v0 API"],
    status_code=204,
)
async def delete_user_account(
    session: Database,
    user_id: int,
    ip_address: IPAddress,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Delete user account permanently (admin only)"""
    current_user = await require_admin(session, user_and_token)

    # Prevent self-deletion
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Create audit log before deletion
    audit_log = AuditLog(
        actor_id=current_user.id,
        actor_username=current_user.username,
        action_type=AuditActionType.USER_DELETE,
        target_type=TargetType.USER,
        target_id=user.id,
        target_name=user.username,
        reason="User account deleted by admin",
        ip_address=ip_address,
        metadata={"username": user.username, "email": user.email},
    )
    session.add(audit_log)
    await session.commit()

    # Delete user
    await session.delete(user)
    await session.commit()


@router.post(
    "/admin/users/{user_id}/notes",
    name="添加用户备注",
    tags=["管理", "g0v0 API"],
)
async def add_user_note(
    session: Database,
    user_id: int,
    req: AddUserNoteRequest,
    ip_address: IPAddress,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Add user note (admin only)"""
    current_user = await require_admin(session, user_and_token)

    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    note = UserAccountHistory(
        user_id=user_id,
        type=UserAccountHistoryType.NOTE,
        description=f"[NOTE] {req.note}",
        length=0,
        permanent=False,
    )
    session.add(note)

    # Create audit log
    audit_log = AuditLog(
        actor_id=current_user.id,
        actor_username=current_user.username,
        action_type=AuditActionType.USER_UPDATE,
        target_type=TargetType.USER,
        target_id=user.id,
        target_name=user.username,
        reason=f"Admin note added: {req.note[:50]}...",
        ip_address=ip_address,
        metadata={"note": req.note},
    )
    session.add(audit_log)
    await session.commit()

    return {"note_added": True}


# ========== Team Management for Users ==========

@router.post(
    "/admin/users/{user_id}/team",
    name="将用户添加到团队",
    tags=["管理", "g0v0 API"],
)
async def add_user_to_team(
    session: Database,
    user_id: int,
    team_id: int,
    ip_address: IPAddress,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Add user to a team (admin only)"""
    current_user = await require_admin(session, user_and_token)

    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    team = await session.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    # Check if user is already in a team
    existing_membership = (
        await session.exec(
            select(TeamMember).where(TeamMember.user_id == user_id)
        )
    ).first()

    if existing_membership:
        # Remove from old team
        await session.delete(existing_membership)

    # Add to new team
    membership = TeamMember(team_id=team_id, user_id=user_id)
    session.add(membership)

    # Create audit log
    audit_log = AuditLog(
        actor_id=current_user.id,
        actor_username=current_user.username,
        action_type=AuditActionType.USER_UPDATE,
        target_type=TargetType.USER,
        target_id=user.id,
        target_name=user.username,
        reason=f"Added to team: {team.name}",
        ip_address=ip_address,
        metadata={"team_id": team_id, "team_name": team.name},
    )
    session.add(audit_log)
    await session.commit()

    return {"added_to_team": team_id, "team_name": team.name}


@router.delete(
    "/admin/users/{user_id}/team",
    name="将用户从团队移除",
    tags=["管理", "g0v0 API"],
    status_code=204,
)
async def remove_user_from_team(
    session: Database,
    user_id: int,
    ip_address: IPAddress,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Remove user from team (admin only)"""
    current_user = await require_admin(session, user_and_token)

    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    existing_membership = (
        await session.exec(
            select(TeamMember).where(TeamMember.user_id == user_id)
        )
    ).first()

    if existing_membership:
        await session.delete(existing_membership)

        # Create audit log
        audit_log = AuditLog(
            actor_id=current_user.id,
            actor_username=current_user.username,
            action_type=AuditActionType.USER_UPDATE,
            target_type=TargetType.USER,
            target_id=user.id,
            target_name=user.username,
            reason="Removed from team",
            ip_address=ip_address,
            metadata={"action": "remove_from_team"},
        )
        session.add(audit_log)
        await session.commit()


# ========== Dev Access Check Endpoint ==========

@router.get(
    "/admin/check-dev",
    name="检查开发权限",
    tags=["管理", "g0v0 API"],
)
async def check_dev_access(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Check if current user has dev access"""
    current_user, _ = user_and_token

    is_dev = False
    try:
        is_dev = await current_user.awaitable_attrs.is_dev
    except Exception:
        pass

    return {"has_dev_access": current_user.is_admin or is_dev}


# ========== Beatmap Blacklist ==========

@router.get(
    "/admin/beatmaps/blacklist",
    name="获取黑名单谱面列表",
    tags=["管理", "g0v0 API"],
)
async def get_blacklisted_beatmaps(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Get all blacklisted beatmaps (admin only)"""
    await require_admin(session, user_and_token)

    # Get all banned beatmaps
    banned_beatmaps = (
        await session.exec(select(BannedBeatmaps))
    ).all()

    result = []
    seen_beatmapsets = set()

    for banned_item in banned_beatmaps:
        # Get the beatmap to find its beatmapset
        beatmap = await session.get(Beatmap, banned_item.beatmap_id)
        if not beatmap:
            continue

        beatmapset_id = beatmap.beatmapset_id

        # Only add each beatmapset once
        if beatmapset_id in seen_beatmapsets:
            continue
        seen_beatmapsets.add(beatmapset_id)

        beatmapset = await session.get(Beatmapset, beatmapset_id)
        beatmapset_dict = None
        if beatmapset:
            beatmapset_dict = {
                "id": beatmapset.id,
                "title": beatmapset.title,
                "artist": beatmapset.artist,
            }
        result.append(
            BeatmapBlacklistItem(
                id=banned_item.id or 0,
                beatmapset_id=beatmapset_id,
                beatmap_id=banned_item.beatmap_id,
                beatmapset=beatmapset_dict,
            )
        )

    return result


class BeatmapBlacklistRequest(BaseModel):
    beatmapset_id: int | None = None
    beatmap_id: int | None = None

    @model_validator(mode="after")
    def validate_target(self):
        if self.beatmapset_id is None and self.beatmap_id is None:
            raise ValueError("Either beatmapset_id or beatmap_id is required")
        if self.beatmapset_id is not None and self.beatmap_id is not None:
            raise ValueError("Provide only one of beatmapset_id or beatmap_id")
        return self


@router.post(
    "/admin/beatmaps/blacklist",
    name="添加谱面到黑名单",
    tags=["管理", "g0v0 API"],
    status_code=201,
)
async def add_blacklisted_beatmap(
    session: Database,
    request: BeatmapBlacklistRequest,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Add a beatmap or beatmapset to blacklist (admin only)"""
    await require_admin(session, user_and_token)

    if request.beatmap_id is not None:
        beatmap_id = request.beatmap_id
        beatmap = await session.get(Beatmap, beatmap_id)
        if not beatmap:
            raise HTTPException(status_code=404, detail="Beatmap not found")

        existing_banned = (
            await session.exec(
                select(BannedBeatmaps).where(BannedBeatmaps.beatmap_id == beatmap_id)
            )
        ).first()
        if existing_banned:
            raise HTTPException(status_code=400, detail="Beatmap is already blacklisted")

        session.add(BannedBeatmaps(beatmap_id=beatmap_id))
        await session.commit()
        return {
            "beatmap_id": beatmap_id,
            "beatmapset_id": beatmap.beatmapset_id,
            "message": "Beatmap added to blacklist",
        }

    beatmapset_id = request.beatmapset_id
    if beatmapset_id is None:
        raise HTTPException(status_code=422, detail="beatmapset_id is required")

    # Verify beatmapset exists
    beatmapset = await session.get(Beatmapset, beatmapset_id)
    if not beatmapset:
        raise HTTPException(status_code=404, detail="Beatmapset not found")

    # Get all beatmaps in this beatmapset
    beatmaps = (
        await session.exec(
            select(Beatmap).where(Beatmap.beatmapset_id == beatmapset_id)
        )
    ).all()

    if not beatmaps:
        raise HTTPException(status_code=404, detail="No beatmaps found in this beatmapset")

    # Check if any beatmap in this set is already banned
    beatmap_ids = [b.id for b in beatmaps]
    existing_banned = (
        await session.exec(
            select(BannedBeatmaps).where(col(BannedBeatmaps.beatmap_id).in_(beatmap_ids))
        )
    ).all()

    if existing_banned:
        raise HTTPException(status_code=400, detail="Some beatmaps in this beatmapset are already blacklisted")

    # Ban all beatmaps in the set
    for beatmap in beatmaps:
        banned_item = BannedBeatmaps(beatmap_id=beatmap.id)
        session.add(banned_item)

    await session.commit()

    return {"beatmapset_id": beatmapset_id, "message": "Beatmapset added to blacklist"}


@router.delete(
    "/admin/beatmaps/blacklist/beatmap/{beatmap_id}",
    name="ä»Žé»‘åå•ç§»é™¤å•ä¸ªè°±é¢",
    tags=["ç®¡ç†", "g0v0 API"],
    status_code=204,
)
async def remove_blacklisted_single_beatmap(
    session: Database,
    beatmap_id: int,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Remove a single beatmap from blacklist (admin only)"""
    await require_admin(session, user_and_token)

    banned_item = (
        await session.exec(
            select(BannedBeatmaps).where(BannedBeatmaps.beatmap_id == beatmap_id)
        )
    ).first()
    if not banned_item:
        raise HTTPException(status_code=404, detail="Beatmap not in blacklist")

    await session.delete(banned_item)
    await session.commit()


@router.delete(
    "/admin/beatmaps/blacklist/{beatmapset_id}",
    name="从黑名单移除谱面",
    tags=["管理", "g0v0 API"],
    status_code=204,
)
async def remove_blacklisted_beatmap(
    session: Database,
    beatmapset_id: int,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Remove a beatmapset from blacklist (admin only)"""
    await require_admin(session, user_and_token)

    # Get all beatmaps in this beatmapset
    beatmaps = (
        await session.exec(
            select(Beatmap).where(Beatmap.beatmapset_id == beatmapset_id)
        )
    ).all()

    if not beatmaps:
        raise HTTPException(status_code=404, detail="Beatmapset not found")

    # Get all banned beatmaps for this beatmapset
    beatmap_ids = [b.id for b in beatmaps]
    banned_items = (
        await session.exec(
            select(BannedBeatmaps).where(col(BannedBeatmaps.beatmap_id).in_(beatmap_ids))
        )
    ).all()

    if not banned_items:
        raise HTTPException(status_code=404, detail="Beatmapset not in blacklist")

    # Remove all banned entries for this beatmapset
    for banned_item in banned_items:
        await session.delete(banned_item)

    await session.commit()


# ========== Beatmap Management ==========

@router.get(
    "/admin/beatmaps",
    name="获取所有谱面",
    tags=["管理", "g0v0 API"],
)
async def get_beatmaps(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
    page: int = Query(1, ge=1),
    limit: int = Query(25, ge=1, le=100),
    search: str = Query("", description="Search by artist, title, or ID"),
):
    """Get all beatmaps with pagination (admin only)"""
    await require_admin(session, user_and_token)

    offset = (page - 1) * limit

    # Build query with optional search
    query = select(Beatmapset)

    if search:
        search_term = f"%{search}%"
        # Try to parse as ID first
        try:
            search_id = int(search)
            query = query.where(
                sql_or(
                    col(Beatmapset.id) == search_id,
                    col(Beatmapset.title).like(search_term),
                    col(Beatmapset.artist).like(search_term)
                )
            )
        except ValueError:
            # Not a number, search in title and artist
            query = query.where(
                sql_or(
                    col(Beatmapset.title).like(search_term),
                    col(Beatmapset.artist).like(search_term)
                )
            )

    # Get total count
    if search:
        total_count = (await session.exec(select(func.count()).select_from(query.subquery()))).one()
    else:
        total_count = (await session.exec(select(func.count()).select_from(Beatmapset))).one()

    # Get beatmapsets with pagination
    beatmapsets = (
        await session.exec(
            query
            .order_by(col(Beatmapset.id).desc())
            .offset(offset)
            .limit(limit)
        )
    ).all()

    result = []
    for beatmapset in beatmapsets:
        # Get beatmaps for this set
        beatmaps = (
            await session.exec(
                select(Beatmap).where(Beatmap.beatmapset_id == beatmapset.id)
            )
        ).all()

        # Get cover URL
        cover_url = None
        if beatmapset.covers:
            cover_url = beatmapset.covers.get("cover") or beatmapset.covers.get("card")

        beatmapset_dict = {
            "id": beatmapset.id,
            "title": beatmapset.title,
            "artist": beatmapset.artist,
            "creator": beatmapset.creator,
            "rank_status": beatmapset.beatmap_status.name.lower() if beatmapset.beatmap_status else None,
            "covers": beatmapset.covers,
            "cover_url": cover_url,
            "beatmaps": [
                {
                    "id": b.id,
                    "version": b.version,
                    "difficulty_rating": b.difficulty_rating,
                    "mode": b.mode.value if b.mode else None,
                }
                for b in beatmaps
            ],
        }
        result.append(beatmapset_dict)

    return {
        "total": total_count,
        "page": page,
        "limit": limit,
        "total_pages": (total_count + limit - 1) // limit,
        "beatmapsets": result,
    }


@router.get(
    "/admin/beatmaps/{beatmap_id}",
    name="获取谱面详情",
    tags=["管理", "g0v0 API"],
)
async def get_beatmap_details(
    session: Database,
    beatmap_id: int,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Get beatmap details (admin only)"""
    await require_admin(session, user_and_token)

    beatmapset = await session.get(Beatmapset, beatmap_id)
    if not beatmapset:
        raise HTTPException(status_code=404, detail="Beatmapset not found")

    # Get all beatmaps in this set
    beatmaps = (
        await session.exec(
            select(Beatmap).where(Beatmap.beatmapset_id == beatmapset.id)
        )
    ).all()

    # Get cover URL
    cover_url = None
    if beatmapset.covers:
        cover_url = beatmapset.covers.get("cover") or beatmapset.covers.get("card")

    return {
        "id": beatmapset.id,
        "title": beatmapset.title,
        "artist": beatmapset.artist,
        "creator": beatmapset.creator,
        "rank_status": beatmapset.beatmap_status.name.lower() if beatmapset.beatmap_status else None,
        "covers": beatmapset.covers,
        "cover_url": cover_url,
        "beatmaps": [
            {
                "id": b.id,
                "version": b.version,
                "difficulty_rating": b.difficulty_rating,
                "mode": b.mode.value if b.mode else None,
            }
            for b in beatmaps
        ],
    }


class RankStatusUpdate(BaseModel):
    status: str


@router.post(
    "/admin/beatmaps/{beatmapset_id}/rank",
    name="更新谱面状态",
    tags=["管理", "g0v0 API"],
)
async def update_beatmap_rank_status(
    session: Database,
    beatmapset_id: int,
    request: RankStatusUpdate,
    ip_address: IPAddress,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Update beatmapset rank status (admin only)"""
    current_user = await require_admin(session, user_and_token)

    from app.models.beatmap import BeatmapRankStatus

    try:
        new_status = BeatmapRankStatus(request.status)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid rank status: {request.status}")

    beatmapset = await session.get(Beatmapset, beatmapset_id)
    if not beatmapset:
        raise HTTPException(status_code=404, detail="Beatmapset not found")

    old_status = beatmapset.beatmap_status
    beatmapset.beatmap_status = new_status

    # Determine audit action type
    if new_status == BeatmapRankStatus.RANKED:
        action_type = AuditActionType.BEATMAP_RANK
    elif new_status == BeatmapRankStatus.LOVED:
        action_type = AuditActionType.BEATMAP_LOVE
    elif new_status in [BeatmapRankStatus.PENDING, BeatmapRankStatus.WIP, BeatmapRankStatus.GRAVEYARD]:
        action_type = AuditActionType.BEATMAP_UNRANK
    else:
        action_type = AuditActionType.BEATMAP_UNRANK

    # Create audit log
    audit_log = AuditLog(
        actor_id=current_user.id,
        actor_username=current_user.username,
        action_type=action_type,
        target_type=TargetType.BEATMAPSET,
        target_id=beatmapset.id,
        target_name=beatmapset.title,
        reason=f"Changed rank status from {old_status.value if old_status else 'None'} to {new_status.value}",
        ip_address=ip_address,
        metadata={
            "old_status": old_status.value if old_status else None,
            "new_status": new_status.value,
            "beatmapset_id": beatmapset.id,
        },
    )
    session.add(audit_log)
    await session.commit()
    await session.refresh(beatmapset)

    return {"id": beatmapset.id, "rank_status": beatmapset.beatmap_status.value}


@router.post(
    "/admin/beatmaps/{beatmapset_id}/ban",
    name="封禁谱面",
    tags=["管理", "g0v0 API"],
)
async def ban_beatmapset(
    session: Database,
    beatmapset_id: int,
    ip_address: IPAddress,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Ban a beatmapset and remove all scores (admin only)"""
    current_user = await require_admin(session, user_and_token)

    from app.database.score import Score

    # Get all beatmaps in this set
    beatmaps = (
        await session.exec(
            select(Beatmap).where(Beatmap.beatmapset_id == beatmapset_id)
        )
    ).all()

    if not beatmaps:
        raise HTTPException(status_code=404, detail="Beatmapset not found")

    beatmapset = await session.get(Beatmapset, beatmapset_id)
    if not beatmapset:
        raise HTTPException(status_code=404, detail="Beatmapset not found")

    # Delete all scores for these beatmaps
    scores_deleted = 0
    for beatmap in beatmaps:
        scores = (
            await session.exec(
                select(Score).where(Score.beatmap_id == beatmap.id)
            )
        ).all()
        for score in scores:
            await session.delete(score)
            scores_deleted += 1

    # Add all beatmaps to blacklist
    beatmaps_banned = 0
    for beatmap in beatmaps:
        # Check if already blacklisted
        existing = (
            await session.exec(
                select(BannedBeatmaps).where(BannedBeatmaps.beatmap_id == beatmap.id)
            )
        ).first()
        if not existing:
            banned_beatmap = BannedBeatmaps(beatmap_id=beatmap.id)
            session.add(banned_beatmap)
            beatmaps_banned += 1

    # Create audit log
    audit_log = AuditLog(
        actor_id=current_user.id,
        actor_username=current_user.username,
        action_type=AuditActionType.BEATMAP_DELETE,
        target_type=TargetType.BEATMAPSET,
        target_id=beatmapset.id,
        target_name=beatmapset.title,
        reason="Beatmapset banned by admin",
        ip_address=ip_address,
        metadata={
            "beatmaps_banned": beatmaps_banned,
            "scores_deleted": scores_deleted,
            "beatmapset_id": beatmapset.id,
        },
    )
    session.add(audit_log)
    await session.commit()

    return {
        "beatmapset_id": beatmapset_id,
        "beatmaps_banned": beatmaps_banned,
        "scores_deleted": scores_deleted,
        "message": f"Beatmapset banned and {scores_deleted} scores removed",
    }


# ========== User Wipe ==========

class WipeRequest(BaseModel):
    mode: str  # e.g., "osu", "taiko", "fruits", "mania"


@router.post(
    "/admin/users/{user_id}/wipe",
    name="清除用户数据",
    tags=["管理", "g0v0 API"],
)
async def wipe_user_stats(
    session: Database,
    user_id: int,
    request: WipeRequest,
    ip_address: IPAddress,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Wipe user statistics and scores for a specific mode (admin only)"""
    current_user = await require_admin(session, user_and_token)

    from app.database.score import Score
    from app.models.score import GameMode

    try:
        mode = GameMode(request.mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid game mode: {request.mode}")

    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Delete scores for this user and mode
    scores = (
        await session.exec(
            select(Score).where(
                Score.user_id == user_id,
                Score.mode == mode
            )
        )
    ).all()

    scores_deleted = 0
    for score in scores:
        await session.delete(score)
        scores_deleted += 1

    # Create audit log
    audit_log = AuditLog(
        actor_id=current_user.id,
        actor_username=current_user.username,
        action_type=AuditActionType.USER_WIPE,
        target_type=TargetType.USER,
        target_id=user.id,
        target_name=user.username,
        reason=f"User statistics wiped for mode {mode.value}",
        ip_address=ip_address,
        metadata={"mode": mode.value, "scores_deleted": scores_deleted},
    )
    session.add(audit_log)
    await session.commit()

    return {
        "user_id": user_id,
        "mode": mode.value,
        "scores_deleted": scores_deleted,
        "message": f"Wiped {scores_deleted} scores for {mode.value} mode",
    }


# ========== Badge Management ==========
# Now using user_badges table instead of JSON in User.badges field

@router.get(
    "/admin/user-badges",
    name="获取所有徽章",
    tags=["管理", "g0v0 API"],
)
async def get_user_badges(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Get all badges from user_badges table (admin only)"""
    await require_admin(session, user_and_token)

    try:
        # Join with User to get username
        statement = (
            select(UserBadge, User.username)
            .outerjoin(User, col(UserBadge.user_id) == User.id)
            .order_by(col(UserBadge.id).desc())
        )
        results = (await session.exec(statement)).all()
    except Exception:
        # Keep admin page usable even if table/schema is missing in a partially migrated environment.
        return []

    badges = []
    for badge, username in results:
        badge_dict = badge.model_dump()
        badge_dict["username"] = username
        # Convert datetime to ISO string
        if badge_dict.get("awarded_at") and isinstance(badge_dict["awarded_at"], datetime):
            badge_dict["awarded_at"] = badge_dict["awarded_at"].isoformat()
        badges.append(badge_dict)

    return badges


@router.post(
    "/admin/user-badges",
    name="创建徽章",
    tags=["管理", "g0v0 API"],
    status_code=201,
)
async def create_user_badge(
    session: Database,
    badge_data: UserBadgeCreate,
    ip_address: IPAddress,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Create a badge in user_badges table (admin only)"""
    current_user = await require_admin(session, user_and_token)

    # Set default awarded_at if not provided
    awarded_at = badge_data.awarded_at or datetime.now()

    # Create new badge
    new_badge = UserBadge(
        description=badge_data.description,
        image_url=badge_data.image_url,
        image_2x_url=badge_data.image_2x_url or badge_data.image_url,
        url=badge_data.url or "",
        awarded_at=awarded_at,
        user_id=badge_data.user_id,
    )

    session.add(new_badge)

    # Create audit log
    audit_log = AuditLog(
        actor_id=current_user.id,
        actor_username=current_user.username,
        action_type=AuditActionType.BADGE_CREATE,
        target_type=TargetType.BADGE,
        target_id=badge_data.user_id,
        target_name=f"Badge for user {badge_data.user_id}",
        reason=f"Created badge: {badge_data.description[:100]}...",
        ip_address=ip_address,
        metadata={
            "badge_description": badge_data.description,
            "user_id": badge_data.user_id,
        },
    )
    session.add(audit_log)
    await session.commit()
    await session.refresh(new_badge)

    return new_badge


@router.patch(
    "/admin/user-badges/{badge_id}",
    name="更新徽章",
    tags=["管理", "g0v0 API"],
)
async def update_user_badge(
    session: Database,
    badge_id: int,
    badge_data: UserBadgeUpdate,
    ip_address: IPAddress,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Update a badge in user_badges table (admin only)"""
    current_user = await require_admin(session, user_and_token)

    # Get the badge
    badge = await session.get(UserBadge, badge_id)
    if not badge:
        raise HTTPException(status_code=404, detail="Badge not found")

    # Track changes
    changes = {}
    if badge_data.description is not None and badge.description != badge_data.description:
        changes["description"] = {"old": badge.description, "new": badge_data.description}
        badge.description = badge_data.description
    if badge_data.image_url is not None and badge.image_url != badge_data.image_url:
        changes["image_url"] = {"old": badge.image_url, "new": badge_data.image_url}
        badge.image_url = badge_data.image_url
    if badge_data.image_2x_url is not None and badge.image_2x_url != badge_data.image_2x_url:
        changes["image_2x_url"] = {"old": badge.image_2x_url, "new": badge_data.image_2x_url}
        badge.image_2x_url = badge_data.image_2x_url
    if badge_data.url is not None and badge.url != badge_data.url:
        changes["url"] = {"old": badge.url, "new": badge_data.url}
        badge.url = badge_data.url
    if badge_data.awarded_at is not None and badge.awarded_at != badge_data.awarded_at:
        changes["awarded_at"] = {"old": str(badge.awarded_at), "new": str(badge_data.awarded_at)}
        badge.awarded_at = badge_data.awarded_at
    if badge_data.user_id is not None and badge.user_id != badge_data.user_id:
        changes["user_id"] = {"old": badge.user_id, "new": badge_data.user_id}
        badge.user_id = badge_data.user_id

    if changes:
        # Create audit log
        audit_log = AuditLog(
            actor_id=current_user.id,
            actor_username=current_user.username,
            action_type=AuditActionType.BADGE_UPDATE,
            target_type=TargetType.BADGE,
            target_id=badge.id,
            target_name=f"Badge {badge.id} for user {badge.user_id}",
            reason=f"Updated badge fields: {', '.join(changes.keys())}",
            ip_address=ip_address,
            metadata={
                "badge_id": badge.id,
                "user_id": badge.user_id,
                "changes": changes,
            },
        )
        session.add(audit_log)

    await session.commit()
    await session.refresh(badge)

    return badge


@router.delete(
    "/admin/user-badges/{badge_id}",
    name="删除徽章",
    tags=["管理", "g0v0 API"],
    status_code=204,
)
async def delete_user_badge(
    session: Database,
    badge_id: int,
    ip_address: IPAddress,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Delete a badge from user_badges table (admin only)"""
    current_user = await require_admin(session, user_and_token)

    badge = await session.get(UserBadge, badge_id)
    if not badge:
        raise HTTPException(status_code=404, detail="Badge not found")

    # Store badge info before deletion for audit log
    badge_info = {
        "id": badge.id,
        "user_id": badge.user_id,
        "description": badge.description,
        "awarded_at": str(badge.awarded_at) if badge.awarded_at else None,
    }

    # Create audit log before deleting
    audit_log = AuditLog(
        actor_id=current_user.id,
        actor_username=current_user.username,
        action_type=AuditActionType.BADGE_DELETE,
        target_type=TargetType.BADGE,
        target_id=badge.id,
        target_name=f"Badge {badge.id} for user {badge.user_id}",
        reason="Badge deleted by admin",
        ip_address=ip_address,
        metadata=badge_info,
    )
    session.add(audit_log)

    # Delete the badge
    await session.delete(badge)
    await session.commit()


# ========== Team Management ==========

@router.get(
    "/admin/teams",
    name="获取所有战队",
    tags=["管理", "g0v0 API"],
)
async def get_all_teams(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Get all teams (admin only)"""
    await require_admin(session, user_and_token)

    teams = (await session.exec(select(Team).order_by(col(Team.created_at).desc()))).all()
    return teams


@router.patch(
    "/admin/teams/{team_id}",
    name="更新战队",
    tags=["管理", "g0v0 API"],
)
async def update_team_admin(
    session: Database,
    team_id: int,
    storage: StorageService,
    ip_address: IPAddress,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
    flag: bytes | None = File(None),
    cover: bytes | None = File(None),
    name: str | None = Form(None, max_length=100),
    short_name: str | None = Form(None, max_length=10),
    leader_id: int | None = Form(None),
    playmode: GameMode | None = Form(None),
    description: str | None = Form(None, max_length=2000),
    website: str | None = Form(None, max_length=255),
):
    """Update team (admin only)."""
    current_user = await require_admin(session, user_and_token)

    team = await session.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    changes = {}

    if name is not None:
        clean_name = name.strip()
        if not clean_name:
            raise HTTPException(status_code=400, detail="Team name cannot be empty")
        if (
            await session.exec(
                select(exists()).where(
                    Team.name == clean_name,
                    Team.id != team_id,
                )
            )
        ).first():
            raise HTTPException(status_code=409, detail="Name already exists")
        if team.name != clean_name:
            changes["name"] = {"old": team.name, "new": clean_name}
        team.name = clean_name

    if short_name is not None:
        clean_short_name = short_name.strip()
        if not clean_short_name:
            raise HTTPException(status_code=400, detail="Team short name cannot be empty")
        if (
            await session.exec(
                select(exists()).where(
                    Team.short_name == clean_short_name,
                    Team.id != team_id,
                )
            )
        ).first():
            raise HTTPException(status_code=409, detail="Short name already exists")
        if team.short_name != clean_short_name:
            changes["short_name"] = {"old": team.short_name, "new": clean_short_name}
        team.short_name = clean_short_name

    if playmode is not None and team.playmode != playmode:
        changes["playmode"] = {"old": str(team.playmode) if team.playmode else None, "new": str(playmode)}
        team.playmode = playmode

    if description is not None:
        clean_description = description.strip()
        old_desc = team.description or ""
        if old_desc != clean_description:
            changes["description"] = "updated"
        team.description = clean_description or None

    if website is not None:
        clean_website = website.strip()
        if clean_website and not (clean_website.startswith("http://") or clean_website.startswith("https://")):
            clean_website = "https://" + clean_website
        old_website = team.website or ""
        if old_website != clean_website:
            changes["website"] = "updated"
        team.website = clean_website or None

    if flag is not None:
        fmt = check_image(flag, 2 * 1024 * 1024, 240, 120)
        if old_flag := team.flag_url:
            if path := storage.get_file_name_by_url(old_flag):
                await storage.delete_file(path)
        filehash = hashlib.sha256(flag).hexdigest()
        storage_path = f"team_flag/{team.id}_{filehash}.png"
        if not await storage.is_exists(storage_path):
            await storage.write_file(storage_path, flag, f"image/{fmt}")
        team.flag_url = await storage.get_file_url(storage_path)
        changes["flag_url"] = "updated"

    if cover is not None:
        fmt = check_image(cover, 10 * 1024 * 1024, 3000, 2000)
        if old_cover := team.cover_url:
            if path := storage.get_file_name_by_url(old_cover):
                await storage.delete_file(path)
        filehash = hashlib.sha256(cover).hexdigest()
        storage_path = f"team_cover/{team.id}_{filehash}.png"
        if not await storage.is_exists(storage_path):
            await storage.write_file(storage_path, cover, f"image/{fmt}")
        team.cover_url = await storage.get_file_url(storage_path)
        changes["cover_url"] = "updated"

    if leader_id is not None:
        if not (await session.exec(select(exists()).where(User.id == leader_id))).first():
            raise HTTPException(status_code=404, detail="Leader not found")
        is_member = (
            await session.exec(
                select(exists()).where(
                    TeamMember.user_id == leader_id,
                    TeamMember.team_id == team_id,
                )
            )
        ).first()
        if not is_member:
            raise HTTPException(status_code=404, detail="Leader is not a member of the team")
        if team.leader_id != leader_id:
            changes["leader_id"] = {"old": team.leader_id, "new": leader_id}
        team.leader_id = leader_id

    await session.commit()
    await session.refresh(team)

    if changes:
        # Create audit log
        audit_log = AuditLog(
            actor_id=current_user.id,
            actor_username=current_user.username,
            action_type=AuditActionType.TEAM_UPDATE,
            target_type=TargetType.TEAM,
            target_id=team.id,
            target_name=team.name,
            reason="Team updated by admin",
            ip_address=ip_address,
            metadata={
                "team_id": team.id,
                "changes": changes,
            },
        )
        session.add(audit_log)
        await session.commit()

    redis = get_redis()
    cache_service = get_ranking_cache_service(redis)
    await cache_service.invalidate_team_cache()

    return team


@router.delete(
    "/admin/teams/{team_id}",
    name="删除战队",
    tags=["管理", "g0v0 API"],
    status_code=204,
)
async def delete_team_admin(
    session: Database,
    team_id: int,
    ip_address: IPAddress,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Delete a team (admin only)"""
    current_user = await require_admin(session, user_and_token)

    team = await session.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    # Store team info before deletion for audit log
    team_info = {
        "id": team.id,
        "name": team.name,
        "short_name": team.short_name,
        "leader_id": team.leader_id,
        "member_count": len(team.members) if hasattr(team, 'members') else 0,
    }

    # Create audit log
    audit_log = AuditLog(
        actor_id=current_user.id,
        actor_username=current_user.username,
        action_type=AuditActionType.TEAM_DISBAND,
        target_type=TargetType.TEAM,
        target_id=team.id,
        target_name=team.name,
        reason="Team deleted by admin",
        ip_address=ip_address,
        metadata=team_info,
    )
    session.add(audit_log)

    await session.delete(team)
    await session.commit()


# ========== Daily Challenge Statistics ==========

class DailyChallengeStatsResponse(BaseModel):
    user_id: int
    daily_streak_best: int = 0
    daily_streak_current: int = 0
    weekly_streak_best: int = 0
    weekly_streak_current: int = 0
    top_10p_placements: int = 0
    top_50p_placements: int = 0
    playcount: int = 0
    last_update: str | None = None  # ISO format
    last_weekly_streak: str | None = None  # ISO format


@router.get(
    "/admin/daily-challenge/stats/{user_id}",
    name="获取用户每日挑战统计",
    tags=["管理", "g0v0 API"],
    response_model=DailyChallengeStatsResponse,
)
async def get_daily_challenge_stats(
    session: Database,
    user_id: int,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Get daily challenge statistics for a user (admin only) - Matches osu.Game APIUserDailyChallengeStatistics"""
    await require_admin(session, user_and_token)

    # For now, return default stats. In a real implementation, this would query user statistics
    # from a dedicated daily challenge statistics table or calculate from scores
    return DailyChallengeStatsResponse(
        user_id=user_id,
        daily_streak_best=0,
        daily_streak_current=0,
        weekly_streak_best=0,
        weekly_streak_current=0,
        top_10p_placements=0,
        top_50p_placements=0,
        playcount=0,
        last_update=None,
        last_weekly_streak=None,
    )


# ========== Daily Challenge Management ==========

class DailyChallengeListResponse(BaseModel):
    total: int
    challenges: list[DailyChallengeResponse]
    page: int = 1
    per_page: int = 50


@router.get(
    "/admin/daily-challenges",
    name="获取每日挑战列表",
    tags=["管理", "g0v0 API"],
    response_model=DailyChallengeListResponse,
)
async def list_daily_challenges(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=100, description="Items per page"),
    date_from: str | None = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="End date (YYYY-MM-DD)"),
):
    """List daily challenges with pagination and optional date filtering - Enhanced for osu.Game compatibility"""
    await require_admin(session, user_and_token)

    # Build query
    query = select(DailyChallenge)

    # Apply date filtering if provided
    if date_from:
        try:
            from_date = datetime.strptime(date_from, "%Y-%m-%d").date()
            query = query.where(col(DailyChallenge.date) >= from_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date_from format. Use YYYY-MM-DD")

    if date_to:
        try:
            to_date = datetime.strptime(date_to, "%Y-%m-%d").date()
            query = query.where(col(DailyChallenge.date) <= to_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date_to format. Use YYYY-MM-DD")

    # Order by date descending (newest first)
    query = query.order_by(col(DailyChallenge.date).desc())

    # Get total count
    total_query = select(func.count()).select_from(DailyChallenge)
    if date_from:
        total_query = total_query.where(col(DailyChallenge.date) >= from_date)
    if date_to:
        total_query = total_query.where(col(DailyChallenge.date) <= to_date)

    total = (await session.exec(total_query)).one()

    # Apply pagination
    offset = (page - 1) * per_page
    query = query.offset(offset).limit(per_page)

    challenges = (await session.exec(query)).all()

    # Add beatmap info to each challenge
    challenges_with_beatmap = []
    for challenge in challenges:
        # Convert to response model to avoid SQLModel/Pydantic validation errors
        challenge_res = DailyChallengeResponse.model_validate(challenge)
        challenge_res.beatmap = await get_beatmap_info_for_response(session, challenge.beatmap_id)
        challenges_with_beatmap.append(challenge_res)

    return DailyChallengeListResponse(
        total=total,
        challenges=challenges_with_beatmap,
        page=page,
        per_page=per_page,
    )

@router.get(
    "/admin/daily-challenge/{date}",
    name="获取每日挑战",
    tags=["管理", "g0v0 API"],
    response_model=DailyChallengeResponse | None,
)
async def get_daily_challenge(
    session: Database,
    date: str,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Get daily challenge for a specific date (admin only)"""
    await require_admin(session, user_and_token)

    try:
        challenge_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    challenge = (
        await session.exec(select(DailyChallenge).where(col(DailyChallenge.date) == challenge_date))
    ).first()

    if not challenge:
        return None

    # Convert to response model to avoid SQLModel/Pydantic validation errors
    challenge_res = DailyChallengeResponse.model_validate(challenge)

    # Get beatmap info with error handling
    challenge_res.beatmap = await get_beatmap_info_for_response(session, challenge.beatmap_id)

    return challenge_res


@router.post(
    "/admin/daily-challenge/trigger",
    name="手动触发每日挑战",
    tags=["管理", "g0v0 API"],
)
async def trigger_daily_challenge(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Manually trigger the daily challenge job (admin only)"""
    await require_admin(session, user_and_token)

    from app.tasks.daily_challenge import daily_challenge_job
    await daily_challenge_job()

    return {"message": "Daily challenge job triggered successfully"}


@router.post(
    "/admin/daily-challenge",
    name="创建每日挑战",
    tags=["管理", "g0v0 API"],
    status_code=201,
    response_model=DailyChallengeResponse,
)
async def create_daily_challenge(
    session: Database,
    challenge_data: DailyChallengeCreate,
    ip_address: IPAddress,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Create a daily challenge (admin only) - Enhanced to match osu.Game Room structure"""
    current_user = await require_admin(session, user_and_token)

    try:
        challenge_date = datetime.strptime(challenge_data.date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    # Check if beatmap exists
    beatmap = await session.get(Beatmap, challenge_data.beatmap_id)
    if not beatmap:
        raise HTTPException(status_code=404, detail="Beatmap not found")

    # Check if challenge already exists for this date
    existing_challenge = (
        await session.exec(select(DailyChallenge).where(col(DailyChallenge.date) == challenge_date))
    ).first()

    if existing_challenge:
        raise HTTPException(status_code=409, detail="Daily challenge already exists for this date")

    # Check if room_id is already used (if provided)
    if hasattr(challenge_data, 'room_id') and challenge_data.room_id is not None:
        existing_room_challenge = (
            await session.exec(select(DailyChallenge).where(col(DailyChallenge.room_id) == challenge_data.room_id))
        ).first()
        if existing_room_challenge:
            raise HTTPException(status_code=409, detail="Room ID already in use by another daily challenge")

    # Create new challenge with enhanced fields
    new_challenge = DailyChallenge(
        date=challenge_date,
        beatmap_id=challenge_data.beatmap_id,
        ruleset_id=challenge_data.ruleset_id,
        required_mods=challenge_data.required_mods,
        allowed_mods=challenge_data.allowed_mods,
        room_id=getattr(challenge_data, 'room_id', None),
        max_attempts=getattr(challenge_data, 'max_attempts', None),
        time_limit=getattr(challenge_data, 'time_limit', None),
    )

    # Sync to Redis and optionally create room
    room_id = await sync_daily_challenge_to_redis(
        challenge_date=challenge_date,
        beatmap_id=new_challenge.beatmap_id,
        ruleset_id=new_challenge.ruleset_id,
        required_mods=challenge_data.required_mods,
        allowed_mods=challenge_data.allowed_mods,
        room_id=new_challenge.room_id,
    )
    if room_id is not None:
        new_challenge.room_id = room_id

    session.add(new_challenge)

    # Create audit log
    audit_log = AuditLog(
        actor_id=current_user.id,
        actor_username=current_user.username,
        action_type=AuditActionType.ANNOUNCEMENT_CREATE,
        target_type=TargetType.SYSTEM,
        target_id=None,
        target_name="Daily Challenge",
        reason=f"Created daily challenge for date {challenge_date}",
        ip_address=ip_address,
        metadata={
            "date": str(challenge_date),
            "beatmap_id": challenge_data.beatmap_id,
            "ruleset_id": challenge_data.ruleset_id,
            "room_id": room_id,
            "required_mods": challenge_data.required_mods,
            "allowed_mods": challenge_data.allowed_mods,
        },
    )
    session.add(audit_log)
    await session.commit()
    await session.refresh(new_challenge)

    # Add beatmap info to response
    challenge_res = DailyChallengeResponse.model_validate(new_challenge)
    challenge_res.beatmap = await get_beatmap_info_for_response(session, new_challenge.beatmap_id)

    return challenge_res


@router.patch(
    "/admin/daily-challenge/{date}",
    name="更新每日挑战",
    tags=["管理", "g0v0 API"],
    response_model=DailyChallengeResponse,
)
async def update_daily_challenge(
    session: Database,
    date: str,
    challenge_data: DailyChallengeUpdate,
    ip_address: IPAddress,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Update a daily challenge (admin only) - Enhanced with new fields"""
    current_user = await require_admin(session, user_and_token)

    try:
        challenge_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    challenge = (
        await session.exec(select(DailyChallenge).where(col(DailyChallenge.date) == challenge_date))
    ).first()

    if not challenge:
        raise HTTPException(status_code=404, detail="Daily challenge not found")

    # Track changes
    changes = {}

    # Update fields if provided
    if getattr(challenge_data, 'beatmap_id', None) is not None:
        # Check if new beatmap exists
        beatmap = await session.get(Beatmap, challenge_data.beatmap_id)
        if not beatmap:
            raise HTTPException(status_code=404, detail="Beatmap not found")
        if challenge.beatmap_id != challenge_data.beatmap_id:
            changes["beatmap_id"] = {"old": challenge.beatmap_id, "new": challenge_data.beatmap_id}
        challenge.beatmap_id = challenge_data.beatmap_id

    if getattr(challenge_data, 'ruleset_id', None) is not None and challenge.ruleset_id != challenge_data.ruleset_id:
        changes["ruleset_id"] = {"old": challenge.ruleset_id, "new": challenge_data.ruleset_id}
        challenge.ruleset_id = challenge_data.ruleset_id

    if getattr(challenge_data, 'required_mods', None) is not None and challenge.required_mods != challenge_data.required_mods:
        changes["required_mods"] = {"old": challenge.required_mods, "new": challenge_data.required_mods}
        challenge.required_mods = challenge_data.required_mods

    if getattr(challenge_data, 'allowed_mods', None) is not None and challenge.allowed_mods != challenge_data.allowed_mods:
        changes["allowed_mods"] = {"old": challenge.allowed_mods, "new": challenge_data.allowed_mods}
        challenge.allowed_mods = challenge_data.allowed_mods

    if getattr(challenge_data, 'room_id', None) is not None:
        # Check if room_id is already used by another challenge
        existing_room_challenge = (
            await session.exec(
                select(DailyChallenge)
                .where(
                    col(DailyChallenge.room_id) == challenge_data.room_id,
                    col(DailyChallenge.date) != challenge_date
                )
            )
        ).first()
        if existing_room_challenge:
            raise HTTPException(status_code=409, detail="Room ID already in use by another daily challenge")
        if challenge.room_id != challenge_data.room_id:
            changes["room_id"] = {"old": challenge.room_id, "new": challenge_data.room_id}
        challenge.room_id = challenge_data.room_id

    if getattr(challenge_data, 'max_attempts', None) is not None and challenge.max_attempts != challenge_data.max_attempts:
        changes["max_attempts"] = {"old": challenge.max_attempts, "new": challenge_data.max_attempts}
        challenge.max_attempts = challenge_data.max_attempts

    if getattr(challenge_data, 'time_limit', None) is not None and challenge.time_limit != challenge_data.time_limit:
        changes["time_limit"] = {"old": challenge.time_limit, "new": challenge_data.time_limit}
        challenge.time_limit = challenge_data.time_limit

    # Sync to Redis and optionally create room
    room_id = await sync_daily_challenge_to_redis(
        challenge_date=challenge_date,
        beatmap_id=challenge.beatmap_id,
        ruleset_id=challenge.ruleset_id,
        required_mods=challenge.required_mods,
        allowed_mods=challenge.allowed_mods,
        room_id=challenge.room_id,
    )
    if room_id is not None:
        challenge.room_id = room_id

    await session.commit()
    await session.refresh(challenge)

    if changes:
        # Create audit log
        audit_log = AuditLog(
            actor_id=current_user.id,
            actor_username=current_user.username,
            action_type=AuditActionType.ANNOUNCEMENT_UPDATE,
            target_type=TargetType.SYSTEM,
            target_id=None,
            target_name="Daily Challenge",
            reason=f"Updated daily challenge for date {challenge_date}",
            ip_address=ip_address,
            metadata={
                "date": str(challenge_date),
                "changes": changes,
            },
        )
        session.add(audit_log)
        await session.commit()

    # Add beatmap info to response
    challenge_res = DailyChallengeResponse.model_validate(challenge)
    challenge_res.beatmap = await get_beatmap_info_for_response(session, challenge.beatmap_id)

    return challenge_res


@router.delete(
    "/admin/daily-challenge/{date}",
    name="删除每日挑战",
    tags=["管理", "g0v0 API"],
    status_code=204,
)
async def delete_daily_challenge(
    session: Database,
    date: str,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Delete a daily challenge (admin only)"""
    await require_admin(session, user_and_token)

    try:
        challenge_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    challenge = (
        await session.exec(select(DailyChallenge).where(col(DailyChallenge.date) == challenge_date))
    ).first()

    if not challenge:
        raise HTTPException(status_code=404, detail="Daily challenge not found")

    # Clean up Redis
    redis = get_redis()
    redis_key = f"daily_challenge:{challenge_date}"
    await redis.delete(redis_key)

    await session.delete(challenge)
    await session.commit()


# ========== Reports System ==========

class ReportListResponse(BaseModel):
    total: int
    page: int
    per_page: int
    reports: list[ReportResponse]


@router.get(
    "/admin/reports",
    name="获取举报列表",
    tags=["管理", "g0v0 API"],
    response_model=ReportListResponse,
)
async def get_reports(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    status: ReportStatus | None = Query(None),
    search: str = Query(""),
):
    """Get all reports with pagination and filtering (admin only)"""
    await require_admin(session, user_and_token)

    conditions = []

    if status:
        conditions.append(col(Report.status) == status)

    if search:
        search_term = f"%{search}%"
        conditions.append(
            sql_or(
                col(Report.reason).ilike(search_term),
                col(Report.description).ilike(search_term),
            )
        )

    count_stmt = select(func.count()).select_from(Report)
    data_stmt = select(Report)

    if conditions:
        count_stmt = count_stmt.where(*conditions)
        data_stmt = data_stmt.where(*conditions)

    total = (await session.exec(count_stmt)).one()

    reports = (
        await session.exec(
            data_stmt.order_by(col(Report.created_at).desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
        )
    ).all()

    # Get usernames for reporter, reported user, and resolver
    user_ids = set()
    for report in reports:
        if report.reporter_id:
            user_ids.add(report.reporter_id)
        if report.reported_user_id:
            user_ids.add(report.reported_user_id)
        if report.resolved_by:
            user_ids.add(report.resolved_by)

    username_map: dict[int, str] = {}
    if user_ids:
        users = (
            await session.exec(
                select(User.id, User.username).where(col(User.id).in_(user_ids))
            )
        ).all()
        username_map = {uid: uname for uid, uname in users}

    report_responses = []
    for report in reports:
        report_dict = report.model_dump()
        report_dict["reporter_username"] = username_map.get(report.reporter_id)
        report_dict["reported_username"] = username_map.get(report.reported_user_id)
        report_dict["resolved_by_username"] = username_map.get(report.resolved_by)
        report_responses.append(ReportResponse(**report_dict))

    return ReportListResponse(
        total=total,
        page=page,
        per_page=per_page,
        reports=report_responses,
    )


class ReportResolveRequest(BaseModel):
    action: str  # "close", "ban", "warn"
    notes: str | None = None


@router.post(
    "/admin/reports/{report_id}/resolve",
    name="解决举报",
    tags=["管理", "g0v0 API"],
)
async def resolve_report(
    session: Database,
    report_id: int,
    request: ReportResolveRequest,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Resolve a report (admin only)"""
    current_user = await require_admin(session, user_and_token)

    report = await session.get(Report, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    if report.status != ReportStatus.PENDING:
        raise HTTPException(status_code=400, detail="Report is already resolved")

    report.status = ReportStatus.RESOLVED
    report.resolved_by = current_user.id
    report.resolution_action = request.action
    report.resolution_notes = request.notes
    report.resolved_at = datetime.utcnow()

    await session.commit()
    await session.refresh(report)

    return {"message": "Report resolved successfully", "report_id": report_id}


# ========== Announcements System ==========

class AnnouncementListResponse(BaseModel):
    total: int
    page: int
    per_page: int
    announcements: list[AnnouncementResponse]


@router.get(
    "/admin/announcements",
    name="获取公告列表",
    tags=["管理", "g0v0 API"],
    response_model=AnnouncementListResponse,
)
async def get_announcements(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    is_active: bool | None = Query(None),
    type: AnnouncementType | None = Query(None),
    include_expired: bool = Query(False),
):
    """Get all announcements with pagination and filtering (admin only)"""
    await require_admin(session, user_and_token)

    conditions = []

    if is_active is not None:
        conditions.append(col(Announcement.is_active) == is_active)

    if type:
        conditions.append(col(Announcement.type) == type)

    if not include_expired:
        conditions.append(
            sql_or(
                col(Announcement.end_at).is_(None),
                col(Announcement.end_at) > datetime.utcnow(),
            )
        )

    count_stmt = select(func.count()).select_from(Announcement)
    data_stmt = select(Announcement)

    if conditions:
        count_stmt = count_stmt.where(*conditions)
        data_stmt = data_stmt.where(*conditions)

    total = (await session.exec(count_stmt)).one()

    announcements = (
        await session.exec(
            data_stmt.order_by(col(Announcement.created_at).desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
        )
    ).all()

    # Get creator usernames
    creator_ids = {a.created_by for a in announcements if a.created_by}
    username_map: dict[int, str] = {}
    if creator_ids:
        users = (
            await session.exec(
                select(User.id, User.username).where(col(User.id).in_(creator_ids))
            )
        ).all()
        username_map = {uid: uname for uid, uname in users}

    announcement_responses = []
    for announcement in announcements:
        announcement_dict = announcement.model_dump()
        announcement_dict["created_by_username"] = username_map.get(announcement.created_by)
        announcement_responses.append(AnnouncementResponse(**announcement_dict))

    return AnnouncementListResponse(
        total=total,
        page=page,
        per_page=per_page,
        announcements=announcement_responses,
    )


@router.post(
    "/admin/announcements",
    name="创建公告",
    tags=["管理", "g0v0 API"],
    status_code=201,
    response_model=AnnouncementResponse,
)
async def create_announcement(
    session: Database,
    announcement_data: AnnouncementCreate,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Create a new announcement (admin only)"""
    current_user = await require_admin(session, user_and_token)

    # Capture user attributes immediately to avoid lazy-loading issues after commit
    current_user_id = current_user.id
    current_username = current_user.username

    new_announcement = Announcement(
        title=announcement_data.title,
        content=announcement_data.content,
        type=announcement_data.type,
        target_roles=announcement_data.target_roles,
        start_at=announcement_data.start_at or datetime.utcnow(),
        end_at=announcement_data.end_at,
        is_active=announcement_data.is_active,
        is_pinned=announcement_data.is_pinned,
        show_in_client=announcement_data.show_in_client,
        show_on_website=announcement_data.show_on_website,
        created_by=current_user_id,
        metadata=announcement_data.metadata,
    )

    # Capture username BEFORE commit to avoid lazy-loading issues
    created_by_username = current_user.username

    session.add(new_announcement)
    await session.commit()
    await session.refresh(new_announcement)

    # Send notification if announcement is created as active and should be shown in client
    if new_announcement.is_active and new_announcement.show_in_client:
        try:
            from app.service.announcement_notification_service import trigger_announcement_notification
            await trigger_announcement_notification(
                session=session,
                announcement=new_announcement,
                current_user_id=current_user_id,
                online_only=False,
            )
        except Exception as e:
            logger.error(f"Failed to send notification for new announcement {new_announcement.id}: {e}")

    announcement_dict = new_announcement.model_dump()
    announcement_dict["created_by_username"] = created_by_username

    return AnnouncementResponse(**announcement_dict)


@router.put(
    "/admin/announcements/{announcement_id}",
    name="更新公告",
    tags=["管理", "g0v0 API"],
    response_model=AnnouncementResponse,
)
async def update_announcement(
    session: Database,
    announcement_id: int,
    announcement_data: AnnouncementUpdate,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Update an announcement (admin only)"""
    await require_admin(session, user_and_token)

    announcement = await session.get(Announcement, announcement_id)
    if not announcement:
        raise HTTPException(status_code=404, detail="Announcement not found")

    if announcement_data.title is not None:
        announcement.title = announcement_data.title
    if announcement_data.content is not None:
        announcement.content = announcement_data.content
    if announcement_data.type is not None:
        announcement.type = announcement_data.type
    if announcement_data.target_roles is not None:
        announcement.target_roles = announcement_data.target_roles
    if announcement_data.start_at is not None:
        announcement.start_at = announcement_data.start_at
    if announcement_data.end_at is not None:
        announcement.end_at = announcement_data.end_at
    if announcement_data.is_active is not None:
        announcement.is_active = announcement_data.is_active
    if announcement_data.is_pinned is not None:
        announcement.is_pinned = announcement_data.is_pinned
    if announcement_data.show_in_client is not None:
        announcement.show_in_client = announcement_data.show_in_client
    if announcement_data.show_on_website is not None:
        announcement.show_on_website = announcement_data.show_on_website
    if announcement_data.metadata is not None:
        announcement.metadata = announcement_data.metadata

    announcement.updated_at = datetime.utcnow()

    # Check if announcement is being activated and should show in client
    was_inactive = not announcement.is_active

    await session.commit()
    await session.refresh(announcement)

    # Send notification if announcement was activated and should be shown in client
    if was_inactive and announcement.is_active and announcement.show_in_client:
        try:
            from app.service.announcement_notification_service import trigger_announcement_notification

            await trigger_announcement_notification(
                session=session,
                announcement=announcement,
                current_user_id=announcement.created_by,
                online_only=False,
            )
        except Exception as e:
            logger.error(f"Failed to send notification for activated announcement {announcement.id}: {e}")

    # Get creator username (use fresh session lookup to avoid lazy-load issues)
    creator = await session.get(User, announcement.created_by)
    creator_username = creator.username if creator else None

    announcement_dict = announcement.model_dump()
    announcement_dict["created_by_username"] = creator_username

    return AnnouncementResponse(**announcement_dict)


@router.delete(
    "/admin/announcements/{announcement_id}",
    name="删除公告",
    tags=["管理", "g0v0 API"],
    status_code=204,
)
async def delete_announcement(
    session: Database,
    announcement_id: int,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Delete an announcement (admin only)"""
    await require_admin(session, user_and_token)

    announcement = await session.get(Announcement, announcement_id)
    if not announcement:
        raise HTTPException(status_code=404, detail="Announcement not found")

    await session.delete(announcement)
    await session.commit()


@router.post(
    "/admin/announcements/{announcement_id}/activate",
    name="激活公告",
    tags=["管理", "g0v0 API"],
)
async def activate_announcement(
    session: Database,
    announcement_id: int,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
    send_notification: bool = Query(False, description="是否发送通知到客户端"),
    online_only: bool = Query(False, description="是否只通知在线用户"),
):
    """Activate an announcement (admin only)"""
    from app.service.announcement_notification_service import trigger_announcement_notification

    current_user = await require_admin(session, user_and_token)

    # Capture user attributes immediately to avoid lazy-loading issues after commit
    current_user_id = current_user.id

    announcement = await session.get(Announcement, announcement_id)
    if not announcement:
        raise HTTPException(status_code=404, detail="Announcement not found")

    was_inactive = not announcement.is_active

    announcement.is_active = True
    announcement.updated_at = datetime.utcnow()

    await session.commit()
    await session.refresh(announcement)

    # Send notification if requested and announcement should be shown in client
    notification_id = None
    if was_inactive and send_notification and announcement.show_in_client:
        try:
            notification_id = await trigger_announcement_notification(
                session=session,
                announcement=announcement,
                current_user_id=current_user_id,
                online_only=online_only,
            )
        except Exception as e:
            # Log error but don't fail the activation
            logger.error(f"Failed to send notification for announcement {announcement_id}: {e}")

    response = {"message": "Announcement activated successfully"}
    if notification_id:
        response["notification_id"] = notification_id
    if send_notification and not announcement.show_in_client:
        response["warning"] = "Notification not sent because show_in_client is false"

    return response


@router.post(
    "/admin/announcements/{announcement_id}/deactivate",
    name="停用公告",
    tags=["管理", "g0v0 API"],
)
async def deactivate_announcement(
    session: Database,
    announcement_id: int,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Deactivate an announcement (admin only)"""
    await require_admin(session, user_and_token)

    announcement = await session.get(Announcement, announcement_id)
    if not announcement:
        raise HTTPException(status_code=404, detail="Announcement not found")

    announcement.is_active = False
    announcement.updated_at = datetime.utcnow()

    await session.commit()
    await session.refresh(announcement)

    return {"message": "Announcement deactivated successfully"}


# ========== System Tools ==========

class MaintenanceModeResponse(BaseModel):
    enabled: bool
    message: str | None = None
    updated_at: str | None = None


@router.get(
    "/admin/system/maintenance-mode",
    name="获取维护模式状态",
    tags=["管理", "g0v0 API"],
    response_model=MaintenanceModeResponse,
)
async def get_maintenance_mode(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Get maintenance mode status (dev or admin only)"""
    await require_dev_or_admin(session, user_and_token)

    setting = (
        await session.exec(
            select(SystemSetting).where(SystemSetting.key == "maintenance_mode")
        )
    ).first()

    if not setting:
        return MaintenanceModeResponse(enabled=False)

    return MaintenanceModeResponse(
        enabled=setting.value.lower() == "true",
        message=setting.description,
        updated_at=setting.updated_at.isoformat() if setting.updated_at else None,
    )


class MaintenanceModeScheduleRequest(BaseModel):
    enabled: bool
    message: str | None = None
    schedule_minutes: int | None = Field(None, ge=0, le=60, description="Minutes until maintenance starts (0 = immediate)")


class MaintenanceModeScheduleResponse(BaseModel):
    enabled: bool
    scheduled: bool
    countdown_minutes: int | None = None
    message: str | None = None
    countdown_announcement_id: int | None = None
    maintenance_announcement_id: int | None = None
    updated_at: str | None = None


class MaintenanceModeStatusResponse(BaseModel):
    enabled: bool
    scheduled: bool
    countdown_active: bool
    countdown_minutes_remaining: int | None = None
    countdown_end_time: str | None = None
    message: str | None = None
    countdown_announcement_id: int | None = None
    maintenance_announcement_id: int | None = None
    updated_at: str | None = None


# In-memory state for scheduled maintenance (will be reset on server restart)
_scheduled_maintenance: dict[str, Any] = {
    "active": False,
    "end_time": None,
    "countdown_announcement_id": None,
    "maintenance_announcement_id": None,
}


@router.get(
    "/admin/system/maintenance-mode/status",
    name="获取维护模式详细状态",
    tags=["管理", "g0v0 API"],
    response_model=MaintenanceModeStatusResponse,
)
async def get_maintenance_mode_status(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Get detailed maintenance mode status including countdown (admin only)"""
    await require_dev_or_admin(session, user_and_token)

    global _scheduled_maintenance

    setting = (
        await session.exec(
            select(SystemSetting).where(SystemSetting.key == "maintenance_mode")
        )
    ).first()

    enabled = setting.value.lower() == "true" if setting else False
    countdown_active = _scheduled_maintenance["active"] and not enabled
    countdown_remaining = None
    countdown_end_time = None

    if countdown_active and _scheduled_maintenance["end_time"]:
        end_time = datetime.fromisoformat(_scheduled_maintenance["end_time"])
        now = datetime.utcnow()
        if end_time > now:
            countdown_remaining = int((end_time - now).total_seconds() / 60)
            countdown_end_time = _scheduled_maintenance["end_time"]
        else:
            # Countdown expired, enable maintenance
            countdown_active = False
            _scheduled_maintenance["active"] = False

    return MaintenanceModeStatusResponse(
        enabled=enabled,
        scheduled=_scheduled_maintenance["active"] or enabled,
        countdown_active=countdown_active,
        countdown_minutes_remaining=countdown_remaining,
        countdown_end_time=countdown_end_time,
        message=setting.description if setting else None,
        countdown_announcement_id=_scheduled_maintenance.get("countdown_announcement_id"),
        maintenance_announcement_id=_scheduled_maintenance.get("maintenance_announcement_id"),
        updated_at=setting.updated_at.isoformat() if setting and setting.updated_at else None,
    )


@router.post(
    "/admin/system/maintenance-mode/schedule",
    name="设置维护模式（支持倒计时）",
    tags=["管理", "g0v0 API"],
    response_model=MaintenanceModeScheduleResponse,
)
async def schedule_maintenance_mode(
    session: Database,
    request: MaintenanceModeScheduleRequest,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Set maintenance mode with optional countdown timer (dev or admin only)"""
    from app.database.announcement import Announcement, AnnouncementType
    from app.service.announcement_notification_service import trigger_announcement_notification

    current_user = await require_dev_or_admin(session, user_and_token)

    global _scheduled_maintenance

    countdown_announcement_id = None
    maintenance_announcement_id = None

    # If disabling maintenance, clear scheduled state
    if not request.enabled:
        _scheduled_maintenance["active"] = False
        _scheduled_maintenance["end_time"] = None

        # Deactivate any active maintenance announcements
        active_announcements = (
            await session.exec(
                select(Announcement).where(
                    col(Announcement.type) == AnnouncementType.MAINTENANCE,
                    col(Announcement.is_active) == True,
                )
            )
        ).all()

        for ann in active_announcements:
            ann.is_active = False
            ann.end_at = datetime.utcnow()

        await session.commit()

        return MaintenanceModeScheduleResponse(
            enabled=False,
            scheduled=False,
            message="Maintenance mode disabled",
            updated_at=datetime.utcnow().isoformat(),
        )

    # If countdown requested
    if request.schedule_minutes and request.schedule_minutes > 0:
        end_time = datetime.utcnow() + timedelta(minutes=request.schedule_minutes)

        # Create countdown announcement
        countdown_message = request.message or f"Score submissions will be disabled in {request.schedule_minutes} minutes for maintenance."

        countdown_announcement = Announcement(
            title="Maintenance Notice",
            message=countdown_message,
            type=AnnouncementType.MAINTENANCE,
            target_audience="ALL",
            is_active=True,
            start_at=datetime.utcnow(),
            end_at=end_time + timedelta(minutes=1),
            created_by=current_user_id,
        )
        session.add(countdown_announcement)
        await session.commit()
        await session.refresh(countdown_announcement)

        countdown_announcement_id = countdown_announcement.id

        # Schedule the actual maintenance after countdown
        _scheduled_maintenance["active"] = True
        _scheduled_maintenance["end_time"] = end_time.isoformat()
        _scheduled_maintenance["countdown_announcement_id"] = countdown_announcement_id

        # Create maintenance announcement (inactive until countdown ends)
        maintenance_message = request.message or "Maintenance mode is now active. Score submissions are temporarily disabled."

        maintenance_announcement = Announcement(
            title="Maintenance Mode Active",
            message=maintenance_message,
            type=AnnouncementType.MAINTENANCE,
            target_audience="ALL",
            is_active=False,  # Will be activated when countdown hits 0
            start_at=end_time,
            end_at=None,
            created_by=current_user_id,
        )
        session.add(maintenance_announcement)
        await session.commit()
        await session.refresh(maintenance_announcement)

        _scheduled_maintenance["maintenance_announcement_id"] = maintenance_announcement.id

        # Broadcast the countdown announcement
        await trigger_announcement_notification(session, countdown_announcement, current_user_id)

        return MaintenanceModeScheduleResponse(
            enabled=False,  # Not yet enabled
            scheduled=True,
            countdown_minutes=request.schedule_minutes,
            message=countdown_message,
            countdown_announcement_id=countdown_announcement_id,
            maintenance_announcement_id=maintenance_announcement_id,
            updated_at=datetime.utcnow().isoformat(),
        )

    # Immediate maintenance mode
    setting = (
        await session.exec(
            select(SystemSetting).where(SystemSetting.key == "maintenance_mode")
        )
    ).first()

    now = datetime.utcnow()

    if setting:
        setting.value = "true"
        setting.description = request.message
        setting.updated_by = current_user.id
        setting.updated_at = now
    else:
        setting = SystemSetting(
            key="maintenance_mode",
            value="true",
            value_type="bool",
            description=request.message,
            updated_by=current_user.id,
            updated_at=now,
        )
        session.add(setting)

    # Create immediate maintenance announcement
    maintenance_message = request.message or "Maintenance mode is now active. Score submissions are temporarily disabled."

    maintenance_announcement = Announcement(
        title="Maintenance Mode Active",
        message=maintenance_message,
        type=AnnouncementType.MAINTENANCE,
        target_audience="ALL",
        is_active=True,
        start_at=now,
        end_at=None,
        created_by=current_user_id,
    )
    session.add(maintenance_announcement)
    await session.commit()
    await session.refresh(maintenance_announcement)

    maintenance_announcement_id = maintenance_announcement.id
    _scheduled_maintenance["maintenance_announcement_id"] = maintenance_announcement_id

    # Broadcast the announcement
    await trigger_announcement_notification(session, maintenance_announcement, current_user.id)

    return MaintenanceModeScheduleResponse(
        enabled=True,
        scheduled=False,
        message=maintenance_message,
        maintenance_announcement_id=maintenance_announcement_id,
        updated_at=now.isoformat(),
    )


class RecalculationResponse(BaseModel):
    task_id: int
    status: str
    message: str


@router.post(
    "/admin/recalculate/user/{user_id}",
    name="重新计算用户PP",
    tags=["管理", "g0v0 API"],
    response_model=RecalculationResponse,
)
async def recalculate_user(
    session: Database,
    user_id: int,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Trigger PP recalculation for a specific user (admin only)"""
    current_user = await require_admin(session, user_and_token)

    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Check if a task for this user is already pending or running
    if await has_pending_or_running_task(session, RecalculationType.USER, user_id):
        raise HTTPException(
            status_code=400,
            detail=f"A recalculation task for user {user_id} is already pending or running",
        )

    # Check if an overall recalculation is running
    overall_running = await session.exec(
        select(RecalculationTask).where(
            RecalculationTask.task_type == RecalculationType.OVERALL,
            RecalculationTask.status == RecalculationStatus.RUNNING
        )
    )
    if overall_running.first():
        raise HTTPException(
            status_code=400,
            detail="An overall recalculation is currently running. Please wait for it to complete.",
        )

    # Check concurrent task limit
    if await check_concurrent_limit(session):
        raise HTTPException(
            status_code=429,
            detail="A recalculation task is already running. Please wait for it to complete.",
        )

    # Create recalculation task
    task = RecalculationTask(
        task_type=RecalculationType.USER,
        target_id=user_id,
        status=RecalculationStatus.PENDING,
        created_by=current_user_id,
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)

    # Trigger immediate processing in background
    import asyncio
    asyncio.create_task(process_pending_recalculation_tasks())

    return RecalculationResponse(
        task_id=task.id,
        status="pending",
        message=f"PP recalculation queued for user {user_id}",
    )


@router.post(
    "/admin/recalculate/beatmap/{beatmap_id}",
    name="重新计算谱面PP",
    tags=["管理", "g0v0 API"],
    response_model=RecalculationResponse,
)
async def recalculate_beatmap(
    session: Database,
    beatmap_id: int,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Trigger PP recalculation for a specific beatmap (admin only)"""
    current_user = await require_admin(session, user_and_token)

    beatmap = await session.get(Beatmap, beatmap_id)
    if not beatmap:
        raise HTTPException(status_code=404, detail="Beatmap not found")

    # Check if a task for this beatmap is already pending or running
    if await has_pending_or_running_task(session, RecalculationType.BEATMAP, beatmap_id):
        raise HTTPException(
            status_code=400,
            detail=f"A recalculation task for beatmap {beatmap_id} is already pending or running",
        )

    # Check if an overall recalculation is running
    overall_running = await session.exec(
        select(RecalculationTask).where(
            RecalculationTask.task_type == RecalculationType.OVERALL,
            RecalculationTask.status == RecalculationStatus.RUNNING
        )
    )
    if overall_running.first():
        raise HTTPException(
            status_code=400,
            detail="An overall recalculation is currently running. Please wait for it to complete.",
        )

    # Check concurrent task limit
    if await check_concurrent_limit(session):
        raise HTTPException(
            status_code=429,
            detail="A recalculation task is already running. Please wait for it to complete.",
        )

    # Create recalculation task
    task = RecalculationTask(
        task_type=RecalculationType.BEATMAP,
        target_id=beatmap_id,
        status=RecalculationStatus.PENDING,
        created_by=current_user_id,
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)

    # Trigger immediate processing in background
    import asyncio
    asyncio.create_task(process_pending_recalculation_tasks())

    return RecalculationResponse(
        task_id=task.id,
        status="pending",
        message=f"PP recalculation queued for beatmap {beatmap_id}",
    )


@router.post(
    "/admin/recalculate/overall",
    name="重新计算整体PP",
    tags=["管理", "g0v0 API"],
    response_model=RecalculationResponse,
)
async def recalculate_overall(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Trigger overall PP recalculation (admin only)"""
    current_user = await require_admin(session, user_and_token)

    # Check if overall recalculation is already pending or running
    if await has_pending_or_running_task(session, RecalculationType.OVERALL):
        raise HTTPException(
            status_code=400,
            detail="An overall recalculation is already pending or running. Please wait for it to complete.",
        )

    # Check if any recalculation is currently running
    if await check_concurrent_limit(session):
        raise HTTPException(
            status_code=429,
            detail="A recalculation task is already running. Please wait for it to complete.",
        )

    # Create recalculation task
    task = RecalculationTask(
        task_type=RecalculationType.OVERALL,
        status=RecalculationStatus.PENDING,
        created_by=current_user_id,
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)

    # Trigger immediate processing in background
    import asyncio
    asyncio.create_task(process_pending_recalculation_tasks())

    return RecalculationResponse(
        task_id=task.id,
        status="pending",
        message="Overall PP recalculation queued",
    )


class RecalculationTasksResponse(BaseModel):
    total: int
    tasks: list[RecalculationTaskResponse]


@router.get(
    "/admin/recalculate/tasks",
    name="获取重新计算任务列表",
    tags=["管理", "g0v0 API"],
    response_model=RecalculationTasksResponse,
)
async def get_recalculation_tasks(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
    status: RecalculationStatus | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    """Get recalculation tasks (admin only)"""
    await require_admin(session, user_and_token)

    conditions = []
    if status:
        conditions.append(col(RecalculationTask.status) == status)

    count_stmt = select(func.count()).select_from(RecalculationTask)
    data_stmt = select(RecalculationTask)

    if conditions:
        count_stmt = count_stmt.where(*conditions)
        data_stmt = data_stmt.where(*conditions)

    total = (await session.exec(count_stmt)).one()

    tasks = (
        await session.exec(
            data_stmt.order_by(col(RecalculationTask.created_at).desc()).limit(limit)
        )
    ).all()

    # Get creator usernames
    creator_ids = {t.created_by for t in tasks if t.created_by}
    username_map: dict[int, str] = {}
    if creator_ids:
        users = (
            await session.exec(
                select(User.id, User.username).where(col(User.id).in_(creator_ids))
            )
        ).all()
        username_map = {uid: uname for uid, uname in users}

    task_responses = []
    for task in tasks:
        task_dict = task.model_dump()
        task_dict["created_by_username"] = username_map.get(task.created_by)
        task_responses.append(RecalculationTaskResponse(**task_dict))

    return RecalculationTasksResponse(total=total, tasks=task_responses)


@router.get(
    "/admin/recalculate/status",
    name="获取重新计算状态",
    tags=["管理", "g0v0 API"],
)
async def get_recalculation_status(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Get current recalculation status (admin only)"""
    await require_admin(session, user_and_token)

    status = await get_current_task_status(session)
    return status


# ========== Logging System ==========

class AuditLogListResponse(BaseModel):
    total: int
    page: int
    per_page: int
    logs: list[AuditLogResponse]


@router.get(
    "/admin/logs/audit-logs",
    name="获取审计日志",
    tags=["管理", "g0v0 API"],
    response_model=AuditLogListResponse,
)
async def get_audit_logs(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    action_type: AuditActionType | None = Query(None),
    target_type: TargetType | None = Query(None),
    search: str = Query(""),
):
    """Get audit logs with pagination and filtering (admin only)"""
    await require_admin(session, user_and_token)

    conditions = []

    if action_type:
        conditions.append(col(AuditLog.action_type) == action_type)

    if target_type:
        conditions.append(col(AuditLog.target_type) == target_type)

    if search:
        search_term = f"%{search}%"
        conditions.append(
            sql_or(
                col(AuditLog.actor_username).ilike(search_term),
                col(AuditLog.target_name).ilike(search_term),
                col(AuditLog.reason).ilike(search_term),
            )
        )

    count_stmt = select(func.count()).select_from(AuditLog)
    data_stmt = select(AuditLog)

    if conditions:
        count_stmt = count_stmt.where(*conditions)
        data_stmt = data_stmt.where(*conditions)

    total = (await session.exec(count_stmt)).one()

    logs = (
        await session.exec(
            data_stmt.order_by(col(AuditLog.created_at).desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
        )
    ).all()

    log_responses = [AuditLogResponse(**log.model_dump()) for log in logs]

    return AuditLogListResponse(
        total=total,
        page=page,
        per_page=per_page,
        logs=log_responses,
    )


class ClientLogListResponse(BaseModel):
    total: int
    page: int
    per_page: int
    logs: list[ClientLogResponse]


@router.get(
    "/admin/logs/client-logs",
    name="获取客户端日志",
    tags=["管理", "g0v0 API"],
    response_model=ClientLogListResponse,
)
async def get_client_logs(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    log_type: ClientLogType | None = Query(None),
    user_id: int | None = Query(None),
    client_version: str | None = Query(None),
    client_hash: str | None = Query(None),
    search: str = Query(""),
):
    """Get client logs with pagination and filtering (admin only)"""
    await require_admin(session, user_and_token)

    conditions = []

    if log_type:
        conditions.append(col(ClientLog.log_type) == log_type)

    if user_id:
        conditions.append(col(ClientLog.user_id) == user_id)

    if client_version:
        conditions.append(col(ClientLog.client_version).ilike(f"%{client_version}%"))

    if client_hash:
        conditions.append(col(ClientLog.client_hash).ilike(f"%{client_hash}%"))

    if search:
        search_term = f"%{search}%"
        conditions.append(
            sql_or(
                col(ClientLog.username).ilike(search_term),
                col(ClientLog.client_version).ilike(search_term),
                col(ClientLog.client_hash).ilike(search_term),
                col(ClientLog.message).ilike(search_term),
                col(ClientLog.os_version).ilike(search_term),
            )
        )

    count_stmt = select(func.count()).select_from(ClientLog)
    data_stmt = select(ClientLog)

    if conditions:
        count_stmt = count_stmt.where(*conditions)
        data_stmt = data_stmt.where(*conditions)

    total = (await session.exec(count_stmt)).one()

    logs = (
        await session.exec(
            data_stmt.order_by(col(ClientLog.created_at).desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
        )
    ).all()

    log_responses = [ClientLogResponse(**log.model_dump()) for log in logs]

    return ClientLogListResponse(
        total=total,
        page=page,
        per_page=per_page,
        logs=log_responses,
    )


class ClientVersionStatsResponse(BaseModel):
    version: str
    count: int
    percentage: float
    last_seen: str


class ClientVersionStatsListResponse(BaseModel):
    total_users: int
    versions: list[ClientVersionStatsResponse]


@router.get(
    "/admin/logs/client-version-stats",
    name="获取客户端版本统计",
    tags=["管理", "g0v0 API"],
    response_model=ClientVersionStatsListResponse,
)
async def get_client_version_stats(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
    time_range: str = Query("7d"),
):
    """Get client version statistics (admin only)

    Aggregates data from both client_logs and score_tokens to show
    version statistics regardless of whether explicit logs were submitted.
    """
    await require_admin(session, user_and_token)

    # Calculate time range
    time_delta_map = {
        "24h": timedelta(hours=24),
        "7d": timedelta(days=7),
        "30d": timedelta(days=30),
        "all": None,
    }
    time_delta = time_delta_map.get(time_range, timedelta(days=7))

    # Build query - aggregate from score_tokens which always has client_version
    from app.database.score import ScoreToken
    query = select(
        ScoreToken.client_version,
        func.count(ScoreToken.id).label("count"),
        func.max(ScoreToken.created_at).label("last_seen"),
    ).where(
        col(ScoreToken.client_version).is_not(None),
        col(ScoreToken.client_version) != "",
    ).group_by(ScoreToken.client_version)

    if time_delta:
        cutoff_time = datetime.utcnow() - time_delta
        query = query.where(col(ScoreToken.created_at) >= cutoff_time)

    results = (await session.exec(query)).all()

    total_users = sum(r.count for r in results)

    version_stats = []
    for result in results:
        version_stats.append(
            ClientVersionStatsResponse(
                version=result.client_version,
                count=result.count,
                percentage=(result.count / total_users * 100) if total_users > 0 else 0,
                last_seen=result.last_seen.isoformat() if result.last_seen else None,
            )
        )

    # Sort by count descending
    version_stats.sort(key=lambda x: x.count, reverse=True)

    return ClientVersionStatsListResponse(
        total_users=total_users,
        versions=version_stats,
    )


class ClientPlatformStatsResponse(BaseModel):
    os_version: str
    count: int
    percentage: float


class ClientPlatformStatsListResponse(BaseModel):
    total_users: int
    platforms: list[ClientPlatformStatsResponse]


@router.get(
    "/admin/logs/client-platform-stats",
    name="获取客户端平台统计",
    tags=["管理", "g0v0 API"],
    response_model=ClientPlatformStatsListResponse,
)
async def get_client_platform_stats(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
    time_range: str = Query("7d"),
):
    """Get client platform statistics (admin only)"""
    await require_admin(session, user_and_token)

    # Calculate time range
    time_delta_map = {
        "24h": timedelta(hours=24),
        "7d": timedelta(days=7),
        "30d": timedelta(days=30),
        "all": None,
    }
    time_delta = time_delta_map.get(time_range, timedelta(days=7))

    # Build query
    query = select(
        ClientLog.os_version,
        func.count(ClientLog.id).label("count"),
    ).group_by(ClientLog.os_version)

    if time_delta:
        cutoff_time = datetime.utcnow() - time_delta
        query = query.where(col(ClientLog.created_at) >= cutoff_time)

    results = (await session.exec(query)).all()

    total_users = sum(r.count for r in results)

    platform_stats = []
    for result in results:
        if result.os_version:  # Filter out None values
            platform_stats.append(
                ClientPlatformStatsResponse(
                    os_version=result.os_version,
                    count=result.count,
                    percentage=(result.count / total_users * 100) if total_users > 0 else 0,
                )
            )

    # Sort by count descending
    platform_stats.sort(key=lambda x: x.count, reverse=True)

    return ClientPlatformStatsListResponse(
        total_users=total_users,
        platforms=platform_stats,
    )


# ========== Beatmap Search ==========

class BeatmapSearchResponse(BaseModel):
    id: int
    beatmapset_id: int
    title: str
    artist: str
    version: str
    difficulty_rating: float
    mode: str
    rank_status: str | None = None


class BeatmapSearchListResponse(BaseModel):
    total: int
    beatmaps: list[BeatmapSearchResponse]


@router.get(
    "/admin/beatmaps/search",
    name="搜索谱面",
    tags=["管理", "g0v0 API"],
    response_model=BeatmapSearchListResponse,
)
async def search_beatmaps(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
    q: str = Query("", min_length=1),
    limit: int = Query(50, ge=1, le=200),
):
    """Search beatmaps by query (admin only)"""
    await require_admin(session, user_and_token)

    if not q:
        return BeatmapSearchListResponse(total=0, beatmaps=[])

    search_term = f"%{q}%"

    # Search in beatmapsets and join with beatmaps
    beatmapset_ids = (
        await session.exec(
            select(Beatmapset.id).where(
                sql_or(
                    col(Beatmapset.title).ilike(search_term),
                    col(Beatmapset.artist).ilike(search_term),
                    col(Beatmapset.creator).ilike(search_term),
                )
            ).limit(limit)
        )
    ).all()

    if not beatmapset_ids:
        return BeatmapSearchListResponse(total=0, beatmaps=[])

    # Get beatmaps for these beatmapsets
    beatmaps = (
        await session.exec(
            select(Beatmap).where(col(Beatmap.beatmapset_id).in_(beatmapset_ids)).limit(limit)
        )
    ).all()

    # Get beatmapset info
    beatmapset_map: dict[int, dict] = {}
    for bs_id in beatmapset_ids:
        bs = await session.get(Beatmapset, bs_id)
        if bs:
            beatmapset_map[bs_id] = {
                "title": bs.title,
                "artist": bs.artist,
                "rank_status": bs.beatmap_status.value.lower() if bs.beatmap_status else None,
            }

    beatmap_responses = []
    for beatmap in beatmaps:
        bs_info = beatmapset_map.get(beatmap.beatmapset_id, {})
        beatmap_responses.append(
            BeatmapSearchResponse(
                id=beatmap.id,
                beatmapset_id=beatmap.beatmapset_id,
                title=bs_info.get("title", "Unknown"),
                artist=bs_info.get("artist", "Unknown"),
                version=beatmap.version,
                difficulty_rating=beatmap.difficulty_rating,
                mode=beatmap.mode.value.lower() if beatmap.mode else "osu",
                rank_status=bs_info.get("rank_status"),
            )
        )

    return BeatmapSearchListResponse(
        total=len(beatmap_responses),
        beatmaps=beatmap_responses,
    )


# ========== Beatmap Rank Requests ==========

class RankRequestListResponse(BaseModel):
    total: int
    page: int
    per_page: int
    requests: list[RankRequestResponse]


@router.get(
    "/admin/beatmap-rank-requests",
    name="获取谱面排名请求列表",
    tags=["管理", "g0v0 API"],
    response_model=RankRequestListResponse,
)
async def get_beatmap_rank_requests(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    status: RankRequestStatus | None = Query(None),
):
    """Get beatmap rank requests with pagination and filtering (admin only)"""
    await require_admin(session, user_and_token)

    conditions = []

    if status:
        conditions.append(col(RankRequest.status) == status)

    count_stmt = select(func.count()).select_from(RankRequest)
    data_stmt = select(RankRequest)

    if conditions:
        count_stmt = count_stmt.where(*conditions)
        data_stmt = data_stmt.where(*conditions)

    total = (await session.exec(count_stmt)).one()

    requests = (
        await session.exec(
            data_stmt.order_by(col(RankRequest.created_at).desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
        )
    ).all()

    # Get usernames and beatmapset info
    user_ids = {r.requester_id for r in requests if r.requester_id}
    reviewer_ids = {r.reviewed_by for r in requests if r.reviewed_by}
    all_user_ids = user_ids | reviewer_ids

    username_map: dict[int, str] = {}
    if all_user_ids:
        users = (
            await session.exec(
                select(User.id, User.username).where(col(User.id).in_(all_user_ids))
            )
        ).all()
        username_map = {uid: uname for uid, uname in users}

    beatmapset_ids = {r.beatmapset_id for r in requests}
    beatmapset_map: dict[int, dict] = {}
    if beatmapset_ids:
        for bs_id in beatmapset_ids:
            bs = await session.get(Beatmapset, bs_id)
            if bs:
                beatmapset_map[bs_id] = {
                    "title": bs.title,
                    "artist": bs.artist,
                }

    request_responses = []
    for request in requests:
        request_dict = request.model_dump()
        request_dict["requester_username"] = username_map.get(request.requester_id)
        request_dict["reviewed_by_username"] = username_map.get(request.reviewed_by)
        bs_info = beatmapset_map.get(request.beatmapset_id, {})
        request_dict["beatmapset_title"] = bs_info.get("title")
        request_dict["beatmapset_artist"] = bs_info.get("artist")
        request_responses.append(RankRequestResponse(**request_dict))

    return RankRequestListResponse(
        total=total,
        page=page,
        per_page=per_page,
        requests=request_responses,
    )


@router.post(
    "/admin/beatmap-rank-requests/{request_id}/approve",
    name="批准谱面排名请求",
    tags=["管理", "g0v0 API"],
)
async def approve_beatmap_rank_request(
    session: Database,
    request_id: int,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Approve a beatmap rank request (admin only)"""
    current_user = await require_admin(session, user_and_token)

    rank_request = await session.get(RankRequest, request_id)
    if not rank_request:
        raise HTTPException(status_code=404, detail="Rank request not found")

    if rank_request.status != RankRequestStatus.PENDING:
        raise HTTPException(status_code=400, detail="Request is not pending")

    rank_request.status = RankRequestStatus.APPROVED
    rank_request.reviewed_by = current_user.id
    rank_request.reviewed_at = datetime.utcnow()

    await session.commit()
    await session.refresh(rank_request)

    return {"message": "Rank request approved successfully", "request_id": request_id}


@router.post(
    "/admin/beatmap-rank-requests/{request_id}/reject",
    name="拒绝谱面排名请求",
    tags=["管理", "g0v0 API"],
)
async def reject_beatmap_rank_request(
    session: Database,
    request_id: int,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
    reason: str | None = None,
):
    """Reject a beatmap rank request (admin only)"""
    current_user = await require_admin(session, user_and_token)

    rank_request = await session.get(RankRequest, request_id)
    if not rank_request:
        raise HTTPException(status_code=404, detail="Rank request not found")

    if rank_request.status != RankRequestStatus.PENDING:
        raise HTTPException(status_code=400, detail="Request is not pending")

    rank_request.status = RankRequestStatus.REJECTED
    rank_request.reviewed_by = current_user.id
    rank_request.rejection_reason = reason
    rank_request.reviewed_at = datetime.utcnow()

    await session.commit()
    await session.refresh(rank_request)

    return {"message": "Rank request rejected successfully", "request_id": request_id}


# ========== Pending Counts ==========

class PendingCountsResponse(BaseModel):
    pending_reports: int
    pending_rank_requests: int
    pending_announcements: int  # Announcements that need attention
    total: int


@router.get(
    "/admin/pending-counts",
    name="获取待处理数量",
    tags=["管理", "g0v0 API"],
    response_model=PendingCountsResponse,
)
async def get_pending_counts(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
):
    """Get pending counts for menu badges (admin only)"""
    await require_admin(session, user_and_token)

    # Count pending reports
    pending_reports = (
        await session.exec(
            select(func.count()).select_from(Report).where(col(Report.status) == ReportStatus.PENDING)
        )
    ).one()

    # Count pending rank requests
    pending_rank_requests = (
        await session.exec(
            select(func.count()).select_from(RankRequest).where(col(RankRequest.status) == RankRequestStatus.PENDING)
        )
    ).one()

    # Count active announcements (those that need attention)
    pending_announcements = (
        await session.exec(
            select(func.count()).select_from(Announcement).where(
                col(Announcement.is_active) == True,
                sql_or(
                    col(Announcement.end_at).is_(None),
                    col(Announcement.end_at) > datetime.utcnow(),
                )
            )
        )
    ).one()

    total = pending_reports + pending_rank_requests + pending_announcements

    return PendingCountsResponse(
        pending_reports=pending_reports,
        pending_rank_requests=pending_rank_requests,
        pending_announcements=pending_announcements,
        total=total,
    )


# ========== Scores Endpoint ==========

@router.get(
    "/admin/scores",
    name="获取分数列表",
    tags=["管理", "g0v0 API"],
)
async def get_scores(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    user_id: int | None = Query(None),
    beatmap_id: int | None = Query(None),
):
    """Get scores with pagination and filtering (admin only)"""
    await require_admin(session, user_and_token)

    conditions = []

    if user_id:
        conditions.append(col(Score.user_id) == user_id)

    if beatmap_id:
        conditions.append(col(Score.beatmap_id) == beatmap_id)

    count_stmt = select(func.count()).select_from(Score)
    data_stmt = select(Score)

    if conditions:
        count_stmt = count_stmt.where(*conditions)
        data_stmt = data_stmt.where(*conditions)

    total = (await session.exec(count_stmt)).one()

    scores = (
        await session.exec(
            data_stmt.order_by(col(Score.id).desc())
            .offset((page - 1) * limit)
            .limit(limit)
        )
    ).all()

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "scores": [score.model_dump() for score in scores],
    }


# ========== Unknown Client Hashes Management ==========

class UnknownHashEntry(BaseModel):
    hash: str
    count: int
    first_seen: str | None = None
    last_seen: str | None = None
    last_user_id: int | None = None
    last_user_agent: str | None = None
    last_detected_os: str | None = None
    last_source: str | None = None


class UnknownHashListResponse(BaseModel):
    total: int
    hashes: list[UnknownHashEntry]


class AddHashOverrideRequest(BaseModel):
    client_hash: str
    client_name: str
    version: str = ""
    os_name: str = ""


class AddHashOverrideResponse(BaseModel):
    message: str
    hash: str


@router.get(
    "/admin/client-hashes/unknown",
    name="获取未知客户端哈希列表",
    tags=["管理", "g0v0 API"],
    response_model=UnknownHashListResponse,
)
async def get_unknown_client_hashes(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    sort_by: str = Query("count", regex="^(count|first_seen|last_seen|hash)$"),
    sort_desc: bool = Query(True),
):
    """Get list of unknown client hashes (admin only)"""
    await require_admin(session, user_and_token)

    from app.dependencies.client_verification import get_client_verification_service

    service = get_client_verification_service()
    unknown_hashes = await service.get_unknown_hashes()

    # Convert to list for sorting and pagination
    entries = []
    for hash_key, data in unknown_hashes.items():
        entries.append(
            UnknownHashEntry(
                hash=hash_key,
                count=data.get("count", 0),
                first_seen=data.get("first_seen"),
                last_seen=data.get("last_seen"),
                last_user_id=data.get("last_user_id"),
                last_user_agent=data.get("last_user_agent"),
                last_detected_os=data.get("last_detected_os"),
                last_source=data.get("last_source"),
            )
        )

    # Sort entries
    sort_key = sort_by
    reverse = sort_desc

    if sort_key == "count":
        entries.sort(key=lambda x: x.count, reverse=reverse)
    elif sort_key == "first_seen":
        entries.sort(key=lambda x: x.first_seen or "", reverse=reverse)
    elif sort_key == "last_seen":
        entries.sort(key=lambda x: x.last_seen or "", reverse=reverse)
    elif sort_key == "hash":
        entries.sort(key=lambda x: x.hash, reverse=reverse)

    total = len(entries)

    # Paginate
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_entries = entries[start_idx:end_idx]

    return UnknownHashListResponse(
        total=total,
        hashes=paginated_entries,
    )


@router.post(
    "/admin/client-hashes/override",
    name="添加哈希到允许列表",
    tags=["管理", "g0v0 API"],
    response_model=AddHashOverrideResponse,
)
async def add_hash_override(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
    request_data: AddHashOverrideRequest,
):
    """Add an unknown client hash to the allow list (admin only)"""
    await require_admin(session, user_and_token)

    from app.dependencies.client_verification import get_client_verification_service

    service = get_client_verification_service()

    await service.assign_hash_override(
        request_data.client_hash,
        client_name=request_data.client_name,
        version=request_data.version,
        os_name=request_data.os_name,
        remove_from_unknown=True,
    )

    logger.info(
        f"Admin {user_and_token[0].id} added hash override: "
        f"{request_data.client_hash[:16]}... as {request_data.client_name}"
    )

    return AddHashOverrideResponse(
        message="Hash successfully added to allow list",
        hash=request_data.client_hash,
    )


@router.delete(
    "/admin/client-hashes/unknown/{client_hash}",
    name="删除未知哈希记录",
    tags=["管理", "g0v0 API"],
)
async def delete_unknown_hash(
    session: Database,
    user_and_token: Annotated[UserAndToken, Security(get_client_user_and_token)],
    client_hash: str,
):
    """Remove an unknown client hash from the registry (admin only)"""
    await require_admin(session, user_and_token)

    from app.dependencies.client_verification import get_client_verification_service

    service = get_client_verification_service()

    async with service._lock:
        normalized_hash = client_hash.strip().lower()
        if normalized_hash in service.unknown_hashes:
            del service.unknown_hashes[normalized_hash]
            await service._persist_unknown_hashes()

    logger.info(
        f"Admin {user_and_token[0].id} deleted unknown hash: {client_hash[:16]}..."
    )

    return {"message": "Unknown hash deleted successfully", "hash": client_hash}


# Enhanced client log filtering
def build_enhanced_client_log_conditions(
    log_type: ClientLogType | None = None,
    user_id: int | None = None,
    client_version: str | None = None,
    client_hash: str | None = None,
    search: str | None = None,
) -> list:
    """Build SQL conditions for client log filtering"""
    conditions = []

    if log_type:
        conditions.append(col(ClientLog.log_type) == log_type)

    if user_id:
        conditions.append(col(ClientLog.user_id) == user_id)

    if client_version:
        conditions.append(col(ClientLog.client_version).ilike(f"%{client_version}%"))

    if client_hash:
        conditions.append(col(ClientLog.client_hash).ilike(f"%{client_hash}%"))

    if search:
        search_term = f"%{search}%"
        conditions.append(
            sql_or(
                col(ClientLog.username).ilike(search_term),
                col(ClientLog.client_version).ilike(search_term),
                col(ClientLog.client_hash).ilike(search_term),
                col(ClientLog.message).ilike(search_term),
            )
        )

    return conditions

