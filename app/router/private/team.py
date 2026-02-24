import hashlib
from datetime import datetime
from typing import Annotated

from app.database.team import Team, TeamMember, TeamRequest, TeamResp
from app.database.user import User, UserModel
from app.dependencies.database import Database, Redis
from app.dependencies.storage import StorageService
from app.dependencies.user import ClientUser
from app.models.notification import (
    TeamApplicationAccept,
    TeamApplicationReject,
    TeamApplicationStore,
)
from app.models.score import GameMode
from app.router.notification import server
from app.service.ranking_cache_service import get_ranking_cache_service
from app.utils import api_doc, check_image, utcnow

from .router import router

from fastapi import File, Form, HTTPException, Path, Query, Request
from pydantic import BaseModel
from sqlmodel import col, exists, select


class TeamJoinRequestResp(BaseModel):
    user_id: int
    team_id: int
    requested_at: datetime
    user: UserModel


@router.post("/team", name="Create Team", response_model=Team, tags=["Team", "g0v0 API"])
async def create_team(
    session: Database,
    storage: StorageService,
    current_user: ClientUser,
    flag: Annotated[bytes, File(..., description="Team flag image")],
    cover: Annotated[bytes, File(..., description="Team cover image")],
    name: Annotated[str, Form(max_length=100, description="Team name")],
    short_name: Annotated[str, Form(max_length=10, description="Team short name")],
    redis: Redis,
    playmode: Annotated[GameMode, Form(description="Preferred team mode")] = GameMode.OSU,
    description: Annotated[str | None, Form(max_length=2000, description="Team description")] = None,
    website: Annotated[str | None, Form(max_length=255, description="Team website")]= None,
):
    """Create a team.

    flag limit 240x120, 2MB; cover limit 3000x2000, 10MB
    Supported formats: PNG, JPEG, GIF
    """
    if await current_user.is_restricted(session):
        raise HTTPException(status_code=403, detail="Your account is restricted and cannot perform this action.")

    if (await current_user.awaitable_attrs.team_membership) is not None:
        raise HTTPException(status_code=403, detail="You are already in a team")

    clean_name = name.strip()
    clean_short_name = short_name.strip()
    clean_description = description.strip() if description else None
    clean_website = website.strip() if website else None

    if not clean_name:
        raise HTTPException(status_code=400, detail="Team name cannot be empty")
    if not clean_short_name:
        raise HTTPException(status_code=400, detail="Team short name cannot be empty")

    if (await session.exec(select(exists()).where(Team.name == clean_name))).first():
        raise HTTPException(status_code=409, detail="Name already exists")
    if (await session.exec(select(exists()).where(Team.short_name == clean_short_name))).first():
        raise HTTPException(status_code=409, detail="Short name already exists")

    flag_format = check_image(flag, 2 * 1024 * 1024, 240, 120)
    cover_format = check_image(cover, 10 * 1024 * 1024, 3000, 2000)

    if clean_website and not (clean_website.startswith("http://") or clean_website.startswith("https://")):
        clean_website = "https://" + clean_website

    now = utcnow()
    team = Team(
        name=clean_name,
        short_name=clean_short_name,
        leader_id=current_user.id,
        created_at=now,
        playmode=playmode,
        description=clean_description,
        website=clean_website,
    )
    session.add(team)
    await session.commit()
    await session.refresh(team)

    filehash = hashlib.sha256(flag).hexdigest()
    storage_path = f"team_flag/{team.id}_{filehash}.png"
    if not await storage.is_exists(storage_path):
        await storage.write_file(storage_path, flag, f"image/{flag_format}")
    team.flag_url = await storage.get_file_url(storage_path)

    filehash = hashlib.sha256(cover).hexdigest()
    storage_path = f"team_cover/{team.id}_{filehash}.png"
    if not await storage.is_exists(storage_path):
        await storage.write_file(storage_path, cover, f"image/{cover_format}")
    team.cover_url = await storage.get_file_url(storage_path)

    session.add(TeamMember(user_id=current_user.id, team_id=team.id, joined_at=now))

    await session.commit()
    await session.refresh(team)

    cache_service = get_ranking_cache_service(redis)
    await cache_service.invalidate_team_cache()
    return team


@router.patch("/team/{team_id}", name="Update Team", response_model=Team, tags=["Team", "g0v0 API"])
async def update_team(
    team_id: int,
    session: Database,
    storage: StorageService,
    current_user: ClientUser,
    redis: Redis,
    flag: Annotated[bytes | None, File(description="Team flag image")] = None,
    cover: Annotated[bytes | None, File(description="Team cover image")] = None,
    name: Annotated[str | None, Form(max_length=100, description="Team name")] = None,
    short_name: Annotated[str | None, Form(max_length=10, description="Team short name")] = None,
    leader_id: Annotated[int | None, Form(description="New leader id")] = None,
    playmode: Annotated[GameMode | None, Form(description="Preferred team mode")] = None,
    description: Annotated[str | None, Form(max_length=2000, description="Team description")] = None,
    website: Annotated[str | None, Form(max_length=255, description="Team website")] = None,
):
    """Update a team.

    flag limit 240x120, 2MB; cover limit 3000x2000, 10MB
    Supported formats: PNG, JPEG, GIF
    """
    if await current_user.is_restricted(session):
        raise HTTPException(status_code=403, detail="Your account is restricted and cannot perform this action.")

    team = await session.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    if team.leader_id != current_user.id:
        raise HTTPException(status_code=403, detail="You are not the team leader")

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
        team.short_name = clean_short_name

    if playmode is not None:
        team.playmode = playmode

    if description is not None:
        clean_description = description.strip()
        team.description = clean_description or None

    if website is not None:
        clean_website = website.strip()
        if clean_website and not (clean_website.startswith("http://") or clean_website.startswith("https://")):
            clean_website = "https://" + clean_website
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

    if leader_id is not None:
        if not (await session.exec(select(exists()).where(User.id == leader_id))).first():
            raise HTTPException(status_code=404, detail="Leader not found")

        leader_membership = await session.exec(
            select(TeamMember).where(TeamMember.user_id == leader_id, TeamMember.team_id == team.id)
        )
        if not leader_membership.first():
            raise HTTPException(status_code=404, detail="Leader is not a member of the team")

        team.leader_id = leader_id

    await session.commit()
    await session.refresh(team)

    cache_service = get_ranking_cache_service(redis)
    await cache_service.invalidate_team_cache()

    return team


@router.delete("/team/{team_id}", name="Delete Team", status_code=204, tags=["Team", "g0v0 API"])
async def delete_team(
    session: Database,
    team_id: Annotated[int, Path(..., description="Team ID")],
    current_user: ClientUser,
    redis: Redis,
):
    if await current_user.is_restricted(session):
        raise HTTPException(status_code=403, detail="Your account is restricted and cannot perform this action.")

    team = await session.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    if team.leader_id != current_user.id:
        raise HTTPException(status_code=403, detail="You are not the team leader")

    team_members = await session.exec(select(TeamMember).where(TeamMember.team_id == team_id))
    for member in team_members:
        await session.delete(member)

    await session.delete(team)
    await session.commit()

    cache_service = get_ranking_cache_service(redis)
    await cache_service.invalidate_team_cache()


@router.get(
    "/team/{team_id}",
    name="Get Team",
    tags=["Team", "g0v0 API"],
    responses={
        200: api_doc(
            "Team info",
            {
                "team": TeamResp,
                "members": list[UserModel],
            },
            ["statistics", "country"],
            name="TeamQueryResp",
        )
    },
)
async def get_team(
    session: Database,
    team_id: Annotated[int, Path(..., description="Team ID")],
    gamemode: Annotated[GameMode | None, Query(description="Game mode")] = None,
):
    team = await session.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    members = (
        await session.exec(
            select(TeamMember).where(
                TeamMember.team_id == team_id,
                ~User.is_restricted_query(col(TeamMember.user_id)),
            )
        )
    ).all()

    return {
        "team": await TeamResp.from_db(team, session, gamemode),
        "members": await UserModel.transform_many([m.user for m in members], includes=["statistics", "country"]),
    }


@router.get(
    "/team/{team_id}/requests",
    name="Get Team Requests",
    tags=["Team", "g0v0 API"],
    response_model=list[TeamJoinRequestResp],
)
async def get_team_requests(
    session: Database,
    team_id: Annotated[int, Path(..., description="Team ID")],
    current_user: ClientUser,
):
    if await current_user.is_restricted(session):
        raise HTTPException(status_code=403, detail="Your account is restricted and cannot perform this action.")

    team = await session.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    if team.leader_id != current_user.id:
        raise HTTPException(status_code=403, detail="You are not the team leader")

    requests = (
        await session.exec(
            select(TeamRequest)
            .join(User, col(User.id) == col(TeamRequest.user_id))
            .where(
                TeamRequest.team_id == team_id,
                ~User.is_restricted_query(col(TeamRequest.user_id)),
            )
            .order_by(col(TeamRequest.requested_at).asc())
        )
    ).all()

    users = await UserModel.transform_many([request.user for request in requests], includes=["statistics", "country"])
    users_map = {user.id: user for user in users}

    return [
        TeamJoinRequestResp(
            user_id=request.user_id,
            team_id=request.team_id,
            requested_at=request.requested_at,
            user=users_map[request.user_id],
        )
        for request in requests
        if request.user_id in users_map
    ]


@router.get(
    "/team/{team_id}/request/status",
    name="Get Team Request Status",
    tags=["Team", "g0v0 API"],
)
async def get_team_request_status(
    session: Database,
    team_id: Annotated[int, Path(..., description="Team ID")],
    current_user: ClientUser,
):
    if await current_user.is_restricted(session):
        raise HTTPException(status_code=403, detail="Your account is restricted and cannot perform this action.")

    team = await session.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    if (await current_user.awaitable_attrs.team_membership) is not None:
        return {"has_pending_request": False}

    has_pending = (
        await session.exec(
            select(exists()).where(
                TeamRequest.team_id == team_id,
                TeamRequest.user_id == current_user.id,
            )
        )
    ).first()

    return {"has_pending_request": bool(has_pending)}


@router.post("/team/{team_id}/request", name="Request Team Join", status_code=204, tags=["Team", "g0v0 API"])
async def request_join_team(
    session: Database,
    team_id: Annotated[int, Path(..., description="Team ID")],
    current_user: ClientUser,
):
    if await current_user.is_restricted(session):
        raise HTTPException(status_code=403, detail="Your account is restricted and cannot perform this action.")

    team = await session.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    if (await current_user.awaitable_attrs.team_membership) is not None:
        raise HTTPException(status_code=403, detail="You are already in a team")

    if (
        await session.exec(
            select(exists()).where(TeamRequest.team_id == team_id, TeamRequest.user_id == current_user.id)
        )
    ).first():
        raise HTTPException(status_code=409, detail="Join request already exists")

    team_request = TeamRequest(user_id=current_user.id, team_id=team_id, requested_at=utcnow())
    session.add(team_request)
    await session.commit()
    await session.refresh(team_request)
    await server.new_private_notification(TeamApplicationStore.init(team_request))


@router.delete("/team/{team_id}/request", name="Cancel Team Join Request", status_code=204, tags=["Team", "g0v0 API"])
async def cancel_join_request(
    session: Database,
    team_id: Annotated[int, Path(..., description="Team ID")],
    current_user: ClientUser,
):
    if await current_user.is_restricted(session):
        raise HTTPException(status_code=403, detail="Your account is restricted and cannot perform this action.")

    team = await session.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    team_request = (
        await session.exec(
            select(TeamRequest).where(
                TeamRequest.team_id == team_id,
                TeamRequest.user_id == current_user.id,
            )
        )
    ).first()
    if not team_request:
        raise HTTPException(status_code=404, detail="Join request not found")

    await session.delete(team_request)
    await session.commit()


@router.post("/team/{team_id}/{user_id}/request", name="Accept Team Join Request", status_code=204, tags=["Team", "g0v0 API"])
@router.delete("/team/{team_id}/{user_id}/request", name="Reject Team Join Request", status_code=204, tags=["Team", "g0v0 API"])
async def handle_request(
    req: Request,
    session: Database,
    team_id: Annotated[int, Path(..., description="Team ID")],
    user_id: Annotated[int, Path(..., description="User ID")],
    current_user: ClientUser,
    redis: Redis,
):
    if await current_user.is_restricted(session):
        raise HTTPException(status_code=403, detail="Your account is restricted and cannot perform this action.")

    team = await session.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    if team.leader_id != current_user.id:
        raise HTTPException(status_code=403, detail="You are not the team leader")

    team_request = (
        await session.exec(select(TeamRequest).where(TeamRequest.team_id == team_id, TeamRequest.user_id == user_id))
    ).first()
    if not team_request:
        raise HTTPException(status_code=404, detail="Join request not found")

    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    accepted = req.method == "POST"

    if accepted:
        if (await session.exec(select(exists()).where(TeamMember.user_id == user_id))).first():
            raise HTTPException(status_code=409, detail="User is already a member of a team")

        session.add(TeamMember(user_id=user_id, team_id=team_id, joined_at=utcnow()))
        await server.new_private_notification(TeamApplicationAccept.init(team_request))
    else:
        await server.new_private_notification(TeamApplicationReject.init(team_request))

    await session.delete(team_request)
    await session.commit()

    if accepted:
        cache_service = get_ranking_cache_service(redis)
        await cache_service.invalidate_team_cache()


@router.delete("/team/{team_id}/{user_id}", name="Kick Member / Leave Team", status_code=204, tags=["Team", "g0v0 API"])
async def kick_member(
    session: Database,
    team_id: Annotated[int, Path(..., description="Team ID")],
    user_id: Annotated[int, Path(..., description="User ID")],
    current_user: ClientUser,
    redis: Redis,
):
    if await current_user.is_restricted(session):
        raise HTTPException(status_code=403, detail="Your account is restricted and cannot perform this action.")

    team = await session.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    if team.leader_id != current_user.id and user_id != current_user.id:
        raise HTTPException(status_code=403, detail="You are not allowed to remove this member")

    team_member = (
        await session.exec(select(TeamMember).where(TeamMember.team_id == team_id, TeamMember.user_id == user_id))
    ).first()
    if not team_member:
        raise HTTPException(status_code=404, detail="User is not a member of the team")

    if user_id == current_user.id and team.leader_id == current_user.id:
        raise HTTPException(status_code=403, detail="You cannot leave because you are the team leader")

    await session.delete(team_member)
    await session.commit()

    cache_service = get_ranking_cache_service(redis)
    await cache_service.invalidate_team_cache()
