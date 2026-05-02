from __future__ import annotations

from datetime import UTC, datetime
from html import escape
from typing import Annotated, Any

from app.database.changelog import (
    ChangelogBuild,
    ChangelogBuildCreate,
    ChangelogEntry,
    ChangelogEntryCreate,
    ChangelogStream,
    ChangelogStreamCreate,
    ChangeType,
    ChangelogCategory,
)
from app.database.user import User
from app.dependencies.database import Database
from app.dependencies.user import get_client_user
from app.log import log

from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel
from sqlmodel import col, select


class CreateEntryFromCommitRequest(BaseModel):
    build_id: int
    commit_sha: str
    commit_message: str
    repo: str = "shikkesora/torii-osu"

logger = log("ChangelogAPI")

router = APIRouter()


class ChangelogEntryResponse(BaseModel):
    id: int
    repository: str
    github_pull_request_id: int | None
    github_url: str
    url: str | None
    type: str
    category: str
    title: str
    message_html: str
    major: bool
    created_at: datetime
    github_user: dict[str, Any]


class ChangelogBuildResponse(BaseModel):
    id: int
    version: str
    display_version: str
    users: int
    created_at: datetime
    update_stream: dict[str, Any] | None
    changelog_entries: list[ChangelogEntryResponse]
    versions: dict[str, Any]


class ChangelogStreamResponse(BaseModel):
    id: int
    name: str
    display_name: str
    is_featured: bool
    user_count: int
    latest_build: ChangelogBuildResponse | None = None


class ChangelogListResponse(BaseModel):
    streams: list[ChangelogStreamResponse]
    builds: list[ChangelogBuildResponse]
    search: dict[str, Any]
    cursor_string: str | None


def _stream_to_dict(stream: ChangelogStream | None) -> dict[str, Any]:
    if not stream:
        return {"id": 0, "name": "unknown", "display_name": "Unknown", "is_featured": False, "user_count": 0}
    return {
        "id": stream.id,
        "name": stream.name,
        "display_name": stream.display_name,
        "is_featured": stream.is_featured,
        "user_count": stream.user_count,
    }


async def _build_ref(session: Database, build: ChangelogBuild) -> dict[str, Any]:
    stream = (await session.exec(select(ChangelogStream).where(ChangelogStream.id == build.stream_id))).first()
    return {
        "id": build.id,
        "version": build.version,
        "display_version": build.display_version,
        "users": build.users,
        "created_at": build.created_at,
        "update_stream": _stream_to_dict(stream),
    }


async def _get_stream_by_id(session: Database, stream_id: int) -> ChangelogStream | None:
    return (await session.exec(select(ChangelogStream).where(ChangelogStream.id == stream_id))).first()


async def _get_entries_for_build(session: Database, build_id: int) -> list[ChangelogEntryResponse]:
    entries = (await session.exec(select(ChangelogEntry).where(ChangelogEntry.build_id == build_id))).all()
    return [
        ChangelogEntryResponse(
            id=e.id,
            repository=e.repository,
            github_pull_request_id=e.github_pull_request_id,
            github_url=e.github_url,
            url=e.url,
            type=e.type.value if hasattr(e.type, "value") else str(e.type),
            category=e.category.value if hasattr(e.category, "value") else str(e.category),
            title=e.title,
            message_html=e.message_html,
            major=e.major,
            created_at=e.created_at,
            github_user=e.github_user or {
                "id": 1,
                "display_name": "Shikkesora",
                "github_url": "https://github.com/shikkesora",
                "osu_username": "Shikkesora",
                "user_id": 19,
                "user_url": "https://lazer.shikkesora.com/users/19",
            },
        )
        for e in entries
    ]


@router.get("/admin/builds", tags=["Changelog"])
async def admin_list_builds(
    session: Database,
    current_user: Annotated[User, Security(get_client_user)],
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")

    builds = (await session.exec(select(ChangelogBuild).order_by(col(ChangelogBuild.created_at).desc()))).all()

    result = []
    for build in builds:
        entries = (await session.exec(select(ChangelogEntry).where(ChangelogEntry.build_id == build.id))).all()
        stream = (await session.exec(select(ChangelogStream).where(ChangelogStream.id == build.stream_id))).first()
        result.append({
            "id": build.id,
            "version": build.version,
            "display_version": build.display_version,
            "stream_name": stream.name if stream else None,
            "stream_id": build.stream_id,
            "users": build.users,
            "created_at": build.created_at.isoformat(),
            "entry_count": len(entries),
        })

    return result


@router.get("/admin/entries/{build_id}", tags=["Changelog"])
async def admin_list_entries(
    session: Database,
    build_id: int,
    current_user: Annotated[User, Security(get_client_user)],
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")

    entries = (await session.exec(select(ChangelogEntry).where(ChangelogEntry.build_id == build_id))).all()

    return [
        {
            "id": e.id,
            "type": e.type.value if hasattr(e.type, 'value') else str(e.type),
            "category": e.category.value if hasattr(e.category, 'value') else str(e.category),
            "title": e.title,
            "major": e.major,
            "created_at": e.created_at.isoformat(),
        }
        for e in entries
    ]


@router.get("/", response_model=ChangelogListResponse, tags=["Changelog"])
async def changelog_index(
    session: Database,
    stream: str | None = None,
    from_: str | None = None,
    to: str | None = None,
):
    streams = (await session.exec(select(ChangelogStream))).all()

    if stream:
        streams = [s for s in streams if s.name == stream]
        if not streams:
            return ChangelogListResponse(
                streams=[],
                builds=[],
                search={"stream": stream, "from": None, "to": None, "limit": 21},
                cursor_string=None,
            )

    builds = (await session.exec(select(ChangelogBuild).order_by(col(ChangelogBuild.created_at).desc()))).all()

    stream_payloads = []
    for s in streams:
        build_refs = [b for b in builds if b.stream_id == s.id]
        latest = await _build_ref(session, build_refs[0]) if build_refs else None
        stream_payloads.append(
            ChangelogStreamResponse(
                id=s.id,
                name=s.name,
                display_name=s.display_name,
                is_featured=s.is_featured,
                user_count=s.user_count,
                latest_build=ChangelogBuildResponse(**latest) if latest else None,
            )
        )

    full_builds = []
    for i, build in enumerate(builds):
        previous_build = builds[i + 1] if i + 1 < len(builds) else None
        next_build = builds[i - 1] if i - 1 >= 0 else None

        full_builds.append(
            ChangelogBuildResponse(
                id=build.id,
                version=build.version,
                display_version=build.display_version,
                users=build.users,
                created_at=build.created_at,
                update_stream=_stream_to_dict(await _get_stream_by_id(session, build.stream_id)),
                changelog_entries=await _get_entries_for_build(session, build.id),
                versions={
                    "previous": await _build_ref(session, previous_build) if previous_build else None,
                    "next": await _build_ref(session, next_build) if next_build else None,
                },
            )
        )

    return ChangelogListResponse(
        streams=stream_payloads,
        builds=full_builds,
        search={"stream": stream or "lazer", "from": None, "to": None, "limit": 21},
        cursor_string=None,
    )


@router.get("/github/test", tags=["Changelog"])
async def test_github():
    return {"message": "GitHub endpoint works!", "repo": "test"}

def extract_repo_from_url(repo: str) -> str:
    """Extract owner/repo from various formats like 'owner/repo', 'https://github.com/owner/repo', etc."""
    if repo.startswith("http"):
        from urllib.parse import urlparse
        parsed = urlparse(repo)
        path = parsed.path.rstrip("/")
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    return repo.strip("/")


@router.get("/github/commits", tags=["Changelog"])
async def get_github_commits(
    repo: str = "shikkesora/torii-osu",
    per_page: int = 20,
):
    repo = extract_repo_from_url(repo)
    logger.info(f"GitHub commits called with repo={repo}, per_page={per_page}")
    try:
        import httpx
        url = f"https://api.github.com/repos/{repo}/commits"
        logger.info(f"Calling GitHub API: {url}")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                params={"per_page": per_page},
                headers={
                    "Accept": "application/vnd.github.v3+json",
                    "User-Agent": "Torii-Server"
                },
                timeout=15.0,
            )
            logger.info(f"GitHub API response: {response.status_code} for repo {repo}, body length: {len(response.text)}")
            
            if response.status_code == 200:
                commits = response.json()
                return [
                    {
                        "sha": c.get("sha", "")[:7],
                        "message": c.get("commit", {}).get("message", "").split("\n")[0],
                        "author": c.get("commit", {}).get("author", {}).get("name", ""),
                        "date": c.get("commit", {}).get("author", {}).get("date", ""),
                        "html_url": c.get("html_url", ""),
                        "full_sha": c.get("sha", ""),
                    }
                    for c in commits[:per_page]
                ]
            elif response.status_code == 403:
                return {"error": "GitHub API rate limited. Try again later or use a GitHub token."}
            elif response.status_code == 404:
                return {"error": f"Repository '{repo}' not found. Check the repo name format (owner/repo)."}
            else:
                return {"error": f"GitHub API returned {response.status_code}: {response.text[:200]}"}
    except Exception as e:
        logger.error(f"GitHub commits error: {e}")
        return {"error": str(e)}


@router.get("/{stream}/{version}", response_model=ChangelogBuildResponse, tags=["Changelog"])
async def changelog_build(
    session: Database,
    stream: str,
    version: str,
):
    stream_obj = (await session.exec(select(ChangelogStream).where(ChangelogStream.name == stream))).first()
    if not stream_obj:
        raise HTTPException(status_code=404, detail="stream not found")

    build = (await session.exec(
        select(ChangelogBuild).where(
            ChangelogBuild.stream_id == stream_obj.id,
            ChangelogBuild.version == version,
        )
    )).first()
    if not build:
        raise HTTPException(status_code=404, detail="build not found")

    all_builds = (await session.exec(
        select(ChangelogBuild)
        .where(ChangelogBuild.stream_id == stream_obj.id)
        .order_by(col(ChangelogBuild.created_at).desc())
    )).all()

    build_index = next((i for i, b in enumerate(all_builds) if b.id == build.id), -1)
    previous_build = all_builds[build_index + 1] if build_index + 1 < len(all_builds) else None
    next_build = all_builds[build_index - 1] if build_index - 1 >= 0 else None

    return ChangelogBuildResponse(
        id=build.id,
        version=build.version,
        display_version=build.display_version,
        users=build.users,
        created_at=build.created_at,
        update_stream=_stream_to_dict(stream_obj),
        changelog_entries=await _get_entries_for_build(session, build.id),
        versions={
            "previous": await _build_ref(session, previous_build) if previous_build else None,
            "next": await _build_ref(session, next_build) if next_build else None,
        },
    )


@router.get("/streams", response_model=list[ChangelogStreamResponse], tags=["Changelog"])
async def list_streams(session: Database):
    streams = (await session.exec(select(ChangelogStream))).all()
    return [
        ChangelogStreamResponse(
            id=s.id,
            name=s.name,
            display_name=s.display_name,
            is_featured=s.is_featured,
            user_count=s.user_count,
        )
        for s in streams
    ]


@router.post("/streams", response_model=ChangelogStreamResponse, tags=["Changelog"])
async def create_stream(
    session: Database,
    data: ChangelogStreamCreate,
    current_user: Annotated[User, Security(get_client_user)],
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")

    stream = ChangelogStream(
        name=data.name,
        display_name=data.display_name,
        is_featured=data.is_featured,
        user_count=data.user_count,
    )

    session.add(stream)
    await session.commit()
    await session.refresh(stream)

    return ChangelogStreamResponse(
        id=stream.id,
        name=stream.name,
        display_name=stream.display_name,
        is_featured=stream.is_featured,
        user_count=stream.user_count,
    )


@router.post("/builds", response_model=ChangelogBuildResponse, tags=["Changelog"])
async def create_build(
    session: Database,
    data: ChangelogBuildCreate,
    current_user: Annotated[User, Security(get_client_user)],
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")

    created_at = data.created_at or datetime.now(UTC)

    build = ChangelogBuild(
        stream_id=data.stream_id,
        version=data.version,
        display_version=data.display_version,
        users=data.users,
        created_at=created_at,
        github_url=data.github_url,
    )

    session.add(build)
    await session.flush()
    build_id = build.id
    await session.commit()

    return ChangelogBuildResponse(
        id=build_id,
        version=data.version,
        display_version=data.display_version,
        users=data.users,
        created_at=created_at,
        update_stream=_stream_to_dict(await _get_stream_by_id(session, data.stream_id)),
        changelog_entries=[],
        versions={"previous": None, "next": None},
    )


@router.post("/entries", response_model=ChangelogEntryResponse, tags=["Changelog"])
async def create_entry(
    session: Database,
    data: ChangelogEntryCreate,
    current_user: Annotated[User, Security(get_client_user)],
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")

    message_html = data.message_html or f"<p>{escape(data.title)}</p>"

    github_user = data.github_user or {
        "id": current_user.id,
        "display_name": current_user.username,
        "github_url": "https://github.com/shikkesora",
        "osu_username": current_user.username,
        "user_id": current_user.id,
        "user_url": f"https://lazer.shikkesora.com/users/{current_user.id}",
    }

    entry = ChangelogEntry(
        build_id=data.build_id,
        repository=data.repository,
        github_pull_request_id=data.github_pull_request_id,
        github_url=data.github_url,
        url=data.url,
        type=data.type.lower() if isinstance(data.type, str) else data.type,
        category=data.category.lower() if isinstance(data.category, str) else data.category,
        title=data.title,
        message_html=message_html,
        major=data.major,
        github_user=github_user,
    )

    session.add(entry)
    await session.flush()
    entry_id = entry.id
    await session.commit()

    return ChangelogEntryResponse(
        id=entry_id,
        repository=data.repository,
        github_pull_request_id=data.github_pull_request_id,
        github_url=data.github_url,
        url=data.url,
        type=data.type,
        category=data.category,
        title=data.title,
        message_html=message_html,
        major=data.major,
        created_at=datetime.now(UTC),
        github_user=github_user,
    )


@router.delete("/entries/{entry_id}", tags=["Changelog"])
async def delete_entry(
    session: Database,
    entry_id: int,
    current_user: Annotated[User, Security(get_client_user)],
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")

    entry = await session.get(ChangelogEntry, entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")

    await session.delete(entry)
    await session.commit()

    return {"message": "Entry deleted"}


@router.delete("/builds/{build_id}", tags=["Changelog"])
async def delete_build(
    session: Database,
    build_id: int,
    current_user: Annotated[User, Security(get_client_user)],
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")

    build = await session.get(ChangelogBuild, build_id)
    if not build:
        raise HTTPException(status_code=404, detail="Build not found")

    entries = (await session.exec(select(ChangelogEntry).where(ChangelogEntry.build_id == build_id))).all()
    for entry in entries:
        await session.delete(entry)

    await session.delete(build)
    await session.commit()

    return {"message": "Build deleted"}


@router.post("/entries/from-commit", response_model=ChangelogEntryResponse, tags=["Changelog"])
async def create_entry_from_github_commit(
    session: Database,
    current_user: Annotated[User, Security(get_client_user)],
    request: CreateEntryFromCommitRequest,
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")
    
    build_id = request.build_id
    commit_sha = request.commit_sha
    commit_message = request.commit_message
    repo = extract_repo_from_url(request.repo)

    message_html = f"<p>{escape(commit_message)}</p>"
    github_url = f"https://github.com/{repo}/commit/{commit_sha}"
    
    if "torii-lazer-web" in repo:
        category = "web"
    elif "g0v0-server" in repo:
        category = "server"
    else:
        category = "client"

    github_user = {
        "id": current_user.id,
        "display_name": current_user.username,
        "github_url": f"https://github.com/{current_user.username}",
        "osu_username": current_user.username,
        "user_id": current_user.id,
        "user_url": f"https://lazer.shikkesora.com/users/{current_user.id}",
    }

    entry = ChangelogEntry(
        build_id=build_id,
        repository=repo,
        github_pull_request_id=None,
        github_url=f"https://github.com/{repo}",
        url=github_url,
        type="misc",
        category=category,
        title=commit_message,
        message_html=message_html,
        major=False,
        github_user=github_user,
    )

    session.add(entry)
    await session.flush()
    entry_id = entry.id
    await session.commit()

    return ChangelogEntryResponse(
        id=entry_id,
        repository=repo,
        github_pull_request_id=None,
        github_url=f"https://github.com/{repo}",
        url=github_url,
        type="misc",
        category="client",
        title=commit_message,
        message_html=message_html,
        major=False,
        created_at=datetime.now(UTC),
        github_user=github_user,
    )
