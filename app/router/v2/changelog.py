from __future__ import annotations

from datetime import UTC, datetime
from html import escape
from typing import Any

from app.dependencies.database import Database

from .router import router

from fastapi import Query
from sqlmodel import col, select


def _ts(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute, tzinfo=UTC)


_STREAM_ID = 1001
_STREAM_NAME = "lazer"
_STREAM_DISPLAY_NAME = "Torii"


_RAW_BUILDS: list[dict[str, Any]] = [
    {
        "id": 20260331,
        "version": "2026.331.0",
        "display_version": "2026.331.0-torii",
        "created_at": _ts(2026, 3, 31, 9, 30),
        "users": 0,
        "entries": [
            ("add", "torii", "Torii settings section now includes Appearance and Connection groups."),
            ("add", "torii", "Added native changelog toolbar button and startup release notes notification."),
            ("add", "pp", "Added pp-dev unlock alias code: luv-weird-pp."),
            ("fix", "ui", "Moved custom UI hue controls into Torii section for cleaner settings layout."),
            ("fix", "network", "Added runtime API endpoint apply and safer host validation in Torii Connection."),
        ],
    },
    {
        "id": 20260330,
        "version": "2026.330.1",
        "display_version": "2026.330.1-torii",
        "created_at": _ts(2026, 3, 30, 22, 10),
        "users": 0,
        "entries": [
            ("fix", "pp", "Stabilised pp-variant request flow for profile/top-play refresh."),
            ("fix", "toolbar", "Aligned toolbar indicator spacing and visibility transitions."),
            ("misc", "client", "Improved local diagnostics around API endpoint and online mode checks."),
        ],
    },
    {
        "id": 20260329,
        "version": "2026.329.0",
        "display_version": "2026.329.0-torii",
        "created_at": _ts(2026, 3, 29, 19, 0),
        "users": 0,
        "entries": [
            ("add", "pp", "Added pp-dev mode toggle plumbing for Torii local testing."),
            ("add", "ui", "Introduced Torii alpha feature gate via code input."),
            ("fix", "download", "Reduced beatmap download pre-redirect latency in backend mirror probing."),
        ],
    },
]


def _stream_stub() -> dict[str, Any]:
    return {
        "id": _STREAM_ID,
        "name": _STREAM_NAME,
        "display_name": _STREAM_DISPLAY_NAME,
        "is_featured": True,
        "user_count": 0,
    }


def _build_ref(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": raw["id"],
        "version": raw["version"],
        "display_version": raw["display_version"],
        "users": raw["users"],
        "created_at": raw["created_at"],
        "update_stream": _stream_stub(),
    }


def _entry_payload(raw_build: dict[str, Any], idx: int, entry: tuple[str, str, str]) -> dict[str, Any]:
    change_type, category, title = entry
    entry_id = int(f"{raw_build['id']}{idx + 1:02d}")
    return {
        "id": entry_id,
        "repository": "torii-osu",
        "github_pull_request_id": None,
        "github_url": "https://github.com/shikkesora",
        "url": f"https://lazer.shikkesora.com/changelog/{raw_build['version']}",
        "type": change_type,
        "category": category,
        "title": title,
        "message_html": f"<p>{escape(title)}</p>",
        "major": change_type == "add",
        "created_at": raw_build["created_at"],
        "github_user": {
            "id": 1,
            "display_name": "Shikkesora",
            "github_url": "https://github.com/shikkesora",
            "osu_username": "Shikkesora",
            "user_id": 19,
            "user_url": "https://lazer.shikkesora.com/users/19",
        },
    }


def _full_build_payload(raw_build: dict[str, Any], previous_raw: dict[str, Any] | None, next_raw: dict[str, Any] | None) -> dict[str, Any]:
    payload = _build_ref(raw_build)
    payload["changelog_entries"] = [
        _entry_payload(raw_build, idx, entry)
        for idx, entry in enumerate(raw_build["entries"])
    ]
    payload["versions"] = {
        "previous": _build_ref(previous_raw) if previous_raw else None,
        "next": _build_ref(next_raw) if next_raw else None,
    }
    return payload


@router.get("/changelog", tags=["Misc"], name="Changelog index")
async def changelog_index(
    session: Database,
    stream: str | None = Query(default=None),
    from_: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None),
):
    from app.database.changelog import ChangelogBuild, ChangelogEntry, ChangelogStream

    del from_, to

    db_streams = (await session.exec(select(ChangelogStream))).all()

    if db_streams:
        db_builds = (await session.exec(
            select(ChangelogBuild).order_by(col(ChangelogBuild.created_at).desc())
        )).all()

        if db_builds:
            stream_payloads = []
            for s in db_streams:
                if stream and stream != s.name:
                    continue
                stream_dict = {
                    "id": s.id,
                    "name": s.name,
                    "display_name": s.display_name,
                    "is_featured": s.is_featured,
                    "user_count": s.user_count,
                }

                latest_build_ref = None
                stream_builds = [b for b in db_builds if b.stream_id == s.id]
                if stream_builds:
                    first_build = stream_builds[0]
                    latest_build_ref = {
                        "id": first_build.id,
                        "version": first_build.version,
                        "display_version": first_build.display_version,
                        "users": first_build.users,
                        "created_at": first_build.created_at,
                        "update_stream": stream_dict,
                    }

                stream_payloads.append({
                    **stream_dict,
                    "latest_build": latest_build_ref,
                })

            full_builds = []
            for build in db_builds:
                stream_obj = next((s for s in db_streams if s.id == build.stream_id), None)
                if stream and stream_obj and stream != stream_obj.name:
                    continue

                entries = (await session.exec(
                    select(ChangelogEntry).where(ChangelogEntry.build_id == build.id)
                )).all()

                stream_builds = [b for b in db_builds if b.stream_id == build.stream_id]
                build_index = next((i for i, b in enumerate(stream_builds) if b.id == build.id), -1)
                previous_build = stream_builds[build_index + 1] if build_index + 1 < len(stream_builds) else None
                next_build = stream_builds[build_index - 1] if build_index - 1 >= 0 else None

                stream_dict = {
                    "id": stream_obj.id,
                    "name": stream_obj.name,
                    "display_name": stream_obj.display_name,
                    "is_featured": stream_obj.is_featured,
                    "user_count": stream_obj.user_count,
                } if stream_obj else None

                full_builds.append({
                    "id": build.id,
                    "version": build.version,
                    "display_version": build.display_version,
                    "users": build.users,
                    "created_at": build.created_at,
                    "github_url": build.github_url,
                    "update_stream": stream_dict,
                    "changelog_entries": [
                        {
                            "id": e.id,
                            "repository": e.repository,
                            "github_pull_request_id": e.github_pull_request_id,
                            "github_url": e.github_url,
                            "url": e.url,
                            "type": e.type.value if hasattr(e.type, "value") else str(e.type),
                            "category": e.category.value if hasattr(e.category, "value") else str(e.category),
                            "title": e.title,
                            "message_html": e.message_html,
                            "major": e.major,
                            "created_at": e.created_at,
                            "github_user": e.github_user or {},
                        }
                        for e in entries
                    ],
                    "versions": {
                        "previous": {
                            "id": previous_build.id,
                            "version": previous_build.version,
                            "display_version": previous_build.display_version,
                            "users": previous_build.users,
                            "created_at": previous_build.created_at,
                            "update_stream": stream_dict,
                        } if previous_build else None,
                        "next": {
                            "id": next_build.id,
                            "version": next_build.version,
                            "display_version": next_build.display_version,
                            "users": next_build.users,
                            "created_at": next_build.created_at,
                            "update_stream": stream_dict,
                        } if next_build else None,
                    },
                })

            return {
                "streams": stream_payloads,
                "builds": full_builds,
                "search": {"stream": stream or _STREAM_NAME, "from": None, "to": None, "limit": 21},
                "cursor_string": None,
            }

    if stream and stream != _STREAM_NAME:
        return {
            "streams": [],
            "builds": [],
            "search": {"stream": stream, "from": None, "to": None, "limit": 21},
            "cursor_string": None,
        }

    raw_builds = sorted(_RAW_BUILDS, key=lambda b: b["created_at"], reverse=True)
    stream_payload = _stream_stub()
    stream_payload["latest_build"] = _build_ref(raw_builds[0])

    full_builds: list[dict[str, Any]] = []
    for i, build in enumerate(raw_builds):
        previous_raw = raw_builds[i + 1] if i + 1 < len(raw_builds) else None
        next_raw = raw_builds[i - 1] if i - 1 >= 0 else None
        full_builds.append(_full_build_payload(build, previous_raw, next_raw))

    return {
        "streams": [stream_payload],
        "builds": full_builds,
        "search": {"stream": stream or _STREAM_NAME, "from": None, "to": None, "limit": 21},
        "cursor_string": None,
    }


@router.get("/changelog/{stream}/{version}", tags=["Misc"], name="Changelog build")
async def changelog_build(
    session: Database,
    stream: str,
    version: str,
):
    from app.database.changelog import ChangelogBuild, ChangelogEntry, ChangelogStream

    stream_obj = (await session.exec(
        select(ChangelogStream).where(ChangelogStream.name == stream)
    )).first()

    if stream_obj:
        build = (await session.exec(
            select(ChangelogBuild).where(
                ChangelogBuild.stream_id == stream_obj.id,
                ChangelogBuild.version == version,
            )
        )).first()

        if build:
            all_builds = (await session.exec(
                select(ChangelogBuild)
                .where(ChangelogBuild.stream_id == stream_obj.id)
                .order_by(col(ChangelogBuild.created_at).desc())
            )).all()

            entries = (await session.exec(
                select(ChangelogEntry).where(ChangelogEntry.build_id == build.id)
            )).all()

            build_index = next((i for i, b in enumerate(all_builds) if b.id == build.id), -1)
            previous_build = all_builds[build_index + 1] if build_index + 1 < len(all_builds) else None
            next_build = all_builds[build_index - 1] if build_index - 1 >= 0 else None

            stream_dict = {
                "id": stream_obj.id,
                "name": stream_obj.name,
                "display_name": stream_obj.display_name,
                "is_featured": stream_obj.is_featured,
                "user_count": stream_obj.user_count,
            }

            return {
                "id": build.id,
                "version": build.version,
                "display_version": build.display_version,
                "users": build.users,
                "created_at": build.created_at,
                "update_stream": stream_dict,
                "changelog_entries": [
                    {
                        "id": e.id,
                        "repository": e.repository,
                        "github_pull_request_id": e.github_pull_request_id,
                        "github_url": e.github_url,
                        "url": e.url,
                        "type": e.type.value if hasattr(e.type, "value") else str(e.type),
                        "category": e.category.value if hasattr(e.category, "value") else str(e.category),
                        "title": e.title,
                        "message_html": e.message_html,
                        "major": e.major,
                        "created_at": e.created_at,
                        "github_user": e.github_user or {},
                    }
                    for e in entries
                ],
                "versions": {
                    "previous": {
                        "id": previous_build.id,
                        "version": previous_build.version,
                        "display_version": previous_build.display_version,
                        "users": previous_build.users,
                        "created_at": previous_build.created_at,
                        "update_stream": stream_dict,
                    } if previous_build else None,
                    "next": {
                        "id": next_build.id,
                        "version": next_build.version,
                        "display_version": next_build.display_version,
                        "users": next_build.users,
                        "created_at": next_build.created_at,
                        "update_stream": stream_dict,
                    } if next_build else None,
                },
            }

    if stream != _STREAM_NAME:
        return {"detail": "build not found"}

    raw_builds = sorted(_RAW_BUILDS, key=lambda b: b["created_at"], reverse=True)

    for i, build in enumerate(raw_builds):
        if build["version"] != version:
            continue

        previous_raw = raw_builds[i + 1] if i + 1 < len(raw_builds) else None
        next_raw = raw_builds[i - 1] if i - 1 >= 0 else None
        return _full_build_payload(build, previous_raw, next_raw)

    return {"detail": "build not found"}
