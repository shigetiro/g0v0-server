from typing import Annotated

from sqlalchemy import func

from app.database import User
from app.dependencies.database import Database
from app.dependencies.user import UserAndToken, get_current_user_and_token
from app.router.private.admin import require_admin
from app.router.private.router import router
from plugins.anticheat.models import RiskLevel, ScoreDetection

from fastapi import Depends, HTTPException, Query
from sqlmodel import select


@router.get("/anticheat/detections")
async def get_detections(
    session: Database,
    user_and_token: UserAndToken = Depends(get_current_user_and_token),
    user_id: int | None = Query(None, description="Filter by user ID"),
    score_id: int | None = Query(None, description="Filter by score ID"),
    risk_level: RiskLevel | None = Query(None, description="Filter by risk level"),
    flagged_only: bool = Query(False, description="Only return flagged detections"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    await require_admin(session, user_and_token)

    stmt = select(ScoreDetection).order_by(ScoreDetection.created_at.desc())

    if user_id is not None:
        stmt = stmt.where(ScoreDetection.user_id == user_id)
    if score_id is not None:
        stmt = stmt.where(ScoreDetection.score_id == score_id)
    if risk_level is not None:
        stmt = stmt.where(ScoreDetection.risk_level == risk_level)
    if flagged_only:
        stmt = stmt.where(ScoreDetection.flagged.is_(True))

    stmt = stmt.offset(offset).limit(limit)
    result = await session.exec(stmt)
    detections = result.all()

    return {
        "data": [
            {
                "id": d.id,
                "score_id": d.score_id,
                "user_id": d.user_id,
                "beatmap_id": d.beatmap_id,
                "risk_level": d.risk_level.value if hasattr(d.risk_level, "value") else d.risk_level,
                "total_risk_score": d.total_risk_score,
                "flagged": d.flagged,
                "details": d.detection_details,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in detections
        ],
        "total": len(detections),
        "limit": limit,
        "offset": offset,
    }


@router.get("/anticheat/detections/{detection_id}")
async def get_detection(
    detection_id: int,
    session: Database,
    user_and_token: UserAndToken = Depends(get_current_user_and_token),
):
    await require_admin(session, user_and_token)

    detection = await session.get(ScoreDetection, detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    return {
        "id": detection.id,
        "score_id": detection.score_id,
        "user_id": detection.user_id,
        "beatmap_id": detection.beatmap_id,
        "risk_level": detection.risk_level.value if hasattr(detection.risk_level, "value") else detection.risk_level,
        "total_risk_score": detection.total_risk_score,
        "flagged": detection.flagged,
        "details": detection.detection_details,
        "analysis_version": detection.analysis_version,
        "created_at": detection.created_at.isoformat() if detection.created_at else None,
    }


@router.post("/anticheat/detections/{detection_id}/flag")
async def flag_detection(
    detection_id: int,
    session: Database,
    user_and_token: UserAndToken = Depends(get_current_user_and_token),
):
    await require_admin(session, user_and_token)

    detection = await session.get(ScoreDetection, detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    detection.flagged = True
    session.add(detection)
    await session.commit()

    return {"success": True, "flagged": True}


@router.post("/anticheat/detections/{detection_id}/dismiss")
async def dismiss_detection(
    detection_id: int,
    session: Database,
    user_and_token: UserAndToken = Depends(get_current_user_and_token),
):
    await require_admin(session, user_and_token)

    detection = await session.get(ScoreDetection, detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    detection.flagged = False
    detection.risk_level = RiskLevel.NONE
    session.add(detection)
    await session.commit()

    return {"success": True, "flagged": False}


@router.get("/anticheat/stats")
async def get_anticheat_stats(
    session: Database,
    user_and_token: UserAndToken = Depends(get_current_user_and_token),
):
    await require_admin(session, user_and_token)

    total = (await session.exec(select(func.count(ScoreDetection.id)))).one()
    flagged = (await session.exec(select(func.count(ScoreDetection.id)).where(ScoreDetection.flagged.is_(True)))).one()
    critical = (
        await session.exec(
            select(func.count(ScoreDetection.id)).where(ScoreDetection.risk_level == RiskLevel.CRITICAL)
        )
    ).one()
    high = (
        await session.exec(
            select(func.count(ScoreDetection.id)).where(ScoreDetection.risk_level == RiskLevel.HIGH)
        )
    ).one()

    return {
        "total_detections": total,
        "flagged": flagged,
        "critical": critical,
        "high": high,
    }

@router.post("/anticheat/test-replay")
async def test_replay(
    payload: dict,
    session: Database,
    user_and_token: UserAndToken = Depends(get_current_user_and_token),
):
    from app.plugins import plugin_manager

    anticheat = plugin_manager.loaded_plugins.get("anticheat")
    if not anticheat:
        raise HTTPException(status_code=503, detail="Anticheat not loaded")

    frames = payload.get("frames", [])
    score_data = {
        "replay_data": {"frames": frames},
        "accuracy": 0.95,
        "pp": 200,
        "statistics": {"great": 100, "ok": 5, "miss": 2},
        "n300": 100,
        "n100": 5,
        "n50": 2,
        "nmiss": 2,
    }

    replay_data = anticheat._extract_replay_data(score_data)
    user_history = {"best_pp": 300, "avg_pp": 200, "play_count": 50}

    results = []
    for detector in ALL_DETECTORS:
        results.extend(detector.analyze(score_data, replay_data, user_history))

    total = min(sum(r.risk_score for r in results), 100)
    level = anticheat._calculate_risk_level(total)

    return {
        "risk_score": total,
        "risk_level": level.value if hasattr(level, "value") else level,
        "flagged": total >= config.auto_flag_threshold,
        "detectors_triggered": [r.plugin_id for r in results if r.is_suspicious],
    }
