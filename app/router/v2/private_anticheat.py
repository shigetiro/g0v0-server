from app.dependencies.database import Database
from app.dependencies.user import ClientUser
from plugins.anticheat import ALL_DETECTORS
from plugins.anticheat.config import config
from app.router.v2.router import router

from fastapi import HTTPException


@router.post("/private/anticheat/test-replay")
async def test_replay(
    payload: dict,
    session: Database,
    current_user: ClientUser,
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
