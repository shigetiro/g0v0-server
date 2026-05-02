from app.dependencies.rate_limit import LIMITERS

from fastapi import APIRouter

router = APIRouter(prefix="/api/private", dependencies=LIMITERS)

# Import subrouters
from .admin import router as admin_router
from .audio_proxy import router as audio_proxy_router

# Include admin router with explicit prefix
router.include_router(admin_router, prefix="")
router.include_router(audio_proxy_router)

# Direct import changelog routes
import logging
logger = logging.getLogger("changelog_import")
logger.info("Starting changelog router import")

try:
    from .changelog import router as changelog_router
    router.include_router(changelog_router, prefix="/changelog")
    logger.info(f"Changelog routes registered: {len(changelog_router.routes)} routes")
except Exception as e:
    logger.error(f"Failed to import changelog router: {e}")
