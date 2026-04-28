"""
Announcement notification service.

Sends announcements as GlobalAnnouncement notifications directly via WebSocket
using the server's new_private_notification method.
"""

from typing import Optional

from app.const import BANCHOBOT_ID
from app.database import Database
from app.database.announcement import Announcement, AnnouncementType
from app.database.user import User
from app.models.notification import GlobalAnnouncement
from app.router.notification.server import server
from sqlmodel import col, select


async def trigger_announcement_notification(
    session: Database,
    announcement: Announcement,
    current_user_id: int,
    online_only: bool = False,
) -> Optional[int]:
    """
    Send an announcement notification to users via WebSocket.

    Creates a GlobalAnnouncement notification and sends via server.
    The session is used for the receiver query.
    """
    if not announcement.is_active:
        return None

    # Map severity
    severity_map = {
        AnnouncementType.INFO: "info",
        AnnouncementType.WARNING: "warning",
        AnnouncementType.ERROR: "error",
        AnnouncementType.SUCCESS: "info",
        AnnouncementType.EVENT: "info",
        AnnouncementType.MAINTENANCE: "warning",
    }
    severity = severity_map.get(announcement.type, "info")

    # Get receivers using the passed session
    # Note: session may have been committed by caller, so we handle that
    try:
        if online_only:
            # Prefer websocket presence - get connected users
            connected_user_ids = [
                uid for uid, sockets in server.connect_client.items() if sockets
            ]
            if not connected_user_ids:
                receivers: list[int] = []
            else:
                receivers = (
                    await session.exec(
                        select(User.id).where(
                            col(User.id).in_(connected_user_ids),
                            User.id != BANCHOBOT_ID,
                            ~User.is_restricted_query(col(User.id)),
                        )
                    )
                ).all()
        else:
            # Send to all users except banned/restricted
            receivers = (
                await session.exec(
                    select(User.id).where(
                        User.id != BANCHOBOT_ID,
                        User.id != current_user_id,
                        ~User.is_restricted_query(col(User.id)),
                    )
                )
            ).all()
    except Exception as e:
        # If session is closed/expired, log and return None
        import logging
        logging.getLogger("announcement").error(f"Failed to query receivers: {e}")
        return None

    if not receivers:
        return None

    # Create and send the notification
    notification = GlobalAnnouncement.init(
        source_user_id=current_user_id,
        title=announcement.title,
        message=announcement.content,
        severity=severity,  # type: ignore
        receiver_ids=receivers,
    )

    # Send directly via server - this handles its own DB session internally
    await server.new_private_notification(notification)

    return notification.object_id
