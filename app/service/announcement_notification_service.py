from datetime import datetime
from typing import Optional

from app.const import BANCHOBOT_ID
from app.dependencies.database import Database
from app.database.announcement import Announcement, AnnouncementType
from app.database.user import User
from app.models.notification import GlobalAnnouncement
from app.router.notification.server import server
from app.utils import utcnow
from sqlmodel import col, select


async def trigger_announcement_notification(
    session: Database,
    announcement: Announcement,
    current_user_id: int,
    online_only: bool = False,
) -> Optional[int]:
    """
    Convert an Announcement to a GlobalAnnouncement notification and send it to users.

    Args:
        session: Database session
        announcement: The announcement to convert and send
        current_user_id: ID of the admin user triggering the notification
        online_only: Whether to send only to online users

    Returns:
        Notification ID if successful, None if failed
    """
    if not announcement.is_active:
        return None

    # Map announcement type to severity
    severity_map = {
        AnnouncementType.INFO: "info",
        AnnouncementType.WARNING: "warning",
        AnnouncementType.ERROR: "error",
        AnnouncementType.SUCCESS: "info",
        AnnouncementType.EVENT: "info",
        AnnouncementType.MAINTENANCE: "warning",
    }

    severity = severity_map.get(announcement.type, "info")

    # Get target user IDs
    if online_only:
        connected_user_ids = [uid for uid, sockets in server.connect_client.items() if sockets]
        if not connected_user_ids:
            receivers: list[int] = []
        else:
            receivers = (
                await session.exec(
                    select(User.id).where(
                        col(User.id).in_(connected_user_ids),
                        User.id != BANCHOBOT_ID,
                        User.id != current_user_id,
                        ~User.is_restricted_query(col(User.id)),
                    )
                )
            ).all()
    else:
        # Target users based on target_roles if specified
        if announcement.target_roles and announcement.target_roles != ["all"]:
            # For now, we'll send to all users except banned/restricted
            # In the future, this could filter by actual user roles
            receivers = (
                await session.exec(
                    select(User.id).where(
                        User.id != BANCHOBOT_ID,
                        User.id != current_user_id,
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

    if not receivers:
        return None

    # Create and send the GlobalAnnouncement notification
    notification = GlobalAnnouncement.init(
        source_user_id=current_user_id,
        title=announcement.title,
        message=announcement.content,
        severity=severity,  # type: ignore
        receiver_ids=receivers,
    )

    await server.new_private_notification(notification)

    return notification.object_id
