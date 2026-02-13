from sqlalchemy import and_, func, literal_column, or_, select
from app.models.user_account_history import UserAccountHistory, UserAccountHistoryType


def active_restriction_exists(user_id_col):
    """
    Returns a SQL EXISTS() expression that is TRUE if the user currently has an active restriction.
    MySQL: TIMESTAMPADD(SECOND, length, timestamp) > NOW()
    """
    # MySQL expects SECOND as an identifier, not a string.
    second_unit = literal_column("SECOND")

    restriction_select = select(1).where(
        and_(
            UserAccountHistory.user_id == user_id_col,
            UserAccountHistory.type == UserAccountHistoryType.RESTRICTION,
            UserAccountHistory.timestamp <= func.now(),
            or_(
                UserAccountHistory.permanent.is_(True),
                func.timestampadd(second_unit, UserAccountHistory.length, UserAccountHistory.timestamp) > func.now(),
            ),
        )
    )

    return restriction_select.exists()
