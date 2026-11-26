"""FastAPI dependencies that resolve the authenticated user.

``get_current_user`` turns the signed session cookie into a :class:`User`;
``require_admin`` gates admin-only routes.
"""

from uuid import UUID

from fastapi import Depends, HTTPException, Request

from ..api.helper.helper import get_user_db
from ..config import settings
from .models import User
from .session import decode_token

_UNAUTHENTICATED = HTTPException(status_code=401, detail="Not authenticated")


def _expected_actor(request: Request) -> UUID | None:
    header = request.headers.getlist("X-Vision-Bill-Expected-User")
    query = request.query_params.getlist("vb_expected_user")
    try:
        if len(header) > 1 or len(query) > 1:
            raise ValueError
        actors = {UUID(value) for value in [*header, *query]}
        if len(actors) > 1:
            raise ValueError
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid_expected_user") from None
    return next(iter(actors), None)


def validate_expected_actor(request: Request, user_id: UUID | None) -> None:
    """Check browser actor intent against any authenticated cookie actor."""
    expected = _expected_actor(request)
    if expected is not None and user_id is not None and expected != user_id:
        raise HTTPException(status_code=409, detail="session_actor_changed")


async def get_current_user(request: Request) -> User:
    """Resolve the authenticated user from the session cookie.

    Reads the signed session cookie, decodes it, loads the user, and derives
    the effective ``can_see_all`` privilege. Raises ``401`` on any failure
    (missing/invalid/expired cookie or unknown user) and ``503`` when the user
    store is unavailable.

    Browser data requests declare their originating actor in
    X-Vision-Bill-Expected-User or vb_expected_user (private media URLs).
    A malformed or contradictory expectation returns 400, and a cookie actor
    mismatch returns 409 before the protected operation. Deliberate API
    callers may omit both to operate as the current cookie user. The cookie
    and normal authorization rules always determine access.
    """
    user_db = get_user_db(request)
    if not user_db.is_ready:
        raise HTTPException(status_code=503, detail="Authentication is unavailable")

    token = request.cookies.get(settings.auth.session_cookie_name)
    if not token:
        raise _UNAUTHENTICATED

    user_id = decode_token(token, settings.auth.secret_key)
    if user_id is None:
        raise _UNAUTHENTICATED

    user = await user_db.get_user_by_id(user_id)
    if user is None:
        raise _UNAUTHENTICATED

    validate_expected_actor(request, user.id)

    user.can_see_all = user.is_admin and settings.auth.admin_can_see_all
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:  # noqa: B008
    """Allow only administrators; otherwise raise 403."""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Administrator privileges required")
    return user
