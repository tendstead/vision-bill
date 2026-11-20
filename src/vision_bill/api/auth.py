"""Authentication endpoints: register, login, logout, me.

Public by design — these are the only non-protected routes (plus
``/system/ui-config``). They manage the stateless, signed session cookie.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from ..config import settings
from ..provider.db.user_db import UserDB
from ..security.dependencies import get_current_user, validate_expected_actor
from ..security.models import User
from ..security.password import hash_password
from ..security.session import create_token, decode_token
from .helper.helper import get_user_db

logger = logging.getLogger(__name__)

router = APIRouter()


class Credentials(BaseModel):
    """Login / registration body."""

    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=200)


def _require_user_db(user_db: UserDB) -> None:
    if not user_db.is_ready:
        raise HTTPException(status_code=503, detail="Database is unavailable")


def _set_session_cookie(response: Response, user_id: UUID) -> None:
    """Set the signed session cookie (HttpOnly, SameSite=Lax)."""
    token = create_token(user_id, settings.auth.session_max_age_seconds, settings.auth.secret_key)
    response.set_cookie(
        settings.auth.session_cookie_name,
        token,
        max_age=settings.auth.session_max_age_seconds,
        httponly=True,
        samesite="lax",
        secure=settings.auth.session_secure,
    )


@router.post("/register", status_code=201, response_model=User)
async def register(
    body: Credentials,
    response: Response,
    user_db: UserDB = Depends(get_user_db),  # noqa: B008
) -> User:
    """Create an account and set the session cookie.

    ``409`` when the username is taken, ``403`` when registration is disabled.
    New accounts are always non-admin (admins are bootstrapped from env).
    """
    _require_user_db(user_db)
    if not settings.auth.allow_registration:
        raise HTTPException(status_code=403, detail="Registration is disabled")
    if await user_db.get_user_by_username(body.username) is not None:
        raise HTTPException(status_code=409, detail="Username is already taken")
    hashed = hash_password(body.password, settings.auth)
    user = await user_db.create_user(body.username, hashed, is_admin=False)
    logger.info("Registered new user '%s' (id=%s)", user.username, user.id)
    _set_session_cookie(response, user.id)
    return user


@router.post("/login", response_model=User)
async def login(
    body: Credentials,
    response: Response,
    user_db: UserDB = Depends(get_user_db),  # noqa: B008
) -> User:
    """Authenticate with username/password and set the session cookie.

    Returns ``401`` for unknown users or a wrong password (same response, to
    avoid user enumeration).
    """
    _require_user_db(user_db)
    user = await user_db.verify_credentials(body.username, body.password, settings.auth)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    _set_session_cookie(response, user.id)
    return user


@router.post("/logout")
async def logout(request: Request, response: Response) -> dict[str, bool]:
    """Clear the cookie unless the browser expects another signed-in actor."""
    token = request.cookies.get(settings.auth.session_cookie_name)
    actor = decode_token(token, settings.auth.secret_key) if token else None
    validate_expected_actor(request, actor)
    response.delete_cookie(settings.auth.session_cookie_name)
    return {"ok": True}


@router.get("/me", response_model=User)
async def me(user: User = Depends(get_current_user)) -> User:  # noqa: B008
    """Return the currently authenticated user, or ``401``."""
    return user
