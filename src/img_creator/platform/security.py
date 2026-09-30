import hashlib
import hmac
import re
import secrets
import time
from fastapi import HTTPException
from sqlalchemy import delete, update
from sqlalchemy.exc import IntegrityError
from .db import RateLimit


def password_hash(password):
    if not 12 <= len(password) <= 128:
        raise HTTPException(422, "Use a password between 12 and 128 characters")
    salt = secrets.token_hex(16)
    digest = hashlib.scrypt(password.encode(), salt=salt.encode(), n=16384, r=8, p=1).hex()
    return f"scrypt${salt}${digest}"


def password_matches(password, encoded):
    if not 12 <= len(password) <= 128:
        return False
    _, salt, digest = encoded.split("$")
    candidate = hashlib.scrypt(password.encode(), salt=salt.encode(), n=16384, r=8, p=1).hex()
    return hmac.compare_digest(digest, candidate)


def email_address(email):
    email = email.strip().lower()
    if len(email) > 254 or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        raise HTTPException(422, "Enter a valid email address")
    return email


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def rate_limit(db, scope, identity, limit=10, seconds=900):
    now = time.time()
    key = digest(f"{scope}:{identity}:{int(now // seconds)}")
    try:
        with db.transaction() as s:
            s.add(RateLimit(key=key, count=0, expires=now + seconds * 2))
    except IntegrityError:
        pass
    with db.transaction() as s:
        s.execute(delete(RateLimit).where(RateLimit.expires < now))
        accepted = s.execute(
            update(RateLimit).where(RateLimit.key == key, RateLimit.count < limit).values(count=RateLimit.count + 1)
        ).rowcount
    if not accepted:
        raise HTTPException(429, "Too many requests. Try again later.", headers={"Retry-After": str(seconds)})
