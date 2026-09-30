"""Administrative CLI; never exposes bootstrap credentials through the public API."""

import argparse
import getpass
from sqlalchemy import select, delete
from .config import PlatformSettings
from .db import Database, User, Audit, Session
from .security import email_address, password_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init-db", "create-owner", "reset-password"])
    parser.add_argument("--email")
    args = parser.parse_args()
    db = Database(PlatformSettings().database_url)
    if args.command == "init-db":
        db.initialize()
        print("Initial schema created")
        return
    if not args.email:
        parser.error("--email is required")
    email = email_address(args.email)
    password = getpass.getpass("New password (12–128 characters): ")
    if password != getpass.getpass("Repeat password: "):
        raise SystemExit("Passwords do not match")
    with db.transaction() as s:
        user = s.scalar(select(User).where(User.email == email))
        if args.command == "create-owner":
            if user:
                raise SystemExit("Account exists. This command does not promote existing accounts.")
            user = User(email=email, password=password_hash(password), role="owner")
            s.add(user)
            s.flush()
        else:
            if not user:
                raise SystemExit("Account not found")
            user.password = password_hash(password)
            s.execute(delete(Session).where(Session.user_id == user.id))
        s.add(Audit(actor="operator-cli", action=args.command, target=user.id))
    print("Account updated")


if __name__ == "__main__":
    main()
