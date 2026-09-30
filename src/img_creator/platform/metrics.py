"""Database-side usage totals; credits, provider tokens and time remain separate units."""

from sqlalchemy import Float, Integer, cast, func, select
from .db import Job


def usage_rows(session, group, user_ids=None):
    seconds = cast(Job.result["seconds"].as_string(), Float)
    tokens = cast(Job.result["tokens"].as_string(), Integer)
    query = (
        select(
            group.label("key"),
            func.count().label("images"),
            func.sum(Job.credits).label("credits"),
            func.coalesce(func.sum(seconds), 0).label("seconds"),
            func.sum(tokens).label("tokens"),
            func.count(tokens).label("token_reports"),
        )
        .where(Job.status == "succeeded")
        .group_by(group)
    )
    if user_ids is not None:
        query = query.where(Job.user_id.in_(user_ids))
    return {
        row.key: {
            "images": row.images,
            "credits": row.credits,
            "seconds": round(row.seconds, 3),
            "provider_tokens": row.tokens,
            "token_reporting_images": row.token_reports,
        }
        for row in session.execute(query)
    }


def empty_usage():
    return {"images": 0, "credits": 0, "seconds": 0, "provider_tokens": None, "token_reporting_images": 0}
