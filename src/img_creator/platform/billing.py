"""Stripe grants credits only from signed, paid recurring invoices, once per invoice."""

import stripe
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from .db import BillingCheckout, BillingEvent, Ledger, Payment, User


def client(settings):
    if not settings.stripe_key:
        raise HTTPException(503, "Subscriptions are not configured yet")
    return stripe.StripeClient(settings.stripe_key)


def checkout(db, settings, user_id, plan):
    if plan not in settings.plans or not settings.plans[plan]["price_id"]:
        raise HTTPException(422, "This plan is not available yet")
    api = client(settings)
    # Lock prevents concurrent checkout requests from creating multiple customers.
    with db.transaction() as s:
        user = s.scalar(select(User).where(User.id == user_id).with_for_update())
        if user.subscription_status in {"active", "trialing", "past_due", "unpaid", "incomplete", "paused"}:
            raise HTTPException(409, "Manage your existing subscription in the billing portal")
        if not user.stripe_customer:
            customer = api.v1.customers.create(
                {"email": user.email, "metadata": {"user_id": user.id}},
                options={"idempotency_key": f"customer-{user.id}"},
            )
            user.stripe_customer = customer.id
        previous = s.get(BillingCheckout, user.id)
        if previous:
            existing = api.v1.checkout.sessions.retrieve(previous.session_id)
            if existing.status == "complete":
                raise HTTPException(409, "Payment is being reconciled. Refresh your account shortly.")
            if existing.status == "open":
                if previous.plan != plan:
                    raise HTTPException(409, "Finish or expire the existing checkout before changing plans")
                return {"url": existing.url}
        result = api.v1.checkout.sessions.create(
            {
                "mode": "subscription",
                "customer": user.stripe_customer,
                "line_items": [{"price": settings.plans[plan]["price_id"], "quantity": 1}],
                "subscription_data": {"metadata": {"user_id": user_id}},
                "success_url": settings.public_url + "/?checkout=success",
                "cancel_url": settings.public_url + "/?checkout=cancelled",
            },
            options={"idempotency_key": f"checkout-{user_id}-{previous.session_id if previous else 'initial'}-{plan}"},
        )
        if previous:
            previous.session_id, previous.plan = result.id, plan
        else:
            s.add(BillingCheckout(user_id=user.id, session_id=result.id, plan=plan))
        return {"url": result.url}


def process_event(db, settings, event, subscription_loader=None):
    kind, obj = event["type"], event["data"]["object"]
    try:
        with db.transaction() as s:
            if s.get(BillingEvent, event["id"]):
                return
            s.add(BillingEvent(id=event["id"], type=kind))
            s.flush()
            user = s.scalar(select(User).where(User.stripe_customer == obj.get("customer", "")).with_for_update())
            if not user:
                return
            if kind == "invoice.paid":
                # Do not grant a whole month's allowance for prorations or one-off invoices.
                if obj.get("billing_reason") not in {"subscription_create", "subscription_cycle"}:
                    return
                subscription_id = obj.get("subscription") or (
                    (obj.get("parent") or {}).get("subscription_details") or {}
                ).get("subscription")
                if not subscription_id or obj.get("status") != "paid":
                    return
                load = subscription_loader or (lambda value: client(settings).v1.subscriptions.retrieve(value))
                sub = load(subscription_id)
                if sub["customer"] != user.stripe_customer:
                    return
                items = sub["items"]["data"]
                if len(items) != 1 or items[0].get("quantity", 1) != 1:
                    raise ValueError("Unexpected subscription items; reconcile this invoice")
                # Use the invoice's purchased price, not a later subscription plan.
                lines = (obj.get("lines") or {}).get("data", [])
                if (obj.get("lines") or {}).get("has_more") or len(lines) != 1:
                    raise ValueError("Unexpected invoice lines; reconcile this invoice")
                line = lines[0]
                price_id = (line.get("price") or {}).get("id") or (
                    (line.get("pricing") or {}).get("price_details") or {}
                ).get("price")
                if line.get("quantity", 1) != 1:
                    raise ValueError("Unexpected invoice quantity")
                plan = next((k for k, v in settings.plans.items() if v["price_id"] == price_id), None)
                if not plan:
                    raise ValueError("Unknown subscription price; reconcile this invoice")
                reference = "invoice:" + obj["id"]
                if not s.scalar(select(Ledger).where(Ledger.reference == reference)):
                    credits = settings.plans[plan]["credits"]
                    s.add(Ledger(user_id=user.id, delta=credits, reason="subscription", reference=reference))
                    user.credits += credits
                    s.add(
                        Payment(
                            invoice=obj["id"],
                            user_id=user.id,
                            amount=obj.get("amount_paid", 0),
                            currency=obj.get("currency", "unknown"),
                        )
                    )
                # Reconcile from current Stripe state, rather than event delivery order.
                current_plan = next(
                    (k for k, v in settings.plans.items() if v["price_id"] == items[0]["price"]["id"]), "free"
                )
                user.plan = current_plan if sub["status"] == "active" else "free"
                user.subscription, user.subscription_status = subscription_id, sub["status"]
                user.billing_updated = max(user.billing_updated, event.get("created", 0))
            elif kind.startswith("customer.subscription."):
                if event.get("created", 0) < user.billing_updated:
                    return
                if user.subscription and user.subscription != obj["id"]:
                    # A second subscription must be reviewed; don't silently replace the first.
                    return
                user.billing_updated = event.get("created", 0)
                user.subscription = obj["id"]
                user.subscription_status = obj["status"]
                if obj["status"] in {"canceled", "unpaid", "incomplete_expired"}:
                    user.plan = "free"
                    pending = s.get(BillingCheckout, user.id)
                    if pending:
                        s.delete(pending)
    except IntegrityError:
        # Only an already committed event/invoice is safe to acknowledge as a duplicate.
        with db.transaction() as s:
            if not s.get(BillingEvent, event["id"]):
                raise
