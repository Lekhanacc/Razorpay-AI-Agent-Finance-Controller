# ⚠ PROVENANCE NOTE — READ BEFORE USING THIS FILE

Requested URL: https://razorpay.com/docs/api/payments/subscriptions/
Expected content: Subscriptions API overview/index page.
Actual content returned: **Plans Entity** reference page (regional variant, served from
`https://razorpay.com/docs/us/api/payments/subscriptions/plans-entity.md`).

This is a real discrepancy from Razorpay's documentation server (likely due to docs-site
restructuring or redirect behavior), not a fabrication or substitution on our part. The
content below is exactly what was returned for the requested URL — it is stored under its
真实 (actual) topic, "Plans Entity," rather than mislabeled as a Subscriptions overview.

If a true Subscriptions-overview page is needed for the knowledge base, it must be
re-fetched from a corrected URL in a future ingestion pass — do not treat this file as
that content.

---

# Plans Entity

Source (actual, as returned): https://razorpay.com/docs/us/api/payments/subscriptions/plans-entity.md

The Plans entity has the following parameters.

## Sample response

```json
{
  "entity": "collection",
  "count": 2,
  "items": [
    {
      "id": "plan_00000000000008",
      "entity": "plan",
      "interval": 1,
      "period": "weekly",
      "item": {
        "id": "item_00000000000005",
        "active": true,
        "name": "Test plan - Monthly",
        "description": "Description for the test plan - Monthly",
        "amount": 89900,
        "unit_amount": 89900,
        "currency": "USD",
        "type": "plan"
      },
      "notes": {"notes_key_1": "Tea, Earl Grey, Hot"},
      "created_at": 1580220481
    }
  ]
}
```

## Fields (from source)

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique identifier for a plan; used when creating a Subscription. |
| `entity` | string | Always `plan`. |
| `interval` | integer | Combined with `period`, defines billing frequency. E.g., a 2-month cycle → `interval=2`. For daily plans, minimum value should be `7`. |
| `period` | string | One of `daily`, `weekly`, `monthly`, `quarterly`, `yearly`. Combined with `interval` to define frequency. |
| `item` | object | Plan details (id, name, amount, currency, description). |
| `item.amount` | integer | Amount charged to the customer each billing cycle when this plan is used in a Subscription. |
| `item.currency` | string | Defaults to `USD` (per this regional doc variant). |
| `notes` | object | Up to 15 key-value pairs for reference, e.g. `"note_key": "Monthly Gym"`. |
| `created_at` | integer | Unix timestamp of plan creation. |
