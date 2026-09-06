# Create a Subscription

Source: https://razorpay.com/docs/api/payments/subscriptions/create-subscription/

`POST /v1/subscriptions`

Use this endpoint to create a Subscription.

## Request Parameters (selected, from source)

| Parameter | Type | Notes |
|---|---|---|
| `plan_id` * | string | Unique identifier of the plan to link, e.g. `plan_00000000000001`. |
| `total_count` * | integer | Number of billing cycles the customer is charged for. Required unless `end_at` is used instead. |
| `quantity` | integer | Number of times the plan amount is charged per invoice. Default `1`. |
| `start_at` | integer | Unix timestamp for Subscription start. If omitted, starts immediately after the authorisation payment. |
| `expire_by` | integer | Unix timestamp until which the customer can complete the authorisation payment. Default: 30 years. |
| `customer_notify` | boolean | `true` (default): Razorpay handles customer communication. `false`: business handles it. |
| `addons` | object | Upfront amount(s) collected as part of the authorisation transaction. |
| `offer_id` | string | Identifier of an offer linked to the Subscription (from Dashboard). |
| `notes` | object | Up to 15 key-value pairs, for reference. |

## Response Parameters (selected, from source)

| Parameter | Type | Notes |
|---|---|---|
| `id` | string | e.g. `sub_00000000000001` |
| `status` | string | One of: `created`, `authenticated`, `active`, `pending`, `halted`, `cancelled`, `completed`, `expired` |
| `customer_id` | string | Auto-populated once the customer completes the authorisation transaction. |
| `current_start` / `current_end` | integer | Start/end of the current billing cycle (Unix timestamp). |
| `charge_at` | integer | When the next charge is scheduled. |
| `auth_attempts` | integer | Number of authorisation charge attempts made. |
| `paid_count` | integer | Number of billing cycles already charged. |
| `remaining_count` | integer | Billing cycles remaining. |
| `has_scheduled_changes` | boolean | Whether the subscription has pending scheduled changes. |
| `schedule_change_at` | string | `now` (default) or `cycle_end` — when an update takes effect. |
| `short_url` | string | URL for the customer to complete the authorisation payment. |

Maximum supported subscription duration: 100 years (per source).

## Errors (selected, from source)

- "The requested URL was not found on the server" (400) — Subscriptions feature not enabled on the account.
- "The id provided does not exist" (400) — invalid `plan_id`.
- "Offer Not Found" (400) — invalid/expired `offer_id`.
- "Offer not applicable for this Subscription" (400) — `offer_id` doesn't apply to the linked plan.
- "The plan id field is required." (400)
- "The plan id must be 19 characters." (400) — expected format `plan_<14 alphanumeric chars>`.
- "The total count field is required when end at is not present." (400) — need either `total_count` or `end_at`.
- "The total count must be at least 1." / "The total count must be an integer." (400)
- "The quantity must be at least 1." (400)
- "start_at cannot be lesser than the current time." (400)
- "The customer notify field must be true or false." (400)
- "{any extra field} is/are not required and should not be sent." (400)
- "Please provide either an end date or a total count, but not both." (400)

## Sample request

```
POST https://api.razorpay.com/v1/subscriptions
{
  "plan_id": "plan_00000000000001",
  "total_count": 6,
  "quantity": 1,
  "customer_notify": true,
  "start_at": 1773461489,
  "expire_by": 1773547889,
  "addons": [{"item": {"name": "Delivery charges", "amount": 3000, "currency": ""}}],
  "notes": {"notes_key_1": "Tea, Earl Grey, Hot"}
}
```
