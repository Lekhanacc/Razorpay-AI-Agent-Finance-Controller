# Payments Webhook Events

Source: https://razorpay.com/docs/webhooks/payments/

By subscribing to payments webhook events, you can get notified about payment state changes.

## Key concept: payload is a point-in-time snapshot

The payload for a webhook is a snapshot of the entity when the event occurred. For example, when a payment is authorised, its status becomes `authorized` and it can then immediately move to `captured`. The `payment.authorized` webhook payload reflects the state at authorisation time, not at capture time.

## payment.captured vs order.paid

Orders and payments go hand-in-hand. Once a payment is captured, the order is marked `paid`. Both the `order.paid` and `payment.captured` webhook events are triggered when the payment associated with the order is captured.

## Payment lifecycle events (from source)

- `payment.authorized` — fired when a payment is authorised. Sample payload includes `id` (e.g. `pay_DESp9bgForNoUd`), `order_id`, `amount`, `status: "authorized"`, `method`, card/bank/wallet details, `acquirer_data` (auth_code, rrn).
- `payment.captured` — fired when a payment is captured. Same entity shape as above with `status: "captured"`.
- `payment.failed` — fired when a payment fails. Same entity shape with `status: "failed"`; includes `error_code`, `error_description`, `error_reason`, `error_source`, `error_step` fields for diagnosing failure (populated when available).

## Payments Downtime events (from source)

Downtime is a period during which one or more payment options underperform, causing delays, due to technical issues/outages at Razorpay's partner or issuing banks. Razorpay notifies merchants of downtime so they can inform customers.

- `payment.downtime.started` — downtime begins. Payload includes `id` (e.g. `down_F1Zppa6lcVheSE`), `method`, `begin` (timestamp), `end` (null while ongoing), `status: "started"`, `severity` (e.g. `high`), `instrument` (e.g. issuer/network + type).
- `payment.downtime.resolved` — downtime ends. Includes populated `end` timestamp and `status: "resolved"`.
- `payment.downtime.updated` — downtime details updated (e.g. instrument info changes) while status remains ongoing.

## Important operational notes (from source)

- If the webhook secret has been rotated, use the **old** secret to validate signatures when retrying **older** requests; using the new secret causes a signature mismatch.
- When generating a signature locally, pass the webhook body as the **raw request body** — do not parse or cast it before validating.
