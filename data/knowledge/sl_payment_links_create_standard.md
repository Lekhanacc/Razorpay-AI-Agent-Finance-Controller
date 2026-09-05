# Create a Standard Payment Link

Source: https://razorpay.com/docs/api/payments/payment-links/create-standard/

`POST /v1/payment_links`

Use this endpoint to create a Payment Link using basic details such as amount, expiry date, reference id, description, customer details, and so on.

- **Basic Payment Links**: regular Payment Links, not customised.
- **Customised Payment Links**: can be customised per business requirements (see "customise Payment Links" guide).

## Handy Tips (from source)

- Per Razorpay's updated security policy, even if the customer's email address and phone number are provided while creating the Payment Link, these details are not auto-populated on the Checkout section of the Payment Link hosted page — the customer must enter them manually.
- **Test Mode Limit**: up to 30 Payment Links per business in test mode. Contact Razorpay Support to create more for testing.
- After successful payment, customers can be redirected to a specific URL using `callback_url` and `callback_method`.
- The `razorpay_signature` parameter should be verified to confirm authenticity.

## Request Parameters (selected, from source)

| Parameter | Type | Notes |
|---|---|---|
| `amount` * | integer | Amount in the smallest currency unit. E.g., for $300, pass `30000`. |
| `currency` | string | Three-letter ISO currency code, e.g. USD. |
| `accept_partial` | boolean | Whether partial payments are allowed. Default `false`. |
| `first_min_partial_amount` | integer | Minimum first partial payment amount, in subunits. Default `100`. Must be passed with `accept_partial`. |
| `description` | string | Max 2048 characters. |
| `reference_id` | string | Unique per Payment Link. Max 40 characters. |
| `customer` | json object | Customer details (name, contact, email). |
| `expire_by` | integer | Unix timestamp. Default validity: six months from creation. Cannot exceed six months from creation. |
| `notify` | array | Who handles notification (sms/email booleans). |
| `notes` | json object | Up to 15 key-value pairs, 256 chars max each. |
| `callback_url` | string | Redirect URL after payment completion. |
| `callback_method` | string | Must be `get` if `callback_url` is passed. |
| `reminder_enable` | boolean | Enables automated reminders. |

## Response Parameters (selected, from source)

| Parameter | Type | Notes |
|---|---|---|
| `id` | string | e.g. `plink_ExjpAUN3gVHrPJ` |
| `status` | string | One of: `created`, `partially_paid`, `expired`, `cancelled`, `paid` |
| `short_url` | string | Shareable short URL for the Payment Link |
| `amount_paid` | integer | Amount paid so far |
| `payments` | array | Populated only after a payment is successfully captured; only captured payments appear here; `null` until then |
| `cancelled_at` / `expired_at` / `created_at` / `updated_at` | integer | Unix timestamps |
| `user_id` | string | Identifier for the user role that created the link |
| `whatsapp_link` | boolean | Whether the link is a WhatsApp link |
| `reminders` | object | Reminder dispatch status: `pending`, `in_progress`, or `failed` |
| `options` | array | Custom checkout options (theme, prefill) applied to the link |

## Errors (from source)

- **"The {input field} is required"** (4xx) — a mandatory field is empty.
- **"payment link with given reference_id: notes already exists..."** (400) — an existing `reference_id` was reused; must be unique.

## Sample request

```
POST https://api.razorpay.com/v1/payment_links/
{
  "amount": 1000,
  "currency": "",
  "accept_partial": true,
  "first_min_partial_amount": 100,
  "expire_by": 1691097057,
  "reference_id": "TS1989",
  "description": "Payment for policy no #23456",
  "customer": {"name": "John Smith", "contact": "+11234567890", "email": "john.smith@example.com"},
  "notify": {"sms": true, "email": true},
  "reminder_enable": true,
  "notes": {"policy_name": "Life Insurance Policy"},
  "callback_url": "https://example-callback-url.com/",
  "callback_method": "get"
}
```
