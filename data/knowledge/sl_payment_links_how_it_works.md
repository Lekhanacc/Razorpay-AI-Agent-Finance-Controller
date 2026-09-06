# How Payment Link Works

Source: https://razorpay.com/docs/payments/payment-links/how-it-works/

Understand the complete end-to-end flow of using Razorpay Payment Links.

## Workflow

### Step 1: Create a Payment Link
Create a Payment Link by providing required details. You can set an expiry date and enable partial payments. Links can later be updated or duplicated.

Related APIs: Create a Payment Link, Edit a Payment Link, Cancel a Payment Link.

### Step 2: Send a Payment Link
Send the link to a customer via email and/or SMS. The customer opens the link and pays using one of the available payment methods.

### Step 3: Receive Payments
The customer opens the link and attempts payment.
- If partial payment is enabled, the customer can choose the amount to pay.
- The customer chooses a payment method.
- On success, the Payment Link is marked `paid` or `partially_paid`, and a notification is sent.
- After capture, the amount is settled to the merchant's account per the settlement schedule.

### Step 4: Track Payment Links and Reports
- **Notifications**: activity notifications via email and webhook.
- **Track Payments**: track payments against issued links from the Dashboard under Payment Links.
- **Reports**: detailed insights via Dashboard reports, usable for accounting/reconciliation.

## Related Information (from source)

- About Payment Links
- FAQs
- Payment Links APIs
