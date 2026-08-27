# Escalation Thresholds

## Standard threshold
Any proposed action — make-good invoice, credit memo, or plan amendment — affecting **$15,000 or more** requires the customer's account manager to be notified before the action is applied, in addition to the standard human approval already required by the `apply()` step.

## Account-specific overrides
Some accounts have a lower threshold by agreement with their procurement team, regardless of the standard dollar amount. Check `internal_notes/customer_accounts.md` for the specific customer before applying a correction — an account-specific note always overrides the standard threshold.

## What "notify" means here
This sandbox doesn't have a notification system — treat this policy as something to surface to the human in your explanation (e.g. "this exceeds the $15,000 threshold; the account manager should be looped in") rather than an automated step.