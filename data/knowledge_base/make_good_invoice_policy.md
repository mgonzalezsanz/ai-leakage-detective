# Make-Good Invoice Policy

## Purpose
A make-good invoice recovers revenue that should have been billed but wasn't — most commonly a missing invoice for a billing period, or an underbilled amount relative to the plan's expected total.

## Before drafting
1. Resolve the plan's current amendment (`load_plan` with no `as_of_date` returns the latest version) so the expected amount reflects any amendments.
2. Confirm the shortfall isn't already covered by an existing credit memo or a prior make-good invoice.
3. Calculate and show the arithmetic — expected amount vs. invoiced amount, in the plan's currency.

## Scope
Make-good invoices only draft an action. They are not written to the sandbox until a human explicitly approves the `apply()` call.