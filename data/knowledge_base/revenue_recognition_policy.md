# Revenue Recognition Policy

## General principle
Revenue is recognized according to the billing plan in effect for the period being invoiced, not the plan's original terms if it has since been amended. When a plan has an `amends` chain, the version whose `start_date` covers the invoice's period is the source of truth.

## Missed or underbilled periods
Any period where invoiced revenue falls short of the plan's expected amount (prorated to the plan's cadence) must be investigated before fiscal quarter-close. Once confirmed, the shortfall should be remediated with a make-good invoice referencing the original billing period — see `make_good_invoice_policy.md`.

## Timing
Make-good invoices should be issued within the same fiscal year as the missed billing period wherever possible. Corrections spanning a prior fiscal year require Finance leadership sign-off before drafting, regardless of dollar amount.
