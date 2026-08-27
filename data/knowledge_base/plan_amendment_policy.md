# Plan Amendment Policy

## What counts as an amendment
An amendment updates an existing plan's total value, cadence, or entitlements when the underlying agreement changes (e.g. an upgrade or a renegotiated renewal). The original plan_id is preserved for history — it's identified by another plan's `amends` field pointing back to it.

## Source of truth
Once a plan is amended, the original version is no longer the billing source of truth as of the amendment's `start_date`. When discussing billing for a plan with an amendment chain, always state which version is currently in effect and since when — customers have been confused in the past by being quoted a superseded total.

## Effective dating
If a question concerns a period before the amendment's `start_date`, resolve the plan `as_of_date` that period, not the current version.