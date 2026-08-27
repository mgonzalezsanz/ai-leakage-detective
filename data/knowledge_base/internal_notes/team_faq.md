# Finance Team FAQ

## Who can apply a proposed fix?
Any finance analyst can apply corrections under $15,000 with standard customer confirmation. At or above that (or a lower account-specific threshold — check `customer_accounts.md`), account-manager sign-off is required first.

## What if a customer already has a credit memo for the issue I'm looking at?
Never propose a duplicate fix. Always call `query_credit_memos` first and cite the existing memo (ID, amount, reason) if one covers the issue.

## How far back can we issue a make-good invoice?
Within the same fiscal year as the missed billing period. Anything crossing into a prior fiscal year needs Finance leadership sign-off before drafting — see `revenue_recognition_policy.md`.