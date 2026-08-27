# FX Conversion Policy

## Rate source
All non-USD invoices are converted using the exchange rate in effect on the invoice's `issue_date`, sourced from Treasury's daily rate feed (`exchange_rates.json` in this environment). Do not use a rate from a different date, even if it's close, without noting the discrepancy.

## Tolerance
A conversion discrepancy within ±1% of the correct rate is considered rounding noise and does not require a fix. Anything larger should be investigated and, if confirmed, corrected with a credit memo (overbilling) or a make-good invoice (underbilling).

## Known issue
EUR-denominated invoices have caused three FX-related billing disputes since Q2 2025. Finance is evaluating requiring USD-only invoicing for new EU contracts going forward — flag any new EUR invoice discrepancy as part of that pattern, not just a one-off, when reporting findings.