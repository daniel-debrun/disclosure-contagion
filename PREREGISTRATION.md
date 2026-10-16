# Pre-registration

Written 2026-10-15, before any model was fitted or any outcome computed. The only data looked at so far are counts
of 8-K items per year and the size of industry groups. Anything changed after results exist goes in an
**Amendments** section at the end, with the reason.

## Questions

1. **Contagion.** When an S&P 500 company files an unscheduled 8-K, do companies in the same industry file more
   unscheduled 8-Ks over the next days than they would after a filing by an unrelated company?
2. **Volatility.** Does the excitation from peers' filings improve next-week volatility forecasts beyond a HAR model
   and the company's own filings?

## Data

- **Universe.** S&P 500 members, point in time by month, 2016-07 to 2026-06. A company is a CIK; share classes are
  merged. Membership before 2016-07 (used only for look-back features) is taken as the 2016-06 list.
- **Events.** Original 8-Ks (not 8-K/A) from EDGAR, by acceptance time. A filing becomes knowable in the trading
  session it was accepted in, or the next one if accepted after 16:00 ET. A filing is an **unscheduled event** if it
  contains at least one item in

  `U = {1.01, 1.02, 1.03, 1.04, 1.05, 2.01, 2.03, 2.04, 2.05, 2.06, 3.01, 3.03, 4.01, 4.02, 5.01, 5.02}`

  and does **not** contain item 2.02 (earnings). The excluded items are scheduled or routine: 2.02 earnings, 5.07
  shareholder votes, 5.03 bylaws, 7.01 Reg FD, 8.01 other, 9.01 exhibits. `y[i,t]` is the number of such filings by
  company i in session t.
- **Robustness set.** `U_narrow = U` minus {1.01, 2.01, 2.03}: drops deals and debt issuance, which cluster with
  market-wide financing windows and with the other party to a deal.
- **Peers.** Sharadar `industry` and `sector` (current classification).
- **Prices.** Sharadar SEP daily open, high, low, close (split-adjusted).
- **Splits.** Train: sessions 2016-07-01 to 2021-12-31. Test: 2022-01-01 to 2026-06-30. Every parameter, including
  the volatility regressions, is estimated on train only. Test is evaluated once.

## Model A: discrete-time Hawkes

```
y[i,t] ~ Poisson(lam[i,t])
lam[i,t] = mu[i] + sum over lag buckets b of
             a_own[b] * O[i,t,b]      own events
           + a_ind[b] * I[i,t,b]      events by other members of i's industry
           + a_sec[b] * S[i,t,b]      events by members of i's sector, other industries
           + a_mkt[b] * M[i,t,b]      events by members of other sectors
         + a_seas * Q[i,t]            industry peers' events one year earlier (sessions t-262..t-242)
```

- Lag buckets `b`: 1, 2-5, 6-20, 21-60 sessions before t. Each feature is a **count** of events in the bucket (not
  divided by the number of firms), so each `a` is the extra daily filing rate per event.
- All `a >= 0` and `mu > 0` (a Hawkes process: events can only excite). Fitted by maximum likelihood.
- `Q` controls for industries whose filing calendars repeat every year.
- **Excitation per event**: `K_x = sum_b a_x[b] * width[b]`, the expected number of extra filings by firm i caused
  by one event at a firm of kind x, over 60 sessions.

**Claim A (primary).** `K_ind / K_mkt > 1`, judged by both of:

1. **Permutation test on train.** Refit 100 times with companies randomly reassigned to the (sector, industry) slots,
   keeping every group's size. p = share of permutations with a ratio at least the observed one. Pass: p < 0.05.
2. **Out-of-sample.** On test, the Poisson log-likelihood of the full model against the same model fitted with
   `a_ind = a_sec = 0`. The statistic is the per-session sum of the difference, with a Newey-West t (5 lags). Pass:
   gain > 0 and t > 2.

Claim A is **supported** if both pass, **refuted** if the test gain is <= 0, and **inconclusive** otherwise.
Reported as well: `K_own`, `K_ind`, `K_sec`, `K_mkt`, the per-bucket `a` values, and the same fit on `U_narrow`.

## Model B: volatility

- Daily variance: `v[i,t] = ln(H/L)^2 / (4 ln 2) + ln(O_t / C_{t-1})^2` (Parkinson range plus overnight gap), floored
  at 1e-6.
- Target: `log(mean(v[i,t+1..t+5]))`, made at the close of session t, using filings with session <= t only.
- **B0 (HAR):** intercept, firm `log v` over the last 1, 5 and 22 sessions, and the same three for the
  equal-weighted market mean of `v`.
- **B1:** B0 + `log(1 + own_t+1 / mu_i)`, where `own_t+1` is the own-event part of the fitted Model A intensity for
  session t+1.
- **B2:** B1 + `log(1 + peer_t+1 / mu_i)`, the industry and sector part of that intensity.
- Pooled OLS on train, using Model A's train fit; evaluated on test.

**Claim B (primary).** B2 has lower test MSE than B1. The statistic is the per-session mean (over firms) of the
squared-error difference, with a Newey-West t (5 lags, since targets overlap). Pass: MSE lower and t > 2.
Reported as well: B1 against B0, the % change in MSE, and the B2 coefficient.

## Checks before the real fit

- Simulated panels with known parameters: the fitter must recover `K` values, and must give a ratio near 1 when the
  simulation has no industry effect.

## What would make this wrong

- **Common shocks.** A market-wide shock makes everyone file. It loads on both `I` and `M` per event, so it cannot by
  itself make `K_ind > K_mkt`; the permutation test checks this directly.
- **Industry calendars.** Handled by excluding 2.02 and by `Q`.
- **Deal pairs.** Both sides of a merger in one industry file follow-ups. The `U_narrow` fit shows how much depends
  on this.
- **Classification.** Industry labels are current, not point in time.

## Amendments

1. **2026-10-15, before any fit: train starts 2017-01-01, not 2016-07-01.** The Sharadar SEP snapshot starts on
   2016-01-04, so the session calendar does too, and `Q` (sessions t-262..t-242) cannot be computed until early 2017.
   Starting train when every feature exists is simpler than filling the gap. Test is unchanged.
2. **2026-10-15, before any fit: implementation details the plan left open.**
   - The 19 firms (1% of member firm-days) with no Sharadar classification are not fitted, since they have no
     peers; their filings still count in `M`.
   - Test is scored on firms that have a train `mu` (members in both periods).
   - Members whose prices are missing (Sharadar lacks a few, e.g. BK, PARA) stay in Model A and drop out of Model B.
