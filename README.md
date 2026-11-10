# Deep Hedging under Transaction Costs

A neural network that learns an option hedging policy directly from simulated paths, instead of computing a delta from a pricing model. It is evaluated against standard hedging strategies once proportional transaction costs and discrete rebalancing are included.

Based on Buehler, Gonon, Teichmann and Wood, "Deep Hedging", Quantitative Finance 19(8), 2019 (arXiv:1802.03042), following the experiment design of Sections 5.2 and 5.3, with an extension to commodity average price options.

Author: Ayushmaan (Nonu) Jerath

## The problem

A short option is hedged with futures, rebalanced daily. Terminal P&L per unit of option:

PnL = premium - payoff + sum over t of delta_t * (F_{t+1} - F_t) - sum over t of c * |delta_t - delta_{t-1}| * F_t

With no costs and continuous trading, the Black-76 delta is the optimal hedge. With proportional costs c > 0, every rebalance costs money. The optimal policy then trades less, leaving a band of holdings it is content to keep. The question is whether a network can learn that policy without being told its form.

Headline claims are always stated with their frictions. At zero transaction cost the network cannot and should not beat the Black-76 delta hedge.

## Design choices

1. Loss: entropic risk measure, rho(X) = (1/lambda) * log E[exp(-lambda * X)]. It is smooth, so it trains cleanly, and it penalises losses more than it rewards gains. The model is never trained on mean squared hedging error, which would reduce the problem to a regression on delta.

2. One lambda for the network and the Whalley-Wilmott benchmark. Minimising the entropic risk measure is equivalent to maximising exponential utility with risk aversion lambda. The Whalley-Wilmott no-trade band is derived under exponential utility. Both therefore use the same lambda, defined once in the experiment config and read by both. This makes the main comparison apples-to-apples: same risk preference, same frictions, different method.

3. Reported risk metrics: CVaR at 95% and 99% of the hedged loss, alongside mean and standard deviation of P&L. Training uses the entropic measure; reporting uses what a risk manager reads.

4. Network state: log-moneyness, time to maturity, current holding delta_{t-1}, and (under stochastic volatility) current variance. The current holding is essential: without it the network cannot know what a trade would cost, and no-trade behaviour cannot emerge.

5. Network: feedforward, 2-3 hidden layers of 32-64 units, ReLU, weights shared across time steps. Adam optimiser, learning rate 1e-3, batches of 512-1024 paths, 50k-200k simulated paths, with a held-out validation set never used for training.

6. Futures simulated with zero drift. Futures are martingales under the pricing measure, and their real-world drift over a one-month horizon is small. With non-zero drift, a risk-averse optimiser adds a directional position on top of the hedge, which would contaminate the comparison with delta. Zero drift isolates hedging from speculation.

7. Zero interest rate in the hedging simulations. Futures are margined daily and financing over a 30-day horizon is second order. This removes premium interest, margin financing and hedge tailing, so P&L reflects hedging alone; the omission of carry is noted as a limitation. The pricer itself supports a non-zero rate and is tested with one.

8. Transaction costs are charged on the opening trade at day 0 and on unwinding the hedge at expiry, identically for every strategy. This is the conservative choice; physical settlement into offsetting futures would remove the unwind cost on in-the-money paths.

9. Futures paths are simulated exactly in log space, F_{t+1} = F_t * exp(-0.5 sigma^2 dt + sigma sqrt(dt) Z), rather than by an Euler step: no discretisation bias and no negative prices.

## Baseline parameters (provisional until Phase 1 training)

- Underlying: futures, F0 = 100
- Option: European call, K = 100 (at-the-money)
- Volatility: 30% annualised
- Maturity: 30 trading days, rebalanced daily (252 trading days per year)
- Transaction cost grid: c in {0, 5, 10, 20, 50} basis points of notional per unit traded
- lambda: 1.0 (P&L in price units with F0 = 100); fixed before Phase 2 training and not changed afterwards

## Benchmarks, in order of seriousness

1. No hedge: the floor.
2. Black-76 delta, rebalanced every step: a straw man under costs. No result claims victory against this alone.
3. Black-76 delta with a Whalley-Wilmott no-trade band: the main benchmark.
4. Leland-adjusted volatility delta hedge.

## Phases and gates

No phase begins until the previous gate passes.

Phase 0 - Foundations. Black-76 pricer, Greeks, Monte Carlo, discrete delta-hedging simulator.
Gate (all three must pass):
- Put-call parity, C - P = exp(-rT) * (F - K), and its delta form, delta_call - delta_put = exp(-rT), hold to within 1e-12 across a grid of strikes (50-150), maturities (1 day to 2 years), vols (10%-100%) and rates (0%, 5%).
- Monte Carlo prices agree with the closed form across 36 independent tests (strikes 80-120, calls and puts, 1 and 6 month maturities, r = 5%). Because many tests are run, a Bonferroni correction is applied: the family-wide chance that a correct pricer fails the gate is set at 1%, giving a per-test threshold of about 3.6 standard errors. Without the correction, a correct pricer would fail a 3-standard-error test on at least one of 36 options about 9% of the time. This correction was made before the test was run. Strikes are limited to 80-120 because deep out-of-the-money estimates rest on too few in-the-money paths for the standard-error approximation to hold.
- The pricer reproduces Hull's published Black's-model futures option example (F = K = 20, r = 9%, T = 4 months, vol = 25%; published 1.12, to 2 decimals).
Secondary (not gating): agreement with CME settlement premiums on near-the-money, European-style strikes, using CME's settlement implied volatility. This tests consistency with CME's conventions; it is not an independent market test, since CME derives premiums from its own volatility surface.

Phase 0 result: PASSED.
- Put-call parity: max price gap 2.8e-14 over 672 combinations.
- Monte Carlo: max deviation 1.91 standard errors against a threshold of 3.64; none beyond 3.
- Hull example: 1.1166 against published 1.12.
- Discrete delta hedge of a short ATM call (30% vol, 30 trading days, 20,000 paths): P&L standard deviation 0.647 with daily rebalancing and 0.329 with 4x daily, against Derman-Kamal predictions of 0.667 and 0.334. At 10bp costs, mean P&L is -0.27 with daily rebalancing and -0.45 with 4x daily: halving hedging error by trading more raises cost by about two thirds.
- Record: outputs/phase0/gate_results.json. CME secondary check deferred.

Phase 1 - Zero-cost replication. GBM paths, no transaction costs. The network should approximately recover the Black-76 delta. Tolerance, fixed before training:
- Error is measured as network hedge minus Black-76 delta, at every (time, price) state visited on held-out validation paths.
- Reported as a table of RMSE bucketed by log-moneyness ln(F/K): below -0.10, -0.10 to -0.03, -0.03 to 0.03, 0.03 to 0.10, above 0.10; and by days to expiry: 30-21, 20-11, 10-5, 4-1.
- Pass requires all three: (a) pooled RMSE no greater than 0.05 (five delta) over states with 5 or more days to expiry; (b) every bucket with |ln(F/K)| no greater than 0.10 and 5 or more days to expiry has RMSE no greater than 0.08; (c) standard deviation of terminal hedged P&L within 10% of the daily Black-76 delta hedge.
- The 1-4 days to expiry buckets are reported but not gated: near expiry, at-the-money delta approaches a step function and any discrete-time hedger deviates there.

Phase 2 - Transaction costs. Proportional costs, entropic loss, benchmarks 3 and 4.
Gate: a no-trade band emerges without being coded. Shown by plotting the network's hedge against Black-76 delta, coloured by previous holding; the hysteresis should be visible.

Phase 3 - Stochastic volatility (Heston). Gate: results hold qualitatively.

Phase 4 - Commodity extension: average price option. State adds running average, fraction of fixings elapsed, and locked exposure. Benchmark delta from a moment-matching approximation (Turnbull-Wakeman / Levy), since Black-76 does not price average price options. Gate: the hedge unwinds on a sensible schedule through the pricing month.

Phase 5 - Limitations. Train at 30% vol, test at 50%. Gate: the degradation is documented and explained.

Phase 6 - Write-up and charts.

## Reproducibility

- Every script and notebook sets fixed random seeds (NumPy and PyTorch), and the seed is recorded.
- Every experiment run writes a record (config.json or gate_results.json) next to its outputs in outputs/, recording parameters, seeds and library versions.
- Validation paths are generated with a different seed from training paths and are never used for training.
- Independent tests use independent seeds.
- Every notebook is restarted and run top to bottom before its results are trusted or committed, so no result depends on hidden state from deleted or out-of-order cells.
- Exact package versions are pinned in requirements.txt.

## Repository structure

- notebooks/ - one notebook per phase, 01 to 06
- src/pricing.py - Black-76 prices and Greeks; later the average price option approximation
- src/simulate.py - path simulation (GBM, later Heston) and Monte Carlo pricing
- src/hedging.py - P&L accounting used by every strategy; later the network, losses and training loop
- src/benchmarks.py - delta, Whalley-Wilmott and Leland hedges
- outputs/ - charts, saved models, and each run's record

## Status

Phase 0 complete. Phase 1 next.