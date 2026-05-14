# 🧬 P2-ETF-CAUSAL-RL

**P2Quant Engine** · Causal Reinforcement Learning · Do-Calculus + PPO · ETF Ranking

[![Causal RL Daily Inference](https://github.com/P2SAMAPA/P2-ETF-CAUSAL-RL/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-CAUSAL-RL/actions/workflows/daily_run.yml)
[![Causal RL Training](https://github.com/P2SAMAPA/P2-ETF-CAUSAL-RL/actions/workflows/causal_train.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-CAUSAL-RL/actions/workflows/causal_train.yml)

---

## What Is This?

The only engine in the P2Quant suite that reasons about **cause and effect** rather than correlation. It combines:

1. **Causal graph discovery** (LiNGAM / PCMCI+ / Granger) — learns which variables *cause* ETF returns
2. **Backdoor adjustment** (Pearl's do-calculus) — computes interventional returns `E[r | do(macro=x)]`
3. **PPO actor-critic** — policy trained on causal state, not raw correlational features
4. **Counterfactual reward shaping** — penalises the policy for exploiting spurious correlations

---

## Why Causal RL vs Standard RL?

| Property | Your PPO/A2C/DQN engines | Causal RL |
|---|---|---|
| State | Raw correlational features | Interventional return `E[r\|do(macro)]` |
| Learns | "VIX high → avoid XLE" (correlation) | "VIX *causes* XLE to drop" (mechanism) |
| Regime breaks | Policy fails when correlation breaks | Robust — causal mechanism unchanged |
| Spurious signals | Learned and exploited | Actively penalised via CF reward |
| Interpretability | Black box | Explains *why* via causal graph |

---

## Pipeline

```
Causal Graph Discovery (LiNGAM / Granger / PCMCI+)
         ↓  252-day rolling window → DAG over ETFs + macro
Backdoor Adjustment (do-calculus)
         ↓  E[r_etf | do(macro=x)] per ETF — strips spurious paths
CausalRLEnv
         ↓  State = interventional returns + raw window + macro + weights
PPO Actor-Critic
         ↓  Action = portfolio weights (ETFs + CASH)
         ↓  Reward = Sharpe(net_return) − CF_PENALTY × counterfactual_regret
Walk-Forward Inference
         ↓  Daily: load checkpoint → causal state → weights → scores
```

---

## Two Workflows

### 1. `causal_train.yml` — Manual, run weekly

```
GitHub → Actions → "Causal RL Training (Manual)" → Run workflow
```

Steps per universe:
- Fit causal graph on full training data (2008 → 2019)
- Build backdoor adjustment lookup table (InterventionalModel)
- Train PPO for `N_TRAIN_UPDATES` (default 200) iterations
- Validate on 2020–2021 (COVID stress test)
- Save policy `.pt` + causal graph `.pkl` + metadata to `P2SAMAPA/p2-etf-causal-rl-model`

**Run this first before daily inference will work.**

### 2. `daily_run.yml` — Automated, Mon-Fri 22:30 UTC

- Loads policy checkpoint + causal graph from HF
- Refits the causal graph every 63 days (quarterly) on rolling window
- Computes interventional returns for today's macro state
- PPO policy outputs portfolio weights deterministically
- Score = `policy_weight × interventional_return`, z-scored cross-sectionally
- Pushes results to `P2SAMAPA/p2-etf-causal-rl-results`

---

## Do-Calculus: How Backdoor Adjustment Works

### The problem with observational data

When VIX is high, XLE tends to drop. But is VIX *causing* XLE to drop, or are both driven by a common factor (e.g. oil supply shock)?

If an oil shock causes *both* VIX to spike *and* XLE to drop, the observed correlation VIX↔XLE is **spurious** — caused by the confounder (oil), not VIX directly.

### The backdoor adjustment formula (Pearl, 2009)

```
P(r_etf | do(VIX = high)) = Σ_z P(r_etf | VIX=high, Z=z) × P(Z=z)

where Z = backdoor adjustment set
        = common causes of VIX and r_etf (confounders)
```

By conditioning on confounders Z and then marginalising, we block the spurious path and isolate VIX's direct causal effect on ETF returns.

### Implementation

1. Discretise each macro variable into `N_INTERVENTION_BINS = 5` quantile bins
2. For each (ETF, macro variable) pair:
   - Find backdoor confounders from the causal graph
   - Compute `E[r_etf | macro_bin=b, Z=z]` for each cell
   - Marginalise over `P(Z=z)` → `E[r_etf | do(macro_bin=b)]`
3. At inference: look up today's macro bin → retrieve interventional return

---

## Counterfactual Reward Shaping

At each environment step:

```python
# Actual portfolio return
port_ret = weights @ next_returns - transaction_costs

# Sample CF_N_SAMPLES = 10 alternative macro states
cf_macro_samples = [random_macro_bins for _ in range(10)]
cf_returns = [interventional_return(cf_macro) @ weights for cf_macro in cf_macro_samples]

# Counterfactual regret: how much better could we have done under other macro states?
cf_regret = max(0, mean(cf_returns) - port_ret)

# Final reward: Sharpe-scaled actual return, penalised by counterfactual regret
reward = REWARD_SCALING × (sharpe(port_ret) − CF_PENALTY_WT × cf_regret)
```

If the policy is exploiting a spurious correlation (e.g. always buys XLE when VIX is high because they happened to co-move in training), the counterfactual returns under other macro states will be much higher → large cf_regret → policy learns to stop exploiting that spurious signal.

---

## Causal Graph Methods

| Method | Speed | Accuracy | Notes |
|---|---|---|---|
| `lingam` (default) | Fast | High for non-Gaussian data | Requires `pip install lingam` |
| `granger` | Fastest | Moderate | Always available via statsmodels |
| `pcmci` | Slow | Highest for time series | Requires `pip install tigramite` |

Set via `config.GRAPH_METHOD`. Falls back gracefully if optional deps missing.

---

## Universes & HuggingFace Repos

| Universe | Tickers |
|---|---|
| EQUITY_SECTORS | SPY QQQ XLK XLF XLE XLV XLI XLY XLP XLU GDX XME IWF XSD XBI IWM |
| FI_COMMODITIES | TLT VCIT LQD HYG VNQ GLD SLV |
| COMBINED | All above |

| HF Repo | Type | Content |
|---|---|---|
| `P2SAMAPA/p2-etf-causal-rl-model` | Model | Policy `.pt` + graph `.pkl` + metadata |
| `P2SAMAPA/p2-etf-causal-rl-results` | Dataset | Daily scores, weights, interventional returns |

---

## Output Files (per universe)

| File | Content |
|---|---|
| `causal_rl_YYYY-MM-DD_{slug}.json` | Latest scores, weights, E[r\|do], config |
| `daily_{slug}.csv` | Top pick, CASH flag, CASH weight, mean IR, n_edges |
| `scores_{slug}.csv` | Full composite score history |
| `weights_{slug}.csv` | Full PPO policy weight history (incl. CASH) |
| `ir_{slug}.csv` | Full interventional return E[r\|do(macro)] history |
| `rankings_{slug}.csv` | Full rank history |

---

## Streamlit Dashboard — 5 Tabs

1. **Rankings & Scores** — composite score bar, policy weight vs IR scatter, top-N cards
2. **Interventional Returns** — E[r\|do(macro)] time-series + heatmap
3. **Policy Weights** — stacked area weight chart, CASH weight over time
4. **Score History** — composite score time-series + heatmap, top-pick frequency
5. **Full Table** — all scores, weights, IR + checkpoint info + daily summary

---

## References

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference.* Cambridge.
- Shimizu, S. et al. (2006). *A Linear Non-Gaussian Acyclic Model for Causal Discovery.* JMLR.
- Runge, J. et al. (2019). *Detecting and Quantifying Causal Associations in Large Nonlinear Time Series Datasets.* Science Advances.
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv.
- Buesing, L. et al. (2019). *Woulda, Coulda, Shoulda: Counterfactually-Guided Policy Search.* ICLR.

---

*P2Quant Engine Suite · Built by P2SAMAPA*
