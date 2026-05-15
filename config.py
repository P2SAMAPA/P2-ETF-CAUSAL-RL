"""config.py — Causal RL Engine configuration.

Pipeline overview
-----------------
  1. Causal graph discovery (PCMCI+ / LiNGAM) on rolling 252d window
     → DAG over ETFs + macro variables
  2. Backdoor adjustment → interventional return E[r | do(macro=x)] per ETF
  3. Custom Gymnasium env: state = interventional features, reward = Sharpe − CF penalty
  4. PPO actor-critic trained on interventional state
  5. Walk-forward: refit graph every GRAPH_REFIT_FREQ days,
                   retrain policy every POLICY_REFIT_FREQ days

Two workflows
-------------
  causal_train.yml  — manual (weekly): fit graph + train policy → save to HF model repo
  daily_run.yml     — automated (Mon-Fri): load checkpoint → interventional state → scores
"""

import os
from datetime import datetime

# ── HuggingFace ───────────────────────────────────────────────────────────────
HF_DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE   = "master_data.parquet"
HF_MODEL_REPO  = "P2SAMAPA/p2-etf-causal-rl-model"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-causal-rl-results"
HF_TOKEN       = os.environ.get("HF_TOKEN", None)

# ── Universes ─────────────────────────────────────────────────────────────────
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM",
]
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
COMBINED_TICKERS       = sorted(set(EQUITY_SECTORS_TICKERS + FI_COMMODITIES_TICKERS))

UNIVERSES = {
    "EQUITY_SECTORS":  EQUITY_SECTORS_TICKERS,
    "FI_COMMODITIES":  FI_COMMODITIES_TICKERS,
    "COMBINED":        COMBINED_TICKERS,
}

MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# ── Dual causal graph modules ────────────────────────────────────────────────
# Module A — LiNGAM (instantaneous causality, non-Gaussian, fast)
LINGAM_WINDOW       = 252           # rolling days for LiNGAM graph (1 year)
LINGAM_REFIT_FREQ   = 63            # refit every N days (quarterly)
LINGAM_THRESHOLD    = 0.10          # min |coef| to keep edge
LINGAM_MAX_ITER     = 1000          # DirectLiNGAM max iterations

# Module B — PCMCI+ (lagged causality, rigorous conditional independence)
PCMCI_WINDOW        = 504           # rolling days for PCMCI graph (2 years)
PCMCI_REFIT_FREQ    = 63            # refit every N days (quarterly)
PCMCI_ALPHA         = 0.05          # significance level for MCI test edges
MAX_LAG             = 2             # maximum causal lag tested
# Note: PCMCI needs PCMCI_WINDOW >= 200 * n_vars for adequate power.
# 504d with 11 vars (FI) = 45 obs/var — adequate.
# 504d with 27 vars (COMBINED) = 18 obs/var — use 1008d for COMBINED.
PCMCI_WINDOW_COMBINED = 1008        # longer window for COMBINED universe (4 years)

# Shared graph settings
GRANGER_ALPHA       = 0.05          # significance level for Granger F-test (fallback)
N_INTERVENTION_BINS = 5             # macro quantile bins for do-calculus
BACKDOOR_MAX_CONFOUNDERS = 3        # max confounders in backdoor adjustment

# Legacy alias (used by engine.py for single-method runs)
GRAPH_METHOD        = "lingam"
GRAPH_WINDOW        = LINGAM_WINDOW
GRAPH_REFIT_FREQ    = LINGAM_REFIT_FREQ

# ── Do-calculus variables ─────────────────────────────────────────────────────
INTERVENTION_VARS   = MACRO_COLS    # variables we intervene on (macro only)

# ── Counterfactual penalty ─────────────────────────────────────────────────────
CF_N_SAMPLES        = 10            # counterfactual macro draws per step
CF_PENALTY_WT       = 0.20          # weight on counterfactual penalty in reward
                                    # reward = sharpe_ret - CF_PENALTY_WT * cf_regret

# ── RL environment ────────────────────────────────────────────────────────────
ENV_WINDOW          = 21            # lookback window inside env state
TRANSACTION_COST    = 0.0010        # 10bps per trade (one-way)
REWARD_SCALING      = 10.0          # scale rewards for stable PPO training
MAX_EPISODE_STEPS   = 252           # max steps per training episode (1 year)
CASH_WEIGHT_MAX     = 0.40          # maximum CASH allocation in action space

# ── PPO hyper-parameters ──────────────────────────────────────────────────────
PPO_HIDDEN          = [256, 128]    # actor + critic MLP hidden dims
PPO_LR              = 3e-4
PPO_GAMMA           = 0.99          # discount factor
PPO_GAE_LAMBDA      = 0.95          # GAE lambda
PPO_CLIP_EPS        = 0.20          # PPO clip epsilon
PPO_ENTROPY_COEF    = 0.01          # entropy bonus (exploration)
PPO_VALUE_COEF      = 0.50          # value loss coefficient
PPO_GRAD_CLIP       = 0.50          # gradient clipping norm
PPO_EPOCHS_PER_UPDATE = 4           # gradient steps per rollout
PPO_BATCH_SIZE      = 64            # minibatch size per PPO update
PPO_ROLLOUT_STEPS   = 512           # steps per rollout collection
N_TRAIN_UPDATES     = 200           # total PPO update iterations per training run
POLICY_REFIT_FREQ   = 21            # refit policy every N days in walk-forward

# ── Data splits ───────────────────────────────────────────────────────────────
TRAIN_END           = "2019-12-31"  # policy training cutoff
VALIDATE_END        = "2021-12-31"  # validation (incl. COVID)
OOS_START           = "2022-01-01"  # live scoring from here

# ── Scoring ───────────────────────────────────────────────────────────────────
TOP_N               = 6
CASH_THRESHOLD      = -0.30         # composite z-score below → full CASH

# ── Checkpoint filenames — one set per module per universe ───────────────────
CKPT_POLICY         = "causal_rl_policy_{slug}.pt"
CKPT_GRAPH_LINGAM   = "causal_graph_lingam_{slug}.pkl"
CKPT_GRAPH_PCMCI    = "causal_graph_pcmci_{slug}.pkl"
CKPT_META           = "causal_rl_meta_{slug}.json"

# ── Output ────────────────────────────────────────────────────────────────────
TODAY = datetime.now().strftime("%Y-%m-%d")
