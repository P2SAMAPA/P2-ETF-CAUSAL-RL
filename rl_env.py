"""rl_env.py — Causal RL Gymnasium environment.

State space
-----------
For each time step t the observation vector contains:
  [interventional_returns (n_etf),     ← E[r | do(macro)] per ETF
   raw_returns_window (n_etf * window), ← last ENV_WINDOW days of raw returns
   macro_values (n_macro),              ← current macro z-scores
   portfolio_weights (n_etf + 1)]       ← current allocation incl. CASH
Total dim = n_etf + n_etf*window + n_macro + (n_etf+1)

Action space
------------
Continuous: (n_etf + 1,) — softmax → portfolio weights including CASH.
CASH weight clipped to [0, CASH_WEIGHT_MAX].

Reward
------
  step_return   = portfolio_return - transaction_costs
  sharpe_reward = step_return / (running_vol + eps)   ← risk-adjusted
  cf_regret     = max(0, mean(cf_portfolio_returns) - portfolio_return)
  reward        = REWARD_SCALING * (sharpe_reward - CF_PENALTY_WT * cf_regret)

Episode structure
-----------------
Each episode draws a random contiguous segment of TRAIN_END history.
Length: MAX_EPISODE_STEPS (252 days = 1 year).
"""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

import config
from do_calculus import InterventionalModel


class CausalRLEnv(gym.Env):
    """Causal RL environment for ETF portfolio allocation.

    Parameters
    ----------
    ret_arr      : (T, n_etf) — full history of log returns
    macro_arr    : (T, n_macro) — full history of macro values (z-scored)
    int_model    : fitted InterventionalModel
    etf_names    : list of ETF tickers
    macro_names  : list of macro variable names
    episode_idx  : starting index for this episode (None = random)
    mode         : "train" | "eval" — eval mode uses fixed episode starts
    rng          : numpy random generator
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        ret_arr:    np.ndarray,
        macro_arr:  np.ndarray,
        int_model:  InterventionalModel,
        etf_names:  list[str],
        macro_names: list[str],
        episode_idx: int | None = None,
        mode:        str = "train",
        rng:         np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        self.ret_arr     = ret_arr.astype(np.float32)
        self.macro_arr   = macro_arr.astype(np.float32)
        self.int_model   = int_model
        self.etf_names   = etf_names
        self.macro_names = macro_names
        self.n_etf       = len(etf_names)
        self.n_macro     = len(macro_names)
        self.mode        = mode
        self.rng         = rng or np.random.default_rng(42)
        self._fixed_start = episode_idx

        # ── Spaces ────────────────────────────────────────────────────────────
        obs_dim = (
            self.n_etf                           # interventional returns
            + self.n_etf * config.ENV_WINDOW     # raw return window
            + self.n_macro                        # macro values
            + self.n_etf + 1                      # current weights incl. CASH
        )
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(obs_dim,), dtype=np.float32,
        )
        # Action: unnormalised logits → softmax → weights
        self.action_space = spaces.Box(
            low=-3.0, high=3.0,
            shape=(self.n_etf + 1,), dtype=np.float32,  # +1 for CASH
        )

        # ── Episode state ─────────────────────────────────────────────────────
        self._t          = 0
        self._start      = 0
        self._weights    = np.zeros(self.n_etf + 1, dtype=np.float32)
        self._weights[-1] = 1.0     # start fully in CASH
        self._returns_hist: list[float] = []
        self._prev_weights = self._weights.copy()

    # ── Observation builder ───────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        t = self._start + self._t

        # 1. Interventional returns
        mac_dict = {
            self.macro_names[k]: float(self.macro_arr[t, k])
            for k in range(self.n_macro)
        }
        int_ret = self.int_model.interventional_return(mac_dict)
        ir_vec  = np.array([int_ret.get(e, 0.0) for e in self.etf_names],
                           dtype=np.float32)

        # 2. Raw return window
        win_start = max(0, t - config.ENV_WINDOW)
        raw_win   = self.ret_arr[win_start:t]
        if len(raw_win) < config.ENV_WINDOW:
            pad = np.zeros((config.ENV_WINDOW - len(raw_win), self.n_etf),
                           dtype=np.float32)
            raw_win = np.concatenate([pad, raw_win], axis=0)
        raw_flat = raw_win.ravel()                          # (n_etf * ENV_WINDOW,)

        # 3. Macro values
        mac_vec = self.macro_arr[t]                         # (n_macro,)

        # 4. Current weights
        w_vec = self._weights                               # (n_etf + 1,)

        obs = np.concatenate([ir_vec, raw_flat, mac_vec, w_vec])
        return np.clip(obs, -10.0, 10.0)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        min_start = config.ENV_WINDOW + config.GRAPH_WINDOW
        max_start = len(self.ret_arr) - config.MAX_EPISODE_STEPS - 1

        if self._fixed_start is not None:
            self._start = self._fixed_start
        elif self.mode == "train":
            self._start = int(self.rng.integers(min_start, max(min_start + 1, max_start)))
        else:
            self._start = min_start

        self._t             = 0
        self._weights       = np.zeros(self.n_etf + 1, dtype=np.float32)
        self._weights[-1]   = 1.0
        self._prev_weights  = self._weights.copy()
        self._returns_hist  = []

        return self._get_obs(), {}

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        t = self._start + self._t

        # ── Action → portfolio weights via softmax ────────────────────────────
        logits = action.astype(np.float64)
        logits -= logits.max()                              # numerical stability
        exp_l  = np.exp(logits)
        weights = exp_l / exp_l.sum()

        # Clip CASH weight
        weights[-1] = np.clip(weights[-1], 0.0, config.CASH_WEIGHT_MAX)
        # Renormalise ETF portion
        etf_sum = weights[:-1].sum()
        if etf_sum > 1e-8:
            weights[:-1] *= (1.0 - weights[-1]) / etf_sum
        weights = weights.astype(np.float32)

        # ── Realised returns ──────────────────────────────────────────────────
        if t + 1 < len(self.ret_arr):
            next_ret = self.ret_arr[t + 1]                 # (n_etf,)
        else:
            next_ret = np.zeros(self.n_etf, dtype=np.float32)

        # Portfolio return (ETF portion only; CASH earns 0 in simplified model)
        port_ret = float(np.dot(weights[:-1], next_ret))

        # Transaction cost: L1 change in weights * cost
        tc = float(np.abs(weights - self._prev_weights).sum()) * config.TRANSACTION_COST
        net_ret = port_ret - tc

        # ── Sharpe-scaled reward ──────────────────────────────────────────────
        self._returns_hist.append(net_ret)
        if len(self._returns_hist) > 21:
            vol = float(np.std(self._returns_hist[-21:])) + 1e-6
        else:
            vol = 1e-6
        sharpe_r = net_ret / vol

        # ── Counterfactual penalty ────────────────────────────────────────────
        mac_dict = {
            self.macro_names[k]: float(self.macro_arr[t, k])
            for k in range(self.n_macro)
        }
        cf_ir = self.int_model.counterfactual_returns(
            mac_dict, n_samples=config.CF_N_SAMPLES, rng=self.rng
        )                                                   # (CF_N_SAMPLES, n_etf)
        cf_port = float((cf_ir @ weights[:-1]).mean())      # mean CF portfolio return
        cf_regret = max(0.0, cf_port - net_ret)

        reward = config.REWARD_SCALING * (
            sharpe_r - config.CF_PENALTY_WT * cf_regret
        )

        # ── Advance state ─────────────────────────────────────────────────────
        self._prev_weights = weights.copy()
        self._weights      = weights.copy()
        self._t           += 1

        terminated = (t + 1 >= len(self.ret_arr) - 1)
        truncated  = (self._t >= config.MAX_EPISODE_STEPS)

        info = {
            "port_ret":   net_ret,
            "tc":         tc,
            "cf_regret":  cf_regret,
            "reward":     reward,
            "weights":    weights.tolist(),
        }

        return self._get_obs(), float(reward), terminated, truncated, info
