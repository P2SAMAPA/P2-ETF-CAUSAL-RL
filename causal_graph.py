"""causal_graph.py — Causal graph discovery over ETF + macro variables.

Three methods available (set via config.GRAPH_METHOD):

1. LiNGAM  (default, fastest)
   - DirectLiNGAM: assumes non-Gaussian noise → identifies unique causal DAG
   - Returns instantaneous causal coefficients B: X = B @ X + noise
   - Best for daily financial returns (empirically non-Gaussian)

2. PCMCI+  (most rigorous for time series)
   - Momentary Conditional Independence (MCI) tests with lag up to MAX_LAG
   - Handles lagged causal effects (e.g. VIX at t-1 causes XLE at t)
   - Requires causallearn or tigramite; falls back to Granger if unavailable

3. Granger (fastest baseline)
   - VAR(p) model, F-test for Granger causality
   - Pure scipy/statsmodels, no special deps
   - Less principled but always available

Output: CausalGraph object containing:
  adj_matrix : (n_vars, n_vars) — A[i,j] = j causally influences i
  coef_matrix: (n_vars, n_vars) — signed causal coefficients
  var_names  : list of variable names
  method     : which method was used
  fit_date   : when this graph was fitted
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

import config


@dataclass
class CausalGraph:
    """Container for a fitted causal graph."""
    adj_matrix  : np.ndarray          # (n, n) bool  — A[i,j]=1 means j→i
    coef_matrix : np.ndarray          # (n, n) float — signed edge weights
    var_names   : list[str]
    method      : str
    fit_date    : str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    n_edges     : int = 0
    fit_window  : int = 0

    def __post_init__(self):
        self.n_edges = int(self.adj_matrix.sum())

    def parents_of(self, var: str) -> list[str]:
        """Return list of variables that directly cause `var`."""
        idx = self.var_names.index(var)
        return [self.var_names[j] for j in range(len(self.var_names))
                if self.adj_matrix[idx, j]]

    def children_of(self, var: str) -> list[str]:
        """Return list of variables directly caused by `var`."""
        idx = self.var_names.index(var)
        return [self.var_names[i] for i in range(len(self.var_names))
                if self.adj_matrix[i, idx]]

    def confounders_of(self, treatment: str, outcome: str) -> list[str]:
        """Return backdoor confounders: parents of treatment that also
        affect outcome (directly or via other paths).

        Used in backdoor adjustment: condition on these to block
        spurious correlations between treatment and outcome.
        """
        t_idx = self.var_names.index(treatment)
        o_idx = self.var_names.index(outcome)
        # Simple backdoor: parents of treatment that have a path to outcome
        parents_t = [j for j in range(len(self.var_names))
                     if self.adj_matrix[t_idx, j] and j != o_idx]
        # Check if each parent also has an edge to outcome (direct confounder)
        confounders = [self.var_names[j] for j in parents_t
                       if self.adj_matrix[o_idx, j]]
        return confounders[: config.BACKDOOR_MAX_CONFOUNDERS]


# ── Method 1: LiNGAM ─────────────────────────────────────────────────────────

def _fit_lingam(data: np.ndarray, var_names: list[str]) -> CausalGraph:
    """Fit DirectLiNGAM to extract instantaneous causal DAG.

    Falls back to correlation-thresholded DAG if lingam not installed.
    """
    try:
        import lingam
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = lingam.DirectLiNGAM()
            model.fit(data)
        B = model.adjacency_matrix_   # (n, n) B[i,j] = effect of j on i
        # Threshold small coefficients
        B_thresh = np.where(np.abs(B) >= config.LINGAM_THRESHOLD, B, 0.0)
        adj = (B_thresh != 0).astype(float)
        return CausalGraph(
            adj_matrix=adj,
            coef_matrix=B_thresh,
            var_names=var_names,
            method="lingam",
            fit_window=len(data),
        )
    except ImportError:
        pass

    # Fallback: correlation-based DAG (topological order by variance)
    return _fit_correlation_dag(data, var_names, method_name="lingam_fallback")


# ── Method 2: PCMCI+ ─────────────────────────────────────────────────────────

def _fit_pcmci(data: np.ndarray, var_names: list[str]) -> CausalGraph:
    """Fit PCMCI+ with MCI conditional independence tests.

    Falls back to Granger if tigramite not installed.
    """
    try:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_tg  = pp.DataFrame(data, var_names=var_names)
            pcmci  = PCMCI(dataframe=df_tg, cond_ind_test=ParCorr(), verbosity=0)
            results = pcmci.run_pcmciplus(
                tau_min=0, tau_max=config.MAX_LAG,
                pc_alpha=config.PCMCI_ALPHA,
            )

        n = len(var_names)
        adj   = np.zeros((n, n))
        coefs = np.zeros((n, n))
        # Aggregate over lags: take strongest significant link
        for lag in range(config.MAX_LAG + 1):
            for i in range(n):
                for j in range(n):
                    p_val = results["p_matrix"][i, j, lag]
                    if p_val < config.PCMCI_ALPHA:
                        val = results["val_matrix"][i, j, lag]
                        if abs(val) > abs(coefs[i, j]):
                            adj[i, j]   = 1.0
                            coefs[i, j] = val

        np.fill_diagonal(adj, 0)
        np.fill_diagonal(coefs, 0)

        return CausalGraph(
            adj_matrix=adj,
            coef_matrix=coefs,
            var_names=var_names,
            method="pcmci",
            fit_window=len(data),
        )
    except ImportError:
        pass

    return _fit_granger(data, var_names)


# ── Method 3: Granger causality (statsmodels) ─────────────────────────────────

def _fit_granger(data: np.ndarray, var_names: list[str]) -> CausalGraph:
    """Granger causality via VAR(p) F-tests. Pure statsmodels."""
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        return _fit_correlation_dag(data, var_names, method_name="granger_fallback")

    n   = len(var_names)
    adj = np.zeros((n, n))
    coef = np.zeros((n, n))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                try:
                    # Test: does j Granger-cause i?
                    test_data = np.column_stack([data[:, i], data[:, j]])
                    results   = grangercausalitytests(
                        test_data, maxlag=config.MAX_LAG, verbose=False
                    )
                    # Use minimum p-value across lags
                    min_p = min(
                        results[lag][0]["ssr_ftest"][1]
                        for lag in range(1, config.MAX_LAG + 1)
                        if lag in results
                    )
                    if min_p < config.GRANGER_ALPHA:
                        adj[i, j]  = 1.0
                        # Approximate coefficient: correlation as proxy
                        coef[i, j] = float(np.corrcoef(
                            data[config.MAX_LAG:, i],
                            data[: -config.MAX_LAG, j]
                        )[0, 1])
                except Exception:
                    pass

    return CausalGraph(
        adj_matrix=adj,
        coef_matrix=coef,
        var_names=var_names,
        method="granger",
        fit_window=len(data),
    )


# ── Fallback: correlation-based DAG ──────────────────────────────────────────

def _fit_correlation_dag(
    data: np.ndarray,
    var_names: list[str],
    method_name: str = "corr_dag",
    threshold: float = 0.20,
) -> CausalGraph:
    """Correlation-based approximate DAG.

    Topological order: higher-variance variables are treated as more
    exogenous (causes), lower-variance as more endogenous (effects).
    Edges added where |corr| > threshold.
    """
    n    = len(var_names)
    corr = np.corrcoef(data.T)                    # (n, n)
    var  = data.var(axis=0)
    order = np.argsort(var)[::-1]                 # high variance = exogenous

    adj  = np.zeros((n, n))
    coef = np.zeros((n, n))

    for rank_i, i in enumerate(order):
        for rank_j, j in enumerate(order):
            if rank_j >= rank_i:                  # only earlier in order can cause
                continue
            if abs(corr[i, j]) >= threshold:
                adj[i, j]  = 1.0                  # j causes i
                coef[i, j] = corr[i, j]

    return CausalGraph(
        adj_matrix=adj,
        coef_matrix=coef,
        var_names=var_names,
        method=method_name,
        fit_window=len(data),
    )


# ── Public fit function ───────────────────────────────────────────────────────

def fit_causal_graph(
    data: np.ndarray,
    var_names: list[str],
    method: str = config.GRAPH_METHOD,
) -> CausalGraph:
    """Fit a causal graph using the specified method.

    Parameters
    ----------
    data      : (T, n_vars) array — z-scored returns + macro
    var_names : list of variable names aligned to columns
    method    : "lingam" | "pcmci" | "granger"

    Returns
    -------
    CausalGraph with adjacency matrix, coefficients, variable names
    """
    print(f"    Fitting causal graph [{method}] on {len(data)} obs × {len(var_names)} vars...", end=" ")

    if method == "lingam":
        g = _fit_lingam(data, var_names)
    elif method == "pcmci":
        g = _fit_pcmci(data, var_names)
    elif method == "granger":
        g = _fit_granger(data, var_names)
    else:
        raise ValueError(f"Unknown graph method: {method}")

    print(f"→ {g.n_edges} causal edges found (method={g.method})")
    return g
