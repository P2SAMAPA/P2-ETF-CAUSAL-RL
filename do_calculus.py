"""do_calculus.py — Do-calculus: backdoor adjustment for interventional returns.

Core idea
---------
Standard RL uses P(return | features) — observational.
Causal RL uses P(return | do(macro = x)) — interventional.

The difference:
  Observational: "When VIX was high, XLE returned -0.8%" 
                 (includes ALL reasons VIX was high simultaneously)
  Interventional: "If we FORCED VIX to be high (holding everything else constant),
                   XLE would return -0.3%"
                 (isolates VIX's DIRECT causal effect on XLE only)

Backdoor adjustment formula (Pearl, 2009):
  P(Y=y | do(X=x)) = Σ_z P(Y=y | X=x, Z=z) * P(Z=z)

  where Z = backdoor adjustment set = confounders of (X → Y)
        = variables that block all backdoor paths from X to Y

Implementation
--------------
We discretise each macro variable into N_INTERVENTION_BINS quantile bins.
For each ETF:
  1. Find confounders Z from the causal graph
  2. Compute E[r_etf | macro_bin=b, Z=z] for each (b, z) cell
  3. Weight by P(Z=z) → marginalise out confounders
  4. Result: E[r_etf | do(macro_bin=b)] for each ETF

This produces an interventional_return vector (n_etf,) for today's macro state,
which is the causal state fed to the RL policy.

Counterfactual reward shaping
-----------------------------
For each action taken by the policy, we also compute the return under
CF_N_SAMPLES alternative macro interventions. If the actual portfolio
return is much lower than the counterfactual average, the policy was
exploiting a spurious correlation → penalise.

  cf_regret = max(0, mean(cf_returns) - actual_return)
  reward    = sharpe_return - CF_PENALTY_WT * cf_regret
"""

from __future__ import annotations

import numpy as np

import config
from causal_graph import CausalGraph


# ── Interventional distribution lookup table ──────────────────────────────────

class InterventionalModel:
    """Pre-computed lookup table: E[r_etf | do(macro_bin=b)] per ETF.

    Built once per causal graph refit; queried at every time step.

    Parameters
    ----------
    graph      : fitted CausalGraph
    data       : (T, n_vars) array aligned to graph.var_names
    etf_names  : list of ETF ticker names
    macro_names: list of macro variable names
    """

    def __init__(
        self,
        graph:       CausalGraph,
        data:        np.ndarray,
        etf_names:   list[str],
        macro_names: list[str],
    ) -> None:
        self.graph       = graph
        self.etf_names   = etf_names
        self.macro_names = macro_names
        self.var_names   = graph.var_names
        self.n_bins      = config.N_INTERVENTION_BINS

        # Build quantile bin edges for each macro variable
        self._bin_edges: dict[str, np.ndarray] = {}
        for mac in macro_names:
            if mac in self.var_names:
                idx  = self.var_names.index(mac)
                vals = data[:, idx]
                self._bin_edges[mac] = np.quantile(
                    vals, np.linspace(0, 1, self.n_bins + 1)
                )

        # Build lookup table: E[r_etf | macro_bin, confounder_bin] per ETF
        self._tables = self._build_tables(data)

    def _get_bin(self, mac: str, value: float) -> int:
        """Map a macro value to its quantile bin index [0, n_bins-1]."""
        edges = self._bin_edges.get(mac)
        if edges is None:
            return self.n_bins // 2
        return int(np.clip(np.searchsorted(edges, value) - 1, 0, self.n_bins - 1))

    def _build_tables(
        self, data: np.ndarray
    ) -> dict[str, dict]:
        """For each ETF, build E[r | do(macro)] lookup table.

        Structure: tables[etf][macro] = array of shape (n_bins,)
                   giving E[r_etf | do(macro_var = bin_b)]
        """
        tables: dict[str, dict] = {}

        for etf in self.etf_names:
            if etf not in self.var_names:
                continue
            etf_idx  = self.var_names.index(etf)
            etf_ret  = data[:, etf_idx]
            tables[etf] = {}

            for mac in self.macro_names:
                if mac not in self.var_names:
                    continue
                mac_idx = self.var_names.index(mac)
                mac_val = data[:, mac_idx]

                # Find backdoor confounders for (mac → etf)
                confounders = self.graph.confounders_of(mac, etf)

                if not confounders:
                    # No confounders → simple conditional mean per bin
                    bin_means = np.zeros(self.n_bins)
                    for b in range(self.n_bins):
                        edges  = self._bin_edges[mac]
                        lo, hi = edges[b], edges[b + 1]
                        mask   = (mac_val >= lo) & (mac_val < hi)
                        if b == self.n_bins - 1:
                            mask = (mac_val >= lo)
                        bin_means[b] = float(etf_ret[mask].mean()) if mask.sum() > 5 else 0.0
                    tables[etf][mac] = bin_means

                else:
                    # Backdoor adjustment: condition on confounders, then marginalise
                    # P(r_etf | do(mac=b)) = Σ_z P(r_etf | mac=b, Z=z) * P(Z=z)
                    conf_idxs = [self.var_names.index(c) for c in confounders
                                 if c in self.var_names]

                    if not conf_idxs:
                        # Confounders not in data → fall back to simple conditional
                        bin_means = np.zeros(self.n_bins)
                        for b in range(self.n_bins):
                            edges  = self._bin_edges[mac]
                            lo, hi = edges[b], edges[b + 1]
                            mask   = (mac_val >= lo) & (mac_val < hi)
                            if b == self.n_bins - 1:
                                mask = (mac_val >= lo)
                            bin_means[b] = float(etf_ret[mask].mean()) if mask.sum() > 5 else 0.0
                        tables[etf][mac] = bin_means
                        continue

                    # Discretise first confounder (keep it tractable)
                    conf_idx = conf_idxs[0]
                    conf_val = data[:, conf_idx]
                    conf_edges = np.quantile(conf_val, np.linspace(0, 1, self.n_bins + 1))

                    bin_means = np.zeros(self.n_bins)
                    for b in range(self.n_bins):
                        mac_edges = self._bin_edges[mac]
                        mac_lo, mac_hi = mac_edges[b], mac_edges[b + 1]
                        if b == self.n_bins - 1:
                            mac_mask = mac_val >= mac_lo
                        else:
                            mac_mask = (mac_val >= mac_lo) & (mac_val < mac_hi)

                        # Marginalise over confounder bins
                        weighted_sum = 0.0
                        for cb in range(self.n_bins):
                            c_lo = conf_edges[cb]
                            c_hi = conf_edges[cb + 1]
                            if cb == self.n_bins - 1:
                                conf_mask = conf_val >= c_lo
                            else:
                                conf_mask = (conf_val >= c_lo) & (conf_val < c_hi)

                            cell_mask = mac_mask & conf_mask
                            p_z       = conf_mask.mean()           # P(Z=z)
                            cond_mean = (float(etf_ret[cell_mask].mean())
                                         if cell_mask.sum() > 3 else 0.0)
                            weighted_sum += cond_mean * p_z

                        bin_means[b] = weighted_sum

                    tables[etf][mac] = bin_means

        return tables

    def interventional_return(
        self,
        macro_values: dict[str, float],
    ) -> dict[str, float]:
        """Compute E[r_etf | do(macro = current_values)] for all ETFs.

        Parameters
        ----------
        macro_values : dict mac_name → current value (z-scored)

        Returns
        -------
        dict etf → interventional expected return
        """
        result: dict[str, float] = {}

        for etf in self.etf_names:
            if etf not in self._tables:
                result[etf] = 0.0
                continue

            # Aggregate interventional return across all macro variables
            # Weight by causal coefficient from graph (stronger causal link = more weight)
            weighted_sum = 0.0
            weight_total = 0.0

            for mac, val in macro_values.items():
                if mac not in self._tables.get(etf, {}):
                    continue
                b      = self._get_bin(mac, val)
                ir     = self._tables[etf][mac][b]

                # Edge weight from causal graph
                if mac in self.var_names and etf in self.var_names:
                    mac_idx = self.var_names.index(mac)
                    etf_idx = self.var_names.index(etf)
                    w = abs(self.graph.coef_matrix[etf_idx, mac_idx]) + 0.01
                else:
                    w = 0.01

                weighted_sum += ir * w
                weight_total += w

            result[etf] = float(weighted_sum / max(weight_total, 1e-8))

        return result

    def counterfactual_returns(
        self,
        macro_values: dict[str, float],
        n_samples: int = config.CF_N_SAMPLES,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample n_samples counterfactual interventional returns.

        For each counterfactual sample:
          - Draw random macro bin assignments (different from actual)
          - Compute interventional return under those hypothetical macro states

        Returns
        -------
        cf_returns : (n_samples, n_etf) array of counterfactual returns
        """
        if rng is None:
            rng = np.random.default_rng()

        etfs = [e for e in self.etf_names if e in self._tables]
        n_etf = len(etfs)
        cf_returns = np.zeros((n_samples, n_etf))

        for s in range(n_samples):
            # Perturb macro values: draw from uniform bins for each macro var
            cf_macro = {}
            for mac, val in macro_values.items():
                if mac in self._bin_edges:
                    edges = self._bin_edges[mac]
                    # Pick a random bin different from actual
                    actual_b = self._get_bin(mac, val)
                    other_bins = [b for b in range(self.n_bins) if b != actual_b]
                    if other_bins:
                        b_cf = rng.choice(other_bins)
                        cf_macro[mac] = float(
                            (edges[b_cf] + edges[b_cf + 1]) / 2
                        )
                    else:
                        cf_macro[mac] = val
                else:
                    cf_macro[mac] = val

            cf_ir = self.interventional_return(cf_macro)
            for k, etf in enumerate(etfs):
                cf_returns[s, k] = cf_ir.get(etf, 0.0)

        return cf_returns
