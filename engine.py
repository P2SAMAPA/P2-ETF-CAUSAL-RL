"""engine.py — Causal RL walk-forward inference engine.

Daily pipeline
--------------
1. Load policy checkpoint + causal graph from HF
2. For each OOS day t:
   a. Rebuild InterventionalModel if graph refit is due
   b. Build causal state (interventional returns + raw window + macro)
   c. Policy.get_action(state, deterministic=True) → portfolio weights
   d. Score = weight_i * sign(interventional_return_i), z-scored cross-sectionally
   e. Store weights, scores, causal signals
"""

from __future__ import annotations

import io
import pickle

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

import config
import data_manager
from causal_graph import CausalGraph, fit_causal_graph
from do_calculus import InterventionalModel
from policy import ActorCritic
from rl_env import CausalRLEnv


def _load_checkpoint(universe: str, token: str | None, device: torch.device) -> dict:
    slug = universe.lower().replace("_", "-")

    policy_file = hf_hub_download(
        repo_id=config.HF_MODEL_REPO,
        filename=config.CKPT_POLICY.format(slug=slug),
        repo_type="model", token=token, cache_dir="./hf_cache",
    )
    graph_file = hf_hub_download(
        repo_id=config.HF_MODEL_REPO,
        filename=config.CKPT_GRAPH.format(slug=slug),
        repo_type="model", token=token, cache_dir="./hf_cache",
    )

    ckpt  = torch.load(io.BytesIO(open(policy_file, "rb").read()),
                       map_location=device, weights_only=False)
    graph = pickle.loads(open(graph_file, "rb").read())

    policy = ActorCritic(
        obs_dim=ckpt["obs_dim"],
        action_dim=ckpt["action_dim"],
    ).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    print(
        f"  Loaded policy: obs_dim={ckpt['obs_dim']}  "
        f"action_dim={ckpt['action_dim']}  "
        f"trained={ckpt.get('train_date','?')}"
    )
    print(
        f"  Causal graph: {graph.n_edges} edges  "
        f"method={graph.method}  "
        f"vars={len(graph.var_names)}"
    )
    return {"policy": policy, "graph": graph, "ckpt": ckpt}


def zscore_cross(arr: np.ndarray) -> np.ndarray:
    mu  = arr.mean()
    std = arr.std() + 1e-8
    return (arr - mu) / std


def run_engine(
    log_returns:      pd.DataFrame,
    macro_df:         pd.DataFrame,
    universe_tickers: list[str],
    universe_name:    str,
    token:            str | None = None,
    device:           torch.device | None = None,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    avail = [t for t in universe_tickers if t in log_returns.columns]
    mac_c = [c for c in config.MACRO_COLS if c in macro_df.columns]

    print(
        f"\n{'='*60}\n"
        f"Universe: {universe_name}  ({len(avail)} ETFs)\n"
        f"Period: {log_returns.index[0].date()} → {log_returns.index[-1].date()}"
        f"  ({len(log_returns)} days)\n"
        f"{'='*60}"
    )

    ckpt_data = _load_checkpoint(universe_name, token, device)
    policy    = ckpt_data["policy"]
    graph_ckpt: CausalGraph = ckpt_data["graph"]

    # Full aligned arrays
    joint_data, var_names, all_dates = data_manager.build_joint_array(
        log_returns, macro_df, avail
    )
    ret_arr   = log_returns[avail].reindex(all_dates).values.astype(np.float32)
    mac_arr   = macro_df[mac_c].reindex(all_dates).values.astype(np.float32)
    dates     = all_dates

    oos_start      = pd.Timestamp(config.OOS_START)
    last_graph_t   = -config.GRAPH_REFIT_FREQ  # force rebuild on first OOS day
    int_model      = None

    # ── Storage ───────────────────────────────────────────────────────────────
    score_records   : list[dict] = []
    weight_records  : list[dict] = []
    ir_records      : list[dict] = []      # interventional returns
    ranking_records : list[dict] = []
    daily_records   : list[dict] = []

    n_scored = 0

    for t in range(config.GRAPH_WINDOW + config.ENV_WINDOW, len(ret_arr)):
        date = dates[t]
        if date < oos_start:
            continue

        # ── Rebuild interventional model if due ───────────────────────────────
        if (t - last_graph_t) >= config.GRAPH_REFIT_FREQ or int_model is None:
            win_data = joint_data[t - config.GRAPH_WINDOW: t]
            graph    = fit_causal_graph(win_data, var_names)
            int_model = InterventionalModel(
                graph=graph,
                data=win_data,
                etf_names=avail,
                macro_names=mac_c,
            )
            last_graph_t = t

        # ── Build causal observation ──────────────────────────────────────────
        mac_dict = {mac_c[k]: float(mac_arr[t, k]) for k in range(len(mac_c))}
        ir       = int_model.interventional_return(mac_dict)
        ir_vec   = np.array([ir.get(e, 0.0) for e in avail], dtype=np.float32)

        win_s    = max(0, t - config.ENV_WINDOW)
        raw_win  = ret_arr[win_s: t]
        if len(raw_win) < config.ENV_WINDOW:
            pad     = np.zeros((config.ENV_WINDOW - len(raw_win), len(avail)),
                               dtype=np.float32)
            raw_win = np.concatenate([pad, raw_win], axis=0)

        # Current weights (equal weight as neutral starting point for inference)
        w_neutral = np.full(len(avail) + 1, 1.0 / (len(avail) + 1), dtype=np.float32)

        obs = np.concatenate([
            ir_vec,
            raw_win.ravel(),
            mac_arr[t],
            w_neutral,
        ])
        obs = np.clip(obs.astype(np.float32), -10.0, 10.0)

        # ── Policy action (deterministic) ─────────────────────────────────────
        with torch.no_grad():
            obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, _ = policy.get_action(obs_t, deterministic=True)
            action_np  = action.squeeze(0).cpu().numpy()

        # Softmax → weights
        action_np -= action_np.max()
        exp_a  = np.exp(action_np)
        weights = (exp_a / exp_a.sum()).astype(np.float32)
        weights[-1] = np.clip(weights[-1], 0.0, config.CASH_WEIGHT_MAX)
        etf_sum = weights[:-1].sum()
        if etf_sum > 1e-8:
            weights[:-1] *= (1.0 - weights[-1]) / etf_sum

        # ── Composite score from weights + interventional signal ──────────────
        # score = policy_weight * interventional_return (directional conviction)
        raw_score   = weights[:-1] * ir_vec
        composite_z = zscore_cross(raw_score)

        # ── Rank ──────────────────────────────────────────────────────────────
        ranked_idx = np.argsort(composite_z)[::-1]
        top_ticker = avail[ranked_idx[0]]
        top_score  = float(composite_z[ranked_idx[0]])
        cash_flag  = top_score < config.CASH_THRESHOLD or weights[-1] > 0.30

        ds = date.strftime("%Y-%m-%d")
        n_scored += 1

        score_records.append({"date": ds,
            **{avail[i]: round(float(composite_z[i]), 6) for i in range(len(avail))}})
        weight_records.append({"date": ds,
            **{avail[i]: round(float(weights[i]), 6) for i in range(len(avail))},
            "CASH": round(float(weights[-1]), 6)})
        ir_records.append({"date": ds,
            **{avail[i]: round(float(ir_vec[i]), 8) for i in range(len(avail))}})
        ranking_records.append({"date": ds,
            **{avail[ranked_idx[r]]: r + 1 for r in range(len(avail))}})
        daily_records.append({
            "date":       ds,
            "top_ticker": "CASH" if cash_flag else top_ticker,
            "top_score":  round(top_score, 6),
            "cash_flag":  cash_flag,
            "cash_wt":    round(float(weights[-1]), 4),
            "mean_ir":    round(float(ir_vec.mean()), 8),
            "n_edges":    int_model.graph.n_edges,
        })

        if n_scored % 252 == 0 or t == len(ret_arr) - 1:
            top5 = [(avail[ranked_idx[r]],
                     round(float(composite_z[ranked_idx[r]]), 3),
                     round(float(weights[ranked_idx[r]]), 3))
                    for r in range(min(5, len(avail)))]
            print(
                f"  {ds} | top5: "
                + "  ".join(f"{tk}(z={sc:+.2f} w={wt:.2f})" for tk, sc, wt in top5)
                + (f" [CASH wt={weights[-1]:.2f}]" if cash_flag else "")
            )

    # ── Latest snapshot ───────────────────────────────────────────────────────
    latest_score   = score_records[-1]
    latest_weight  = weight_records[-1]
    latest_ir      = ir_records[-1]
    latest_ranking = ranking_records[-1]
    latest_date    = daily_records[-1]["date"]

    latest_out: dict[str, dict] = {}
    for i, tkr in enumerate(avail):
        latest_out[tkr] = {
            "composite_score":   latest_score[tkr],
            "policy_weight":     latest_weight[tkr],
            "interventional_ret":latest_ir[tkr],
            "rank":              int(latest_ranking[tkr]),
        }
    latest_out["CASH"] = {"policy_weight": latest_weight.get("CASH", 0.0)}

    latest_ranked = sorted(
        [(t, v) for t, v in latest_out.items() if t != "CASH"],
        key=lambda x: x[1]["composite_score"], reverse=True,
    )

    return {
        "latest_date":   latest_date,
        "latest_scores": latest_out,
        "latest_ranked": latest_ranked,
        "daily_df":      pd.DataFrame(daily_records).set_index("date"),
        "score_df":      pd.DataFrame(score_records).set_index("date"),
        "weight_df":     pd.DataFrame(weight_records).set_index("date"),
        "ir_df":         pd.DataFrame(ir_records).set_index("date"),
        "ranking_df":    pd.DataFrame(ranking_records).set_index("date"),
        "universe":      universe_name,
        "n_etf":         len(avail),
        "ckpt_meta":     ckpt_data["ckpt"],
    }
