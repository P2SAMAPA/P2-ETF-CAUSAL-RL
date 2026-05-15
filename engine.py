"""engine.py — Causal RL dual-module walk-forward inference (LiNGAM + PCMCI)."""

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


def _load_checkpoint(universe: str, token: str | None, device: torch.device) -> dict:
    slug = universe.lower().replace("_", "-")
    policy_file = hf_hub_download(
        repo_id=config.HF_MODEL_REPO,
        filename=config.CKPT_POLICY.format(slug=slug),
        repo_type="model", token=token, cache_dir="./hf_cache",
    )
    ckpt = torch.load(io.BytesIO(open(policy_file,"rb").read()),
                      map_location=device, weights_only=False)
    policy = ActorCritic(obs_dim=ckpt["obs_dim"], action_dim=ckpt["action_dim"]).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    print(f"  Policy: obs_dim={ckpt['obs_dim']} trained={ckpt.get('train_date','?')}")

    graph_lingam = None
    try:
        gf = hf_hub_download(repo_id=config.HF_MODEL_REPO,
                             filename=config.CKPT_GRAPH_LINGAM.format(slug=slug),
                             repo_type="model", token=token, cache_dir="./hf_cache")
        graph_lingam = pickle.loads(open(gf,"rb").read())
        print(f"  LiNGAM graph: {graph_lingam.n_edges} edges  window={config.LINGAM_WINDOW}d")
    except Exception as e:
        print(f"  LiNGAM graph not in HF ({e}) -- will refit at runtime")

    graph_pcmci = None
    try:
        gf2 = hf_hub_download(repo_id=config.HF_MODEL_REPO,
                              filename=config.CKPT_GRAPH_PCMCI.format(slug=slug),
                              repo_type="model", token=token, cache_dir="./hf_cache")
        graph_pcmci = pickle.loads(open(gf2,"rb").read())
        pcmci_win = (config.PCMCI_WINDOW_COMBINED if universe == "COMBINED"
                     else config.PCMCI_WINDOW)
        print(f"  PCMCI  graph: {graph_pcmci.n_edges} edges  window={pcmci_win}d")
    except Exception as e:
        print(f"  PCMCI graph not in HF ({e}) -- will refit at runtime")

    return {"policy": policy, "graph_lingam": graph_lingam,
            "graph_pcmci": graph_pcmci, "ckpt": ckpt}


def zscore_cross(arr: np.ndarray) -> np.ndarray:
    mu  = arr.mean()
    std = arr.std() + 1e-8
    return (arr - mu) / std


def _run_single_module(
    ret_arr, mac_arr, mac_c, avail, joint_data, var_names, dates,
    policy, device, graph_method, graph_window, graph_refit_freq,
    oos_start, prefit_graph=None,
) -> dict:
    """Walk-forward inference for one causal module (LiNGAM or PCMCI)."""
    last_graph_t  = -graph_refit_freq
    int_model     = None
    current_graph = prefit_graph

    score_records, weight_records, ir_records = [], [], []
    ranking_records, daily_records = [], []
    n_scored = 0

    for t in range(graph_window + config.ENV_WINDOW, len(ret_arr)):
        date = dates[t]
        if date < oos_start:
            continue

        # Refit causal graph if due
        if (t - last_graph_t) >= graph_refit_freq or int_model is None:
            win_data      = joint_data[t - graph_window : t]
            current_graph = fit_causal_graph(win_data, var_names, method=graph_method)
            int_model     = InterventionalModel(
                graph=current_graph, data=win_data,
                etf_names=avail, macro_names=mac_c,
            )
            last_graph_t = t

        # Build causal observation
        mac_dict = {mac_c[k]: float(mac_arr[t, k]) for k in range(len(mac_c))}
        ir       = int_model.interventional_return(mac_dict)
        ir_vec   = np.array([ir.get(e, 0.0) for e in avail], dtype=np.float32)

        win_s   = max(0, t - config.ENV_WINDOW)
        raw_win = ret_arr[win_s : t]
        if len(raw_win) < config.ENV_WINDOW:
            pad     = np.zeros((config.ENV_WINDOW - len(raw_win), len(avail)), dtype=np.float32)
            raw_win = np.concatenate([pad, raw_win], axis=0)

        w_neutral = np.full(len(avail) + 1, 1.0 / (len(avail) + 1), dtype=np.float32)
        obs = np.clip(
            np.concatenate([ir_vec, raw_win.ravel(), mac_arr[t], w_neutral]).astype(np.float32),
            -10.0, 10.0,
        )

        # Policy action
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, _ = policy.get_action(obs_t, deterministic=True)
            action_np = action.squeeze(0).cpu().numpy()

        action_np -= action_np.max()
        exp_a   = np.exp(action_np)
        weights = (exp_a / exp_a.sum()).astype(np.float32)
        weights[-1] = np.clip(weights[-1], 0.0, config.CASH_WEIGHT_MAX)
        etf_sum = weights[:-1].sum()
        if etf_sum > 1e-8:
            weights[:-1] *= (1.0 - weights[-1]) / etf_sum

        raw_score   = weights[:-1] * ir_vec
        composite_z = zscore_cross(raw_score)
        ranked_idx  = np.argsort(composite_z)[::-1]
        top_ticker  = avail[ranked_idx[0]]
        top_score   = float(composite_z[ranked_idx[0]])
        cash_flag   = top_score < config.CASH_THRESHOLD or weights[-1] > 0.30

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
            "date": ds, "top_ticker": "CASH" if cash_flag else top_ticker,
            "top_score": round(top_score, 6), "cash_flag": cash_flag,
            "cash_wt": round(float(weights[-1]), 4),
            "mean_ir": round(float(ir_vec.mean()), 8),
            "n_edges": current_graph.n_edges if current_graph else 0,
            "graph_method": graph_method,
        })

        if n_scored % 252 == 0 or t == len(ret_arr) - 1:
            top5 = [(avail[ranked_idx[r]],
                     round(float(composite_z[ranked_idx[r]]), 3),
                     round(float(weights[ranked_idx[r]]), 3))
                    for r in range(min(5, len(avail)))]
            print(f"  [{graph_method.upper()} {graph_window}d] {ds} | "
                  + "  ".join(f"{tk}(z={sc:+.2f} w={wt:.2f})" for tk, sc, wt in top5)
                  + (f" [CASH {weights[-1]:.2f}]" if cash_flag else ""))

    if not daily_records:
        return {}

    latest_score   = score_records[-1]
    latest_weight  = weight_records[-1]
    latest_ir      = ir_records[-1]
    latest_ranking = ranking_records[-1]
    latest_date    = daily_records[-1]["date"]

    latest_out: dict = {}
    for i, tkr in enumerate(avail):
        latest_out[tkr] = {
            "composite_score":    latest_score[tkr],
            "policy_weight":      latest_weight[tkr],
            "interventional_ret": latest_ir[tkr],
            "rank":               int(latest_ranking[tkr]),
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
        "n_scored":      n_scored,
        "graph_method":  graph_method,
        "graph_window":  graph_window,
    }


def run_engine(
    log_returns:      pd.DataFrame,
    macro_df:         pd.DataFrame,
    universe_tickers: list[str],
    universe_name:    str,
    token:            str | None = None,
    device:           torch.device | None = None,
) -> dict:
    """Run both LiNGAM and PCMCI causal modules for one universe."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    avail = [t for t in universe_tickers if t in log_returns.columns]
    mac_c = [c for c in config.MACRO_COLS if c in macro_df.columns]

    print(f"\n{'='*60}\nUniverse: {universe_name}  ({len(avail)} ETFs)\n"
          f"Period: {log_returns.index[0].date()} -> {log_returns.index[-1].date()}"
          f"  ({len(log_returns)} days)\n{'='*60}")

    ckpt_data    = _load_checkpoint(universe_name, token, device)
    policy       = ckpt_data["policy"]
    graph_lingam = ckpt_data["graph_lingam"]
    graph_pcmci  = ckpt_data["graph_pcmci"]

    joint_data, var_names, all_dates = data_manager.build_joint_array(
        log_returns, macro_df, avail
    )
    ret_arr   = log_returns[avail].reindex(all_dates).values.astype(np.float32)
    mac_arr   = macro_df[mac_c].reindex(all_dates).values.astype(np.float32)
    oos_start = pd.Timestamp(config.OOS_START)
    pcmci_win = (config.PCMCI_WINDOW_COMBINED if universe_name == "COMBINED"
                 else config.PCMCI_WINDOW)

    # Module A: LiNGAM
    print(f"\nModule A -- LiNGAM ({config.LINGAM_WINDOW}d, refit every {config.LINGAM_REFIT_FREQ}d):")
    result_lingam = _run_single_module(
        ret_arr=ret_arr, mac_arr=mac_arr, mac_c=mac_c, avail=avail,
        joint_data=joint_data, var_names=var_names, dates=all_dates,
        policy=policy, device=device, graph_method="lingam",
        graph_window=config.LINGAM_WINDOW, graph_refit_freq=config.LINGAM_REFIT_FREQ,
        oos_start=oos_start, prefit_graph=graph_lingam,
    )

    # Module B: PCMCI
    print(f"\nModule B -- PCMCI+ ({pcmci_win}d, refit every {config.PCMCI_REFIT_FREQ}d):")
    result_pcmci = _run_single_module(
        ret_arr=ret_arr, mac_arr=mac_arr, mac_c=mac_c, avail=avail,
        joint_data=joint_data, var_names=var_names, dates=all_dates,
        policy=policy, device=device, graph_method="pcmci",
        graph_window=pcmci_win, graph_refit_freq=config.PCMCI_REFIT_FREQ,
        oos_start=oos_start, prefit_graph=graph_pcmci,
    )

    return {
        "lingam":    result_lingam,
        "pcmci":     result_pcmci,
        "universe":  universe_name,
        "n_etf":     len(avail),
        "ckpt_meta": ckpt_data["ckpt"],
    }
