"""trainer.py — Causal RL daily inference orchestrator."""

from __future__ import annotations

import io
import json
import os

import torch
from huggingface_hub import HfApi

import config
import data_manager
from engine import run_engine


def push_results(result: dict, universe: str, token: str) -> None:
    slug = universe.lower().replace("_", "-")
    api  = HfApi(token=token)

    api.create_repo(
        repo_id=config.HF_OUTPUT_REPO,
        repo_type="dataset", exist_ok=True, private=False,
    )

    ckpt = result.get("ckpt_meta", {})
    output = {
        "run_date":      config.TODAY,
        "universe":      universe,
        "latest_date":   result["latest_date"],
        "latest_scores": result["latest_scores"],
        "latest_ranked": [{"ticker": t, **v} for t, v in result["latest_ranked"]],
        "ckpt_meta": {
            "train_date":    ckpt.get("train_date", "?"),
            "best_ep_ret":   ckpt.get("best_ep_ret", None),
            "graph_method":  ckpt.get("config", {}).get("graph_method", "?"),
        },
        "config": {
            "graph_method":     config.GRAPH_METHOD,
            "graph_window":     config.GRAPH_WINDOW,
            "graph_refit_freq": config.GRAPH_REFIT_FREQ,
            "cf_penalty_wt":    config.CF_PENALTY_WT,
            "cf_n_samples":     config.CF_N_SAMPLES,
            "cash_threshold":   config.CASH_THRESHOLD,
            "top_n":            config.TOP_N,
            "oos_start":        config.OOS_START,
        },
    }

    def _push(data: bytes, path: str, msg: str) -> None:
        api.upload_file(
            path_or_fileobj=io.BytesIO(data),
            path_in_repo=path,
            repo_id=config.HF_OUTPUT_REPO,
            repo_type="dataset",
            commit_message=msg,
        )

    _push(json.dumps(output, indent=2, default=str).encode(),
          f"causal_rl_{config.TODAY}_{slug}.json",
          f"Causal RL results {config.TODAY} — {slug}")

    for name, df in [
        ("daily",    result["daily_df"]),
        ("scores",   result["score_df"]),
        ("weights",  result["weight_df"]),
        ("ir",       result["ir_df"]),
        ("rankings", result["ranking_df"]),
    ]:
        _push(df.to_csv().encode(),
              f"{name}_{slug}.csv",
              f"{name} {config.TODAY} — {slug}")

    print(f"  ✅ Pushed → {config.HF_OUTPUT_REPO}/causal_rl_{config.TODAY}_{slug}.json")


def main() -> None:
    token = config.HF_TOKEN
    if not token:
        print("HF_TOKEN not set — aborting.")
        return

    target = os.environ.get("CAUSAL_RL_UNIVERSE", "ALL").upper()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_returns, macro_df = data_manager.load_data(token=token)

    for universe_name, tickers in config.UNIVERSES.items():
        if target != "ALL" and universe_name != target:
            continue
        result = run_engine(
            log_returns=log_returns,
            macro_df=macro_df,
            universe_tickers=tickers,
            universe_name=universe_name,
            token=token,
            device=device,
        )
        push_results(result, universe_name, token)

    print("\n✅ Causal RL daily inference complete.")


if __name__ == "__main__":
    main()
