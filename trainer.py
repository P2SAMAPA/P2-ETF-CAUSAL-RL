"""trainer.py — Causal RL daily inference orchestrator (dual-module)."""

from __future__ import annotations

import io
import json
import os

import torch
from huggingface_hub import HfApi

import config
import data_manager
from engine import run_engine


def push_module_results(result: dict, universe: str, module: str, token: str) -> None:
    """Push one module's (lingam or pcmci) results to HF."""
    slug = universe.lower().replace("_", "-")
    api  = HfApi(token=token)
    api.create_repo(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset",
                    exist_ok=True, private=False)

    def _push(data: bytes, path: str, msg: str) -> None:
        api.upload_file(path_or_fileobj=io.BytesIO(data), path_in_repo=path,
                        repo_id=config.HF_OUTPUT_REPO, repo_type="dataset",
                        commit_message=msg)

    output = {
        "run_date":      config.TODAY,
        "universe":      universe,
        "module":        module,
        "graph_window":  result.get("graph_window"),
        "latest_date":   result.get("latest_date"),
        "latest_scores": result.get("latest_scores", {}),
        "latest_ranked": [{"ticker": t, **v}
                          for t, v in result.get("latest_ranked", [])],
        "config": {
            "graph_method":     result.get("graph_method"),
            "graph_window":     result.get("graph_window"),
            "cf_penalty_wt":    config.CF_PENALTY_WT,
            "cash_threshold":   config.CASH_THRESHOLD,
            "top_n":            config.TOP_N,
            "oos_start":        config.OOS_START,
        },
    }

    prefix = f"{module}_{slug}"
    _push(json.dumps(output, indent=2, default=str).encode(),
          f"causal_rl_{config.TODAY}_{prefix}.json",
          f"Causal RL [{module}] {config.TODAY} -- {slug}")

    for name, df in [
        ("daily",    result.get("daily_df")),
        ("scores",   result.get("score_df")),
        ("weights",  result.get("weight_df")),
        ("ir",       result.get("ir_df")),
        ("rankings", result.get("ranking_df")),
    ]:
        if df is not None:
            _push(df.to_csv().encode(), f"{name}_{prefix}.csv",
                  f"{name} [{module}] {config.TODAY} -- {slug}")

    print(f"  Pushed [{module}] -> {config.HF_OUTPUT_REPO}/causal_rl_{config.TODAY}_{prefix}.json")


def main() -> None:
    token = config.HF_TOKEN
    if not token:
        print("HF_TOKEN not set -- aborting.")
        return

    target = os.environ.get("CAUSAL_RL_UNIVERSE", "ALL").upper()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_returns, macro_df = data_manager.load_data(token=token)

    for universe_name, tickers in config.UNIVERSES.items():
        if target != "ALL" and universe_name != target:
            continue

        result = run_engine(log_returns=log_returns, macro_df=macro_df,
                            universe_tickers=tickers, universe_name=universe_name,
                            token=token, device=device)

        if result.get("lingam"):
            push_module_results(result["lingam"], universe_name, "lingam", token)
        if result.get("pcmci"):
            push_module_results(result["pcmci"], universe_name, "pcmci", token)

    print("\nCausal RL daily inference complete.")


if __name__ == "__main__":
    main()
