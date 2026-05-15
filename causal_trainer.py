"""causal_trainer.py — Causal graph discovery + PPO policy training.

Run via causal_train.yml (manual, weekly).

Workflow
--------
1. Load full dataset
2. For each universe:
   a. Fit causal graph on TRAIN_END window
   b. Build InterventionalModel
   c. Create CausalRLEnv (train + val splits)
   d. Train PPO for N_TRAIN_UPDATES iterations
   e. Evaluate on val period → report mean episode return
   f. Save policy checkpoint + causal graph to HF model repo
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import time

import numpy as np
import torch
from huggingface_hub import HfApi

import config
import data_manager
from causal_graph import fit_causal_graph
from do_calculus import InterventionalModel
from policy import PPOTrainer
from rl_env import CausalRLEnv


def train(
    universe_name: str,
    tickers:       list[str],
    log_returns,
    macro_df,
    n_updates:     int,
    device:        torch.device,
    token:         str,
) -> None:
    print(f"\n{'='*60}")
    print(f"Causal RL Training — Universe: {universe_name}")
    print(f"Device: {device} | Updates: {n_updates}")
    print(f"Graph method: {config.GRAPH_METHOD} | RL: PPO")
    print(f"Train: 2008 → {config.TRAIN_END}  |  Val: → {config.VALIDATE_END}")
    print(f"{'='*60}")

    avail = [t for t in tickers if t in log_returns.columns]
    mac_c = [c for c in config.MACRO_COLS if c in macro_df.columns]

    # ── Step 1: Causal graph on full train window ─────────────────────────────
    joint_data, var_names, all_dates = data_manager.build_joint_array(
        log_returns, macro_df, avail
    )
    train_mask = all_dates <= config.TRAIN_END
    train_data = joint_data[train_mask]

    lingam_data = train_data[-config.LINGAM_WINDOW:] if len(train_data) >= config.LINGAM_WINDOW else train_data
    print(f"\nFitting LiNGAM graph on {len(lingam_data)} train days...")
    graph = fit_causal_graph(lingam_data, var_names, method="lingam")
    print(f"LiNGAM graph: {graph.n_edges} edges")

    pcmci_win = (config.PCMCI_WINDOW_COMBINED if universe_name == "COMBINED" else config.PCMCI_WINDOW)
    pcmci_data = train_data[-pcmci_win:] if len(train_data) >= pcmci_win else train_data
    print(f"Fitting PCMCI+ graph on {len(pcmci_data)} train days (window={pcmci_win}d)...")
    graph_pcmci = fit_causal_graph(pcmci_data, var_names, method="pcmci")
    print(f"PCMCI graph: {graph_pcmci.n_edges} edges")

    # ── Step 2: InterventionalModel on train data ─────────────────────────────
    print("Building interventional model (backdoor adjustment lookup table)...")
    int_model = InterventionalModel(
        graph=graph, data=lingam_data,
        etf_names=avail, macro_names=mac_c,
    )
    print("Interventional model ready.")

    # ── Step 3: Arrays for env ────────────────────────────────────────────────
    ret_arr  = log_returns[avail].reindex(all_dates).values.astype(np.float32)
    mac_arr  = macro_df[mac_c].reindex(all_dates).values.astype(np.float32)

    train_idx = np.where(train_mask)[0]
    val_mask  = (all_dates > config.TRAIN_END) & (all_dates <= config.VALIDATE_END)
    val_idx   = np.where(val_mask)[0]

    # ── Step 4: Create environments ───────────────────────────────────────────
    rng_train = np.random.default_rng(42)
    train_env = CausalRLEnv(
        ret_arr=ret_arr[:train_idx[-1] + 1],
        macro_arr=mac_arr[:train_idx[-1] + 1],
        int_model=int_model,
        etf_names=avail,
        macro_names=mac_c,
        mode="train",
        rng=rng_train,
    )

    # Compute obs/action dims from env
    obs_dim    = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]
    print(f"\nEnv: obs_dim={obs_dim}  action_dim={action_dim}")
    print(f"     n_etf={len(avail)}  n_macro={len(mac_c)}")

    # ── Step 5: PPO training ──────────────────────────────────────────────────
    trainer = PPOTrainer(obs_dim=obs_dim, action_dim=action_dim, device=device)
    n_params = sum(p.numel() for p in trainer.policy.parameters())
    print(f"Policy parameters: {n_params:,}")
    print(f"\nTraining PPO for {n_updates} updates...")
    print(f"  Rollout steps/update: {config.PPO_ROLLOUT_STEPS}")
    print(f"  Batch size: {config.PPO_BATCH_SIZE}")
    print(f"  PPO epochs/update: {config.PPO_EPOCHS_PER_UPDATE}")

    history = {
        "actor_loss": [], "critic_loss": [], "entropy_loss": [],
        "mean_ep_ret": [], "update": [],
    }
    best_ep_ret = -np.inf
    best_state  = None
    t0_total    = time.time()

    for update in range(1, n_updates + 1):
        t0 = time.time()

        rollout_info = trainer.collect_rollout(train_env)
        update_info  = trainer.update()

        history["actor_loss"].append(update_info["actor_loss"])
        history["critic_loss"].append(update_info["critic_loss"])
        history["entropy_loss"].append(update_info["entropy_loss"])
        history["mean_ep_ret"].append(rollout_info["mean_ep_ret"])
        history["update"].append(update)

        if rollout_info["mean_ep_ret"] > best_ep_ret:
            best_ep_ret = rollout_info["mean_ep_ret"]
            best_state  = {k: v.cpu().clone()
                           for k, v in trainer.policy.state_dict().items()}

        if update % 20 == 0 or update == n_updates:
            elapsed = time.time() - t0
            print(
                f"  Update {update:4d}/{n_updates} | "
                f"ep_ret={rollout_info['mean_ep_ret']:+.4f}  "
                f"actor={update_info['actor_loss']:.4f}  "
                f"critic={update_info['critic_loss']:.4f}  "
                f"entropy={update_info['entropy_loss']:.4f}  "
                f"[{elapsed:.1f}s]"
            )

    total_time = time.time() - t0_total
    print(f"\nTraining complete in {total_time/60:.1f} min")
    print(f"Best mean episode return: {best_ep_ret:.6f}")

    # ── Step 6: Validation ────────────────────────────────────────────────────
    if len(val_idx) > 0:
        print("\nEvaluating on validation period...")
        trainer.policy.load_state_dict(best_state)
        trainer.policy.eval()

        val_env = CausalRLEnv(
            ret_arr=ret_arr[:val_idx[-1] + 1],
            macro_arr=mac_arr[:val_idx[-1] + 1],
            int_model=int_model,
            etf_names=avail,
            macro_names=mac_c,
            episode_idx=int(val_idx[0]),
            mode="eval",
        )
        val_obs, _ = val_env.reset()
        val_rets   = []
        done       = False

        with torch.no_grad():
            for _ in range(min(len(val_idx), config.MAX_EPISODE_STEPS)):
                obs_t  = torch.tensor(val_obs, dtype=torch.float32,
                                      device=device).unsqueeze(0)
                act, _ = trainer.policy.get_action(obs_t, deterministic=True)
                val_obs, r, terminated, truncated, _ = val_env.step(
                    act.squeeze(0).cpu().numpy()
                )
                val_rets.append(r)
                if terminated or truncated:
                    break

        val_mean = float(np.mean(val_rets)) if val_rets else 0.0
        print(f"Validation mean reward: {val_mean:.6f}")
    else:
        val_mean = 0.0

    # ── Step 7: Save to HuggingFace ───────────────────────────────────────────
    slug = universe_name.lower().replace("_", "-")
    api  = HfApi(token=token)
    api.create_repo(
        repo_id=config.HF_MODEL_REPO,
        repo_type="model", exist_ok=True, private=False,
    )

    # Policy checkpoint
    policy_ckpt = {
        "policy_state_dict": best_state,
        "obs_dim":           obs_dim,
        "action_dim":        action_dim,
        "tickers":           avail,
        "universe":          universe_name,
        "train_date":        config.TODAY,
        "best_ep_ret":       best_ep_ret,
        "val_mean_reward":   val_mean,
        "config": {
            "graph_method":    config.GRAPH_METHOD,
            "ppo_hidden":      config.PPO_HIDDEN,
            "ppo_lr":          config.PPO_LR,
            "n_train_updates": n_updates,
            "rollout_steps":   config.PPO_ROLLOUT_STEPS,
            "cf_penalty_wt":   config.CF_PENALTY_WT,
            "train_end":       config.TRAIN_END,
        },
    }
    buf = io.BytesIO()
    torch.save(policy_ckpt, buf)
    buf.seek(0)

    api.upload_file(
        path_or_fileobj=buf,
        path_in_repo=config.CKPT_POLICY.format(slug=slug),
        repo_id=config.HF_MODEL_REPO, repo_type="model",
        commit_message=f"Causal RL policy {slug} — {config.TODAY}",
    )

    # Causal graph
    # Save LiNGAM graph
    api.upload_file(
        path_or_fileobj=io.BytesIO(pickle.dumps(graph)),
        path_in_repo=config.CKPT_GRAPH_LINGAM.format(slug=slug),
        repo_id=config.HF_MODEL_REPO, repo_type="model",
        commit_message=f"LiNGAM graph {slug} -- {config.TODAY}",
    )
    # Save PCMCI graph
    api.upload_file(
        path_or_fileobj=io.BytesIO(pickle.dumps(graph_pcmci)),
        path_in_repo=config.CKPT_GRAPH_PCMCI.format(slug=slug),
        repo_id=config.HF_MODEL_REPO, repo_type="model",
        commit_message=f"PCMCI graph {slug} -- {config.TODAY}",
    )

    # Metadata
    meta = {
        "universe":        universe_name,
        "train_date":      config.TODAY,
        "best_ep_ret":     best_ep_ret,
        "val_mean_reward": val_mean,
        "n_edges_lingam":     graph.n_edges,
        "n_edges_pcmci":      graph_pcmci.n_edges,
        "graph_method_lingam":graph.method,
        "graph_method_pcmci": graph_pcmci.method,
        "tickers":         avail,
        "obs_dim":         obs_dim,
        "action_dim":      action_dim,
        "history":         history,
    }
    api.upload_file(
        path_or_fileobj=io.BytesIO(json.dumps(meta, indent=2).encode()),
        path_in_repo=config.CKPT_META.format(slug=slug),
        repo_id=config.HF_MODEL_REPO, repo_type="model",
        commit_message=f"Causal RL meta {slug} — {config.TODAY}",
    )

    print(f"\n  ✅ Policy  → {config.HF_MODEL_REPO}/{config.CKPT_POLICY.format(slug=slug)}")
    print(f"  ✅ LiNGAM graph → {config.HF_MODEL_REPO}/{config.CKPT_GRAPH_LINGAM.format(slug=slug)}")
    print(f"  ✅ PCMCI  graph → {config.HF_MODEL_REPO}/{config.CKPT_GRAPH_PCMCI.format(slug=slug)}")
    print(f"  ✅ Meta    → {config.HF_MODEL_REPO}/{config.CKPT_META.format(slug=slug)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal RL Trainer")
    parser.add_argument("--universe", default="ALL")
    parser.add_argument("--updates",  type=int, default=config.N_TRAIN_UPDATES)
    args = parser.parse_args()

    token = config.HF_TOKEN
    if not token:
        print("HF_TOKEN not set — aborting.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    log_returns, macro_df = data_manager.load_data(token=token)

    target = args.universe.upper()
    for universe_name, tickers in config.UNIVERSES.items():
        if target != "ALL" and universe_name != target:
            continue
        train(
            universe_name=universe_name,
            tickers=tickers,
            log_returns=log_returns,
            macro_df=macro_df,
            n_updates=args.updates,
            device=device,
            token=token,
        )

    print("\n✅ Causal RL training complete.")


if __name__ == "__main__":
    main()
