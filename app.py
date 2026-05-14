"""app.py — Causal RL Engine · Streamlit Dashboard."""

from __future__ import annotations

import os
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

import config
from us_calendar import next_trading_day

st.set_page_config(
    page_title="Causal RL · P2Quant",
    layout="wide",
    page_icon="🧬",
)

HF_TOKEN = os.environ.get("HF_TOKEN")
BASE_RAW = f"https://huggingface.co/datasets/{config.HF_OUTPUT_REPO}/resolve/main"
BASE_API = f"https://huggingface.co/api/datasets/{config.HF_OUTPUT_REPO}/tree/main"
HEADERS  = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

PALETTE = [
    "#1B4F8A", "#27AE60", "#E74C3C", "#F39C12", "#8E44AD", "#148F77",
    "#CA6F1E", "#2471A3", "#CB4335", "#1A5276", "#117A65", "#B7950B",
    "#884EA0", "#1F618D", "#B9770E", "#922B21", "#1A5276", "#117A65",
]

def score_colour(v: float) -> str:
    if v >= 0.5:  return "#1D9E75"
    if v >= 0.0:  return "#82C3A9"
    if v >= -0.5: return "#F0A07A"
    return "#E74C3C"

def fmt(v: float, d: int = 4) -> str:
    return f"{v:+.{d}f}"


@st.cache_data(ttl=3600, show_spinner="Loading Causal RL results…")
def load_json(universe: str) -> dict | None:
    slug = universe.lower().replace("_", "-")
    try:
        r = requests.get(BASE_API, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return None
        files   = sorted(f["path"] for f in r.json() if f["path"].endswith(".json"))
        matches = [f for f in files if f"_{slug}.json" in f]
        if not matches:
            return None
        resp = requests.get(f"{BASE_RAW}/{matches[-1]}", headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner="Loading history…")
def load_csv(filename: str) -> pd.DataFrame | None:
    try:
        r = requests.get(f"{BASE_RAW}/{filename}", headers=HEADERS, timeout=60)
        if r.status_code != 200:
            return None
        df = pd.read_csv(StringIO(r.text), index_col=0, parse_dates=True)
        return df if not df.empty else None
    except Exception:
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    universe = st.selectbox("Universe", list(config.UNIVERSES.keys()))
    st.divider()
    st.markdown(f"**Graph method:** {config.GRAPH_METHOD}")
    st.markdown(f"**Graph window:** {config.GRAPH_WINDOW}d")
    st.markdown(f"**Graph refit every:** {config.GRAPH_REFIT_FREQ}d")
    st.markdown(f"**CF penalty weight:** {config.CF_PENALTY_WT}")
    st.markdown(f"**CF samples/step:** {config.CF_N_SAMPLES}")
    st.markdown(f"**CASH max wt:** {config.CASH_WEIGHT_MAX}")
    st.markdown(f"**OOS from:** {config.OOS_START}")
    st.markdown(f"**Next trading day:** {next_trading_day()}")
    st.divider()
    st.markdown("**Workflows:**")
    st.markdown("🧬 `causal_train.yml` — manual weekly training")
    st.markdown("📅 `daily_run.yml` — daily auto inference")
    st.divider()
    st.markdown("**Score formula:**")
    st.code(
        "score = policy_weight × interventional_return\n"
        "      = PPO(do-calculus state) × E[r|do(macro)]",
        language="python",
    )
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🧬 Causal RL Engine — Do-Calculus + PPO")
st.caption(
    f"Causal graph ({config.GRAPH_METHOD}) → backdoor adjustment → "
    f"E[r|do(macro)] interventional returns → PPO policy → "
    f"counterfactual reward shaping (CF penalty={config.CF_PENALTY_WT})"
)

slug       = universe.lower().replace("_", "-")
data       = load_json(universe)
daily_df   = load_csv(f"daily_{slug}.csv")
score_df   = load_csv(f"scores_{slug}.csv")
weight_df  = load_csv(f"weights_{slug}.csv")
ir_df      = load_csv(f"ir_{slug}.csv")
ranking_df = load_csv(f"rankings_{slug}.csv")

if data is None:
    st.warning(
        "⚠️ No results found. Run `causal_train.yml` first, "
        "then `daily_run.yml` to generate scores."
    )
    st.stop()

latest_scores = data.get("latest_scores", {})
latest_ranked = data.get("latest_ranked", [])
latest_date   = data.get("latest_date", "?")
run_date      = data.get("run_date", "?")
ckpt_meta     = data.get("ckpt_meta", {})
cfg           = data.get("config", {})

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Run Date",      run_date)
k2.metric("Latest Date",   latest_date)
k3.metric("Model Trained", ckpt_meta.get("train_date", "?"))
k4.metric("Graph Method",  ckpt_meta.get("graph_method", cfg.get("graph_method","?")))

if latest_ranked:
    top       = latest_ranked[0]
    cash_flag = (top.get("composite_score", 0) < config.CASH_THRESHOLD
                 or latest_scores.get("CASH", {}).get("policy_weight", 0) > 0.30)
    cash_wt   = latest_scores.get("CASH", {}).get("policy_weight", 0.0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🏆 Top Pick",       "CASH" if cash_flag else top["ticker"])
    m2.metric("Top Score",         fmt(top.get("composite_score", 0)))
    m3.metric("CASH Policy Weight",f"{cash_wt:.2%}")
    m4.metric("CASH Signal",       "Yes ⚠️" if cash_flag else "No ✅")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Rankings & Scores",
    "🧬 Interventional Returns",
    "⚖️ Policy Weights",
    "📈 Score History",
    "📋 Full Table",
])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Rankings & Scores
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader(f"Causal RL Rankings as of {latest_date}")

    tickers_r = [r["ticker"] for r in latest_ranked]
    scores_r  = [r.get("composite_score", 0)    for r in latest_ranked]
    ir_r      = [r.get("interventional_ret", 0) for r in latest_ranked]
    wt_r      = [r.get("policy_weight", 0)      for r in latest_ranked]
    colours_r = [score_colour(s) for s in scores_r]

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Composite Score = Policy Weight × Interventional Return**")
        fig = go.Figure(go.Bar(
            y=tickers_r, x=scores_r, orientation="h",
            marker_color=colours_r,
            text=[fmt(s) for s in scores_r],
            textposition="outside",
        ))
        fig.add_vline(x=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title="Causal RL score: PPO allocation × E[r|do(macro)]",
            xaxis_title="Composite z-score",
            yaxis=dict(autorange="reversed"),
            height=max(300, len(tickers_r) * 30),
            margin=dict(t=50, b=20, l=60, r=80),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, key="rank_bar")

    with col_r:
        st.markdown("**Policy Weight vs Interventional Return (scatter)**")
        fig2 = go.Figure(go.Scatter(
            x=ir_r, y=wt_r,
            mode="markers+text",
            text=tickers_r,
            textposition="top center",
            marker=dict(
                size=12, color=scores_r,
                colorscale="RdYlGn", showscale=True,
                colorbar=dict(title="Score"),
            ),
        ))
        fig2.add_vline(x=0, line_dash="dot", line_color="gray")
        fig2.update_layout(
            title="PPO weight vs E[r|do(macro)] — top-right = high conviction causal long",
            xaxis_title="Interventional return E[r|do(macro)]",
            yaxis_title="Policy weight",
            height=max(300, len(tickers_r) * 30),
            margin=dict(t=50, b=40, l=60, r=80),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True, key="ir_wt_scatter")

    # Top-N cards
    st.markdown(f"### 🎯 Top {config.TOP_N} for {next_trading_day()}")
    cols = st.columns(config.TOP_N)
    for i, row in enumerate(latest_ranked[: config.TOP_N]):
        with cols[i]:
            sc  = row.get("composite_score", 0)
            ir  = row.get("interventional_ret", 0)
            wt  = row.get("policy_weight", 0)
            bg  = score_colour(sc)
            st.markdown(
                f"**#{i+1} {row['ticker']}**\n\n"
                f"Score: `{fmt(sc)}`\n\n"
                f"E[r|do(·)]: `{fmt(ir, 6)}`\n\n"
                f"PPO wt: `{wt:.3f}`\n\n"
                f'<span style="background:{bg};color:white;padding:2px 8px;'
                f'border-radius:8px;font-size:11px">Rank #{row.get("rank",i+1)}</span>',
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Interventional Returns
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Interventional Returns E[r | do(macro)] Over Time")
    st.caption(
        "E[r | do(macro=x)] is the causal expected return under a forced macro "
        "intervention — NOT the observed return. It strips out spurious correlations "
        "via backdoor adjustment. **Positive = causally bullish given today's macro.**"
    )

    if ir_df is not None:
        etf_cols = [c for c in ir_df.columns if c in config.UNIVERSES[universe]]
        selected = st.multiselect("Select ETFs", etf_cols, default=etf_cols[:6], key="ir_sel")
        period   = st.radio("Period", ["Last 2 years", "Last 5 years", "Full OOS"],
                            horizontal=True, key="ir_period")
        df_ir = ir_df.copy()
        if period == "Last 2 years":
            df_ir = df_ir[df_ir.index >= "2024-01-01"]
        elif period == "Last 5 years":
            df_ir = df_ir[df_ir.index >= "2021-01-01"]

        if selected:
            fig_ir = go.Figure()
            for i, tkr in enumerate(selected):
                if tkr in df_ir.columns:
                    fig_ir.add_trace(go.Scatter(
                        x=df_ir.index, y=df_ir[tkr], mode="lines", name=tkr,
                        line=dict(width=1.4, color=PALETTE[i % len(PALETTE)]),
                    ))
            fig_ir.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_ir.update_layout(
                title="E[r|do(macro)] — causal interventional return per ETF",
                yaxis_title="Interventional return",
                height=420,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_ir, use_container_width=True, key="ir_ts")

        # IR heatmap
        recent_ir = ir_df[[c for c in ir_df.columns
                           if c in config.UNIVERSES[universe]]].tail(126)
        fig_irh = go.Figure(go.Heatmap(
            z=recent_ir.values.T,
            x=recent_ir.index.strftime("%Y-%m-%d"),
            y=list(recent_ir.columns),
            colorscale="RdYlGn", zmid=0,
            colorbar=dict(title="E[r|do]"),
        ))
        fig_irh.update_layout(
            title="Interventional Return Heatmap — last 126 days",
            height=max(300, len(recent_ir.columns) * 22 + 80),
            margin=dict(t=40, b=60, l=60, r=20),
            xaxis=dict(tickangle=-45, nticks=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_irh, use_container_width=True, key="ir_heat")
    else:
        st.info("No interventional return history found.")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Policy Weights
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("PPO Policy Weights Over Time")
    st.caption(
        "Portfolio weights output by the PPO policy. The policy was trained "
        "on causal (interventional) state and penalised for exploiting "
        "spurious correlations via counterfactual reward shaping."
    )

    if weight_df is not None:
        etf_cols_w = [c for c in weight_df.columns if c in config.UNIVERSES[universe]]
        cash_col   = "CASH" if "CASH" in weight_df.columns else None

        # Stacked area chart of weights
        period_w = st.radio("Period", ["Last 2 years", "Last 5 years", "Full OOS"],
                             horizontal=True, key="wt_period")
        df_wt = weight_df.copy()
        if period_w == "Last 2 years":
            df_wt = df_wt[df_wt.index >= "2024-01-01"]
        elif period_w == "Last 5 years":
            df_wt = df_wt[df_wt.index >= "2021-01-01"]

        show_cols = etf_cols_w[:8]
        if cash_col:
            show_cols = show_cols + [cash_col]

        fig_wt = go.Figure()
        for i, col in enumerate(show_cols):
            if col in df_wt.columns:
                color = "#888888" if col == "CASH" else PALETTE[i % len(PALETTE)]
                fig_wt.add_trace(go.Scatter(
                    x=df_wt.index, y=df_wt[col],
                    mode="lines", name=col,
                    line=dict(width=1.3, color=color),
                    stackgroup="one",
                    fillcolor=color.replace(")", ",0.6)").replace("rgb", "rgba")
                    if color.startswith("rgb") else color,
                ))
        fig_wt.update_layout(
            title="PPO portfolio weights over time (stacked)",
            yaxis_title="Weight",
            yaxis=dict(range=[0, 1]),
            height=380,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_wt, use_container_width=True, key="wt_stack")

        # CASH weight over time
        if cash_col and cash_col in weight_df.columns:
            st.markdown("**CASH weight — counterfactual uncertainty signal**")
            fig_cash = go.Figure(go.Scatter(
                x=weight_df.index, y=weight_df[cash_col],
                mode="lines", line=dict(color="#888888", width=1.5),
                fill="tozeroy", fillcolor="rgba(136,136,136,0.1)",
                name="CASH weight",
            ))
            fig_cash.add_hline(y=0.30, line_dash="dash", line_color="#E74C3C",
                               annotation_text="CASH signal threshold (30%)")
            fig_cash.update_layout(
                title="CASH policy weight — spikes = high counterfactual regret",
                yaxis_title="CASH weight",
                yaxis=dict(range=[0, config.CASH_WEIGHT_MAX + 0.05]),
                height=300,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_cash, use_container_width=True, key="cash_wt")
    else:
        st.info("No weight history found.")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Score History
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Composite Score History")

    if score_df is not None:
        etf_cols_s = [c for c in score_df.columns if c in config.UNIVERSES[universe]]
        selected_s = st.multiselect("Select ETFs", etf_cols_s,
                                    default=etf_cols_s[:6], key="score_sel")
        period_s = st.radio("Period", ["Last 2 years", "Last 5 years", "Full OOS"],
                            horizontal=True, key="score_period")
        df_sc = score_df.copy()
        if period_s == "Last 2 years":
            df_sc = df_sc[df_sc.index >= "2024-01-01"]
        elif period_s == "Last 5 years":
            df_sc = df_sc[df_sc.index >= "2021-01-01"]

        if selected_s:
            fig_sc = go.Figure()
            for i, tkr in enumerate(selected_s):
                if tkr in df_sc.columns:
                    fig_sc.add_trace(go.Scatter(
                        x=df_sc.index, y=df_sc[tkr], mode="lines", name=tkr,
                        line=dict(width=1.4, color=PALETTE[i % len(PALETTE)]),
                    ))
            fig_sc.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_sc.update_layout(
                title="Causal RL composite score (z-scored cross-sectionally)",
                yaxis_title="Score", height=400,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_sc, use_container_width=True, key="score_ts")

        # Score heatmap
        recent_sc = score_df[etf_cols_s].tail(252)
        fig_sch = go.Figure(go.Heatmap(
            z=recent_sc.values.T,
            x=recent_sc.index.strftime("%Y-%m-%d"),
            y=list(recent_sc.columns),
            colorscale="RdYlGn", zmid=0,
            colorbar=dict(title="Score"),
        ))
        fig_sch.update_layout(
            title="Score Heatmap — last 252 days",
            height=max(300, len(recent_sc.columns) * 22 + 80),
            margin=dict(t=40, b=60, l=60, r=20),
            xaxis=dict(tickangle=-45, nticks=12),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_sch, use_container_width=True, key="score_heat")

        # Top-pick frequency
        if daily_df is not None and "top_ticker" in daily_df.columns:
            picks = daily_df["top_ticker"].value_counts()
            fig_freq = go.Figure(go.Bar(
                x=picks.index, y=picks.values,
                marker_color="#1B4F8A",
                text=picks.values, textposition="outside",
            ))
            fig_freq.update_layout(
                title="Top-Pick Frequency (full OOS)",
                yaxis_title="Days as #1 causal RL pick", height=280,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_freq, use_container_width=True, key="pick_freq")
    else:
        st.info("No score history found.")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Full Table
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader(f"Full Causal RL Table — {latest_date}")

    if latest_ranked:
        rows = []
        for i, row in enumerate(latest_ranked):
            rows.append({
                "Rank":               i + 1,
                "Ticker":             row["ticker"],
                "Composite Score":    fmt(row.get("composite_score", 0)),
                "Policy Weight":      f"{row.get('policy_weight', 0):.4f}",
                "E[r|do(macro)]":     fmt(row.get("interventional_ret", 0), 6),
            })
        # Add CASH row
        cash_wt = latest_scores.get("CASH", {}).get("policy_weight", 0.0)
        rows.append({
            "Rank": "-", "Ticker": "CASH",
            "Composite Score": "—",
            "Policy Weight": f"{cash_wt:.4f}",
            "E[r|do(macro)]": "—",
        })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True, height=600)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Model Checkpoint Info**")
        st.json(ckpt_meta)
    with c2:
        st.markdown("**Engine Configuration**")
        st.json(cfg)

    if daily_df is not None:
        st.divider()
        st.markdown("**Daily summary (last 20 days)**")
        st.dataframe(daily_df.tail(20), use_container_width=True)

    st.divider()
    st.caption(
        f"P2Quant Causal RL Engine · Run: {run_date} · "
        f"Do-Calculus ({config.GRAPH_METHOD}) + PPO · "
        f"Counterfactual reward shaping · Data: {config.HF_DATA_REPO}"
    )
