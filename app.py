"""app.py — Causal RL Dual-Module Dashboard (LiNGAM tab + PCMCI tab)."""

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

st.set_page_config(page_title="Causal RL · P2Quant", layout="wide", page_icon="🧬")

HF_TOKEN = os.environ.get("HF_TOKEN")
BASE_RAW = f"https://huggingface.co/datasets/{config.HF_OUTPUT_REPO}/resolve/main"
BASE_API = f"https://huggingface.co/api/datasets/{config.HF_OUTPUT_REPO}/tree/main"
HEADERS  = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

PALETTE = ["#1B4F8A","#27AE60","#E74C3C","#F39C12","#8E44AD","#148F77",
           "#CA6F1E","#2471A3","#CB4335","#1A5276","#117A65","#B7950B",
           "#884EA0","#1F618D","#B9770E","#922B21","#1A5276","#117A65"]

def score_colour(v):
    if v >= 0.5:  return "#1D9E75"
    if v >= 0.0:  return "#82C3A9"
    if v >= -0.5: return "#F0A07A"
    return "#E74C3C"

def fmt(v, d=4): return f"{v:+.{d}f}"


@st.cache_data(ttl=3600, show_spinner="Loading results...")
def load_json(universe: str, module: str) -> dict | None:
    slug   = universe.lower().replace("_","-")
    prefix = f"{module}_{slug}"
    try:
        r = requests.get(BASE_API, headers=HEADERS, timeout=30)
        if r.status_code != 200: return None
        files   = sorted(f["path"] for f in r.json() if f["path"].endswith(".json"))
        matches = [f for f in files if f"_{prefix}.json" in f]
        if not matches: return None
        resp = requests.get(f"{BASE_RAW}/{matches[-1]}", headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner="Loading history...")
def load_csv(filename: str) -> pd.DataFrame | None:
    try:
        r = requests.get(f"{BASE_RAW}/{filename}", headers=HEADERS, timeout=60)
        if r.status_code != 200: return None
        df = pd.read_csv(StringIO(r.text), index_col=0, parse_dates=True)
        return df if not df.empty else None
    except Exception:
        return None


def load_module(universe: str, module: str) -> dict:
    slug   = universe.lower().replace("_","-")
    prefix = f"{module}_{slug}"
    return {
        "data":       load_json(universe, module),
        "daily_df":   load_csv(f"daily_{prefix}.csv"),
        "score_df":   load_csv(f"scores_{prefix}.csv"),
        "weight_df":  load_csv(f"weights_{prefix}.csv"),
        "ir_df":      load_csv(f"ir_{prefix}.csv"),
        "ranking_df": load_csv(f"rankings_{prefix}.csv"),
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")
    universe = st.selectbox("Universe", list(config.UNIVERSES.keys()))
    st.divider()
    st.markdown(f"**LiNGAM window:** {config.LINGAM_WINDOW}d")
    st.markdown(f"**LiNGAM refit:** every {config.LINGAM_REFIT_FREQ}d")
    pcmci_w = (config.PCMCI_WINDOW_COMBINED if universe == "COMBINED"
               else config.PCMCI_WINDOW)
    st.markdown(f"**PCMCI+ window:** {pcmci_w}d")
    st.markdown(f"**PCMCI+ refit:** every {config.PCMCI_REFIT_FREQ}d")
    st.markdown(f"**CF penalty:** {config.CF_PENALTY_WT}")
    st.markdown(f"**OOS from:** {config.OOS_START}")
    st.markdown(f"**Next trading day:** {next_trading_day()}")
    st.divider()
    st.markdown("**When to trust which module:**")
    st.markdown("- **LiNGAM** — same-day macro effects, non-Gaussian tails, fast signals")
    st.markdown("- **PCMCI+** — lagged causality, multi-day transmission, slower regimes")
    st.markdown("- **Agreement** between both = highest conviction")
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🧬 Causal RL Engine — Dual Causal Module")
st.caption(
    "LiNGAM (252d) · instantaneous causality · non-Gaussian  |  "
    f"PCMCI+ ({pcmci_w}d) · lagged causality · conditional independence  |  "
    "Same PPO policy · different causal state"
)

lingam_mod = load_module(universe, "lingam")
pcmci_mod  = load_module(universe, "pcmci")


def _build_consensus(lingam_data: dict | None, pcmci_data: dict | None,
                     universe: str) -> pd.DataFrame:
    """Build consensus table comparing both modules side by side."""
    tickers = config.UNIVERSES.get(universe, [])
    rows = []
    for tkr in tickers:
        l_score = (lingam_data.get("latest_scores",{}).get(tkr,{})
                   .get("composite_score", None)) if lingam_data else None
        p_score = (pcmci_data.get("latest_scores",{}).get(tkr,{})
                   .get("composite_score", None)) if pcmci_data else None
        if l_score is None and p_score is None:
            continue
        agree = (l_score is not None and p_score is not None
                 and l_score > 0 and p_score > 0)
        rows.append({
            "Ticker":       tkr,
            "LiNGAM Score": round(l_score, 4) if l_score is not None else "—",
            "PCMCI Score":  round(p_score, 4) if p_score is not None else "—",
            "Agreement":    "✅ Both positive" if agree else (
                            "⚠️ Disagree" if (l_score is not None and p_score is not None) else "—"),
            "Avg Score":    round(((l_score or 0) + (p_score or 0)) / 2, 4),
        })
    return pd.DataFrame(rows).sort_values("Avg Score", ascending=False).reset_index(drop=True)


# ── Top-level comparison ──────────────────────────────────────────────────────
lingam_data = lingam_mod["data"]
pcmci_data  = pcmci_mod["data"]

if lingam_data or pcmci_data:
    l_top = (lingam_data.get("latest_ranked",[{}])[0].get("ticker","?")
             if lingam_data and lingam_data.get("latest_ranked") else "?")
    p_top = (pcmci_data.get("latest_ranked",[{}])[0].get("ticker","?")
             if pcmci_data and pcmci_data.get("latest_ranked") else "?")
    l_date = lingam_data.get("latest_date","?") if lingam_data else "?"
    p_date = pcmci_data.get("latest_date","?") if pcmci_data else "?"

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("LiNGAM top pick", l_top, help=f"As of {l_date}")
    k2.metric("PCMCI+ top pick", p_top, help=f"As of {p_date}")
    k3.metric("Agreement", "✅ YES" if l_top == p_top else "⚠️ NO",
              help="Both modules agree on #1 ETF")
    k4.metric("Next trading day", next_trading_day())

    st.divider()

# ── Consensus table ───────────────────────────────────────────────────────────
with st.expander("📊 Consensus Table — Both Modules Side by Side", expanded=True):
    consensus_df = _build_consensus(lingam_data, pcmci_data, universe)
    if not consensus_df.empty:
        st.dataframe(consensus_df, use_container_width=True, hide_index=True)
        agree_count = (consensus_df["Agreement"] == "✅ Both positive").sum()
        st.caption(
            f"{agree_count} / {len(consensus_df)} ETFs have positive scores "
            f"in BOTH modules — these are the highest conviction causal longs."
        )
    else:
        st.info("Run both modules to see consensus.")

st.divider()


# ── Module rendering helper ───────────────────────────────────────────────────
def render_module_tabs(mod: dict, module_name: str, graph_window: int,
                       universe: str, key_prefix: str) -> None:
    data       = mod["data"]
    daily_df   = mod["daily_df"]
    score_df   = mod["score_df"]
    weight_df  = mod["weight_df"]
    ir_df      = mod["ir_df"]

    if data is None:
        st.warning(f"No {module_name} results found. Run `causal_train.yml` first.")
        return

    latest_scores = data.get("latest_scores", {})
    latest_ranked = data.get("latest_ranked", [])
    latest_date   = data.get("latest_date", "?")
    cfg           = data.get("config", {})

    tickers_r = [r["ticker"] for r in latest_ranked]
    scores_r  = [r.get("composite_score",0) for r in latest_ranked]
    ir_r      = [r.get("interventional_ret",0) for r in latest_ranked]
    wt_r      = [r.get("policy_weight",0) for r in latest_ranked]

    # ── Rankings & Scores ─────────────────────────────────────────────────────
    st.subheader(f"Rankings as of {latest_date}")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Composite Score**")
        fig = go.Figure(go.Bar(
            y=tickers_r, x=scores_r, orientation="h",
            marker_color=[score_colour(s) for s in scores_r],
            text=[fmt(s) for s in scores_r], textposition="outside",
        ))
        fig.add_vline(x=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title=f"{module_name} score: PPO wt x E[r|do(macro)]",
            xaxis_title="Composite z-score",
            yaxis=dict(autorange="reversed"),
            height=max(300, len(tickers_r)*30),
            margin=dict(t=50,b=20,l=60,r=80),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_rank")

    with col_r:
        st.markdown("**Policy Weight vs E[r|do(macro)] Scatter**")
        fig2 = go.Figure(go.Scatter(
            x=ir_r, y=wt_r, mode="markers+text",
            text=tickers_r, textposition="top center",
            marker=dict(size=12, color=scores_r, colorscale="RdYlGn",
                        showscale=True, colorbar=dict(title="Score")),
        ))
        fig2.add_vline(x=0, line_dash="dot", line_color="gray")
        fig2.update_layout(
            title="Top-right = high conviction causal long",
            xaxis_title="Interventional return E[r|do(macro)]",
            yaxis_title="Policy weight",
            height=max(300, len(tickers_r)*30),
            margin=dict(t=50,b=40,l=60,r=80),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True, key=f"{key_prefix}_scatter")

    # Top-N cards
    st.markdown(f"### Top {config.TOP_N} for {next_trading_day()}")
    cols = st.columns(config.TOP_N)
    for i, row in enumerate(latest_ranked[:config.TOP_N]):
        with cols[i]:
            sc = row.get("composite_score",0)
            st.markdown(
                f"**#{i+1} {row['ticker']}**\n\n"
                f"Score: `{fmt(sc)}`\n\n"
                f"E[r|do]: `{fmt(row.get('interventional_ret',0),6)}`\n\n"
                f"PPO wt: `{row.get('policy_weight',0):.3f}`\n\n"
                f'<span style="background:{score_colour(sc)};color:white;'
                f'padding:2px 8px;border-radius:8px;font-size:11px">'
                f'Rank #{row.get("rank",i+1)}</span>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Score history ─────────────────────────────────────────────────────────
    if score_df is not None:
        etf_cols = [c for c in score_df.columns if c in config.UNIVERSES[universe]]
        selected = st.multiselect("Score history — select ETFs", etf_cols,
                                  default=etf_cols[:5], key=f"{key_prefix}_score_sel")
        if selected:
            period = st.radio("Period", ["Last 2 years","Last 5 years","Full OOS"],
                              horizontal=True, key=f"{key_prefix}_period")
            df_s = score_df.copy()
            if period == "Last 2 years":   df_s = df_s[df_s.index >= "2024-01-01"]
            elif period == "Last 5 years": df_s = df_s[df_s.index >= "2021-01-01"]
            fig_s = go.Figure()
            for i, tkr in enumerate(selected):
                if tkr in df_s.columns:
                    fig_s.add_trace(go.Scatter(x=df_s.index, y=df_s[tkr],
                        mode="lines", name=tkr,
                        line=dict(width=1.4, color=PALETTE[i%len(PALETTE)])))
            fig_s.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_s.update_layout(title=f"{module_name} composite score history",
                yaxis_title="Score", height=380,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_s, use_container_width=True, key=f"{key_prefix}_score_ts")

        # Score heatmap
        recent_s = score_df[[c for c in score_df.columns
                              if c in config.UNIVERSES[universe]]].tail(252)
        fig_h = go.Figure(go.Heatmap(
            z=recent_s.values.T,
            x=recent_s.index.strftime("%Y-%m-%d"),
            y=list(recent_s.columns),
            colorscale="RdYlGn", zmid=0,
            colorbar=dict(title="Score"),
        ))
        fig_h.update_layout(
            title=f"{module_name} score heatmap — last 252 days",
            height=max(300, len(recent_s.columns)*22+80),
            margin=dict(t=40,b=60,l=60,r=20),
            xaxis=dict(tickangle=-45, nticks=12),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_h, use_container_width=True, key=f"{key_prefix}_heat")

    st.divider()

    # ── CASH weight + top pick frequency ──────────────────────────────────────
    if daily_df is not None:
        c1, c2 = st.columns(2)
        with c1:
            if "cash_wt" in daily_df.columns:
                fig_c = go.Figure(go.Scatter(
                    x=daily_df.index, y=daily_df["cash_wt"],
                    mode="lines", line=dict(color="#888888", width=1.3),
                    fill="tozeroy", fillcolor="rgba(136,136,136,0.1)",
                ))
                fig_c.add_hline(y=0.30, line_dash="dash", line_color="#E74C3C",
                                annotation_text="CASH signal (30%)")
                fig_c.update_layout(
                    title="CASH policy weight",
                    yaxis_title="CASH weight",
                    yaxis=dict(range=[0, config.CASH_WEIGHT_MAX+0.05]),
                    height=280, plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_c, use_container_width=True, key=f"{key_prefix}_cash")

        with c2:
            if "top_ticker" in daily_df.columns:
                picks = daily_df["top_ticker"].value_counts()
                fig_f = go.Figure(go.Bar(
                    x=picks.index, y=picks.values,
                    marker_color="#1B4F8A",
                    text=picks.values, textposition="outside",
                ))
                fig_f.update_layout(
                    title="Top-pick frequency (OOS)",
                    yaxis_title="Days as #1", height=280,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_f, use_container_width=True, key=f"{key_prefix}_freq")

    st.caption(
        f"Module: {module_name} | Graph window: {graph_window}d | "
        f"CF penalty: {config.CF_PENALTY_WT} | OOS: {config.OOS_START}"
    )


# ── Main two-module tabs ──────────────────────────────────────────────────────
tab_lingam, tab_pcmci = st.tabs([
    f"🔬 Module A — LiNGAM ({config.LINGAM_WINDOW}d)",
    f"📡 Module B — PCMCI+ ({pcmci_w}d)",
])

with tab_lingam:
    st.markdown(
        "**LiNGAM (Linear Non-Gaussian Acyclic Model)** — identifies instantaneous "
        "causal effects using non-Gaussian noise structure. Best for same-day macro "
        "impacts (e.g. Fed announcement → XLF on the same bar). Fast, 252-day rolling window."
    )
    render_module_tabs(lingam_mod, "LiNGAM", config.LINGAM_WINDOW,
                       universe, "lingam")

with tab_pcmci:
    st.markdown(
        f"**PCMCI+ (Momentary Conditional Independence)** — tests lagged causal "
        f"links using partial correlation with lag up to {config.MAX_LAG}. Best for "
        f"multi-day causal transmission (e.g. VIX spike → XLE over 2 days). "
        f"Longer {pcmci_w}d window for statistical power."
    )
    render_module_tabs(pcmci_mod, "PCMCI+", pcmci_w,
                       universe, "pcmci")
