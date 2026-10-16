"""
Convertible Bond Pricer — Streamlit website
===========================================

How to run (in the VS Code terminal, with (.venv) showing):
    streamlit run app.py

A browser tab opens with the site. Every time you save this file,
the site offers to refresh with your changes.
"""

import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from convertible_pricer import (ConvertibleBond, Market, price_convertible, greeks,
                                accrued_interest, implied_vol, implied_spread,
                                realised_vol, solve_fd_grid)
from stress import stress_grid
from arb_sim import SimSettings, Scenario, SCENARIOS, run_simulation
from mc_pricer import soft_call_value, TriggerRule
import os
import pandas as pd


# Colours (colour-blind friendly palette, fixed order)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREY = "#8a8985"
RED = "#c4332b"

COMPONENT_COLOURS = {"vol": BLUE, "credit": ORANGE, "carry": AQUA,
                     "shock": YELLOW, "default": MAGENTA, "other": GREY}
COMPONENT_LABELS = {"vol": "Vol (gamma net of theta)", "credit": "Credit carry",
                    "carry": "Borrow & financing", "shock": "Stress shock",
                    "default": "Default", "other": "Other (jumps, slippage)"}

st.set_page_config(page_title="Convertible Bond Pricer", layout="wide")


# ---------------------------------------------------------------------
# Helper calculations (cached so moving a slider back doesn't recompute)
# ---------------------------------------------------------------------
def bond_floor(bond: ConvertibleBond, mkt: Market) -> float:
    """Value with no conversion option: coupons + redemption at rate + spread."""
    y = mkt.rate + mkt.credit_spread
    c = bond.face * bond.coupon_rate / bond.coupon_freq
    n = int(round(bond.maturity * bond.coupon_freq))
    pv = sum(c * math.exp(-y * (bond.maturity - k / bond.coupon_freq))
             for k in range(n) if bond.maturity - k / bond.coupon_freq > 0)
    return pv + bond.face * math.exp(-y * bond.maturity)


@st.cache_data
def curve_vs_spot(bond_kw, mkt_kw, steps, method, model, mkw):
    bond = ConvertibleBond(**bond_kw)
    lo, hi = bond.conversion_price * 0.02, bond.conversion_price * 2.4
    if method == "fd":
        # One grid solve gives the price at every share price at once
        S, V, _, _ = solve_fd_grid(bond, Market(**mkt_kw), steps, model=model, **mkw)
        keep = (S >= lo) & (S <= hi)
        delta = np.gradient(V, S) / bond.conversion_ratio * 100
        return S[keep], V[keep], delta[keep]
    spots = np.linspace(lo, hi, 90)
    prices, deltas = [], []
    for s in spots:
        res = price_convertible(bond, Market(**{**mkt_kw, "spot": s}), steps, method)
        prices.append(res["price"])
        deltas.append(res["delta"] / bond.conversion_ratio * 100)
    return spots, np.array(prices), np.array(deltas)


@st.cache_data
def curve_vs_input(bond_kw, mkt_kw, field, values, steps, method, model, mkw):
    bond = ConvertibleBond(**bond_kw)
    return np.array([
        price_convertible(bond, Market(**{**mkt_kw, field: v}), steps, method,
                          model, **mkw)["price"]
        for v in values
    ])


@st.cache_data
def headline(bond_kw, mkt_kw, steps, method, model, mkw):
    bond, mkt = ConvertibleBond(**bond_kw), Market(**mkt_kw)
    return (price_convertible(bond, mkt, steps, method, model, **mkw),
            greeks(bond, mkt, steps, method, model, **mkw))


@st.cache_data
def solve_implied(bond_kw, mkt_kw, dirty, steps, method, model, mkw):
    bond, mkt = ConvertibleBond(**bond_kw), Market(**mkt_kw)
    return (implied_vol(bond, mkt, dirty, steps, method, model, **mkw),
            implied_spread(bond, mkt, dirty, steps, method, model, **mkw))


@st.cache_data
def cached_stress(bond_kw, mkt_kw, spot_shocks, other_shocks, shock_type, hedged,
                  model, mkw):
    return stress_grid(ConvertibleBond(**bond_kw), Market(**mkt_kw), spot_shocks,
                       other_shocks, shock_type, hedged, 300, model, **mkw)


@st.cache_data
def cached_sim(bond_kw, mkt_kw, sim_kw, scenario_kw, model, mkw):
    sc = Scenario(**scenario_kw) if scenario_kw else None
    sim = SimSettings(**sim_kw, scenario=sc)
    return run_simulation(ConvertibleBond(**bond_kw), Market(**mkt_kw), sim,
                          model=model, nt=300, **mkw)


@st.cache_data
def cached_mc(bond_kw, mkt_kw, days_needed, window, n_paths, recovery, hazard_power):
    return soft_call_value(ConvertibleBond(**bond_kw), Market(**mkt_kw),
                           TriggerRule(days_needed, window), n_paths=n_paths,
                           recovery=recovery, hazard_power=hazard_power)


@st.cache_data(ttl=3600)
def fetch_history(ticker: str):
    """Last ~1 year of daily closes from Yahoo Finance (free, sometimes patchy)."""
    import yfinance as yf
    df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    close = df["Close"]
    if hasattr(close, "columns"):          # newer yfinance returns a table
        close = close.iloc[:, 0]
    return close.dropna()


def tidy(fig, height=420, y_title="", x_title=""):
    """Consistent, quiet chart styling."""
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title=x_title, yaxis_title=y_title,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


def heatmap(z, x_labels, y_labels, x_title, y_title):
    """P&L heatmap: red = loss, grey = flat, blue = gain. Values written in each cell."""
    lim = float(np.nanmax(np.abs(z))) or 1.0
    fig = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=y_labels, zmid=0, zmin=-lim, zmax=lim,
        colorscale=[[0, RED], [0.5, "#f0efec"], [1, BLUE]],
        text=np.round(z, 1), texttemplate="%{text}", textfont=dict(size=12),
        hovertemplate=f"{x_title}: %{{x}}<br>{y_title}: %{{y}}<br>P&L: %{{z:.2f}}<extra></extra>",
        colorbar=dict(title="P&L"),
    ))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title=x_title, yaxis_title=y_title,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    return fig


# ---------------------------------------------------------------------
# Sidebar: inputs
# ---------------------------------------------------------------------
st.sidebar.header("Bond terms")
coupon = st.sidebar.number_input("Coupon (%)", 0.0, 15.0, 2.0, 0.125, format="%.3f")
coupon_freq = st.sidebar.selectbox("Coupons per year", [2, 1, 4], index=0)
maturity = st.sidebar.number_input("Years to maturity", 0.1, 30.0, 5.0, 0.25, format="%.2f")
conv_price = st.sidebar.number_input("Conversion price", 0.01, 10000.0, 50.0, 1.0)

callable_ = st.sidebar.checkbox("Issuer can call", value=True)
call_price = st.sidebar.number_input("Call price", 90.0, 150.0, 103.0, 0.5,
                                     disabled=not callable_)
call_start = st.sidebar.number_input("Callable from (years from now)", 0.0, float(maturity),
                                     float(min(3.0, maturity)), 0.25,
                                     disabled=not callable_)
soft_call = st.sidebar.checkbox("Soft call (share price trigger)", value=False,
                                disabled=not callable_)
trigger = st.sidebar.number_input("Trigger (% of conversion price)", 100, 200, 130, 5,
                                  disabled=not (callable_ and soft_call))

puttable = st.sidebar.checkbox("Holder can put", value=False)
put_price = st.sidebar.number_input("Put price", 80.0, 120.0, 100.0, 0.5,
                                    disabled=not puttable)
put_year = st.sidebar.number_input("Put date (years from now)", 0.25, float(maturity),
                                   float(min(3.0, maturity)), 0.25, disabled=not puttable)

st.sidebar.header("Market")
spot = st.sidebar.number_input("Share price", 0.01, 100000.0,
                               float(round(conv_price * 0.8, 2)), 0.5)
vol = st.sidebar.slider("Volatility (%)", 5.0, 120.0, 30.0, 1.0)
rate = st.sidebar.slider("Risk-free rate (%)", 0.0, 8.0, 4.0, 0.05)
spread = st.sidebar.slider("Credit spread (bp)", 0, 1500, 300, 5)
div = st.sidebar.slider("Dividend yield (%)", 0.0, 8.0, 0.0, 0.1)

st.sidebar.header("Credit model")
model_label = st.sidebar.radio(
    "How default risk is modelled",
    ["Fixed spread", "Jump-to-default"],
    help="Fixed spread: the cash part of the bond is discounted at rate + spread "
         "(Tsiveriotis-Fernandes). Jump-to-default: the issuer can default at any "
         "moment, more likely when the shares are low; on default the shares go to "
         "zero and the bond pays its recovery value.")
model = "tf" if model_label == "Fixed spread" else "jtd"
if model == "jtd":
    recovery = st.sidebar.slider("Recovery rate (%)", 0, 80, 40, 5) / 100
    hazard_power = st.sidebar.slider(
        "Default risk sensitivity to share price", 0.0, 3.0, 1.0, 0.25,
        help="0 = default risk doesn't depend on the share price. 1 = halving the "
             "share price doubles the default risk. Higher = steeper.")
    mkw = {"recovery": recovery, "hazard_power": hazard_power}
else:
    mkw = {}

st.sidebar.header("Numerics")
method_label = st.sidebar.radio(
    "Engine", ["Finite difference", "Binomial tree"],
    disabled=(model == "jtd"),
    help="Finite difference is smooth and the default. The tree is there for "
         "comparison: simpler, but jumpy with calls. Jump-to-default needs finite difference.")
method = "fd" if (method_label == "Finite difference" or model == "jtd") else "tree"
steps = st.sidebar.select_slider("Time steps", options=[100, 200, 300, 400, 800], value=300)
st.sidebar.caption("More steps = more accurate but slower.")

bond_kw = dict(
    face=100.0, coupon_rate=coupon / 100, coupon_freq=coupon_freq, maturity=maturity,
    conversion_ratio=100.0 / conv_price,
    call_price=call_price if callable_ else None, call_start=call_start,
    call_trigger=trigger / 100 if (callable_ and soft_call) else None,
    put_price=put_price if puttable else None,
    put_dates=[put_year] if puttable else [],
)
mkt_kw = dict(spot=spot, vol=vol / 100, rate=rate / 100,
              credit_spread=spread / 1e4, div_yield=div / 100)
bond, mkt = ConvertibleBond(**bond_kw), Market(**mkt_kw)
args = (steps, method, model, mkw)


# ---------------------------------------------------------------------
# Headline numbers
# ---------------------------------------------------------------------
st.title("Convertible Bond Pricer")
st.caption("Fixed-spread (Tsiveriotis–Fernandes) or jump-to-default credit model, solved by "
           "Crank–Nicolson finite differences, with stress testing and a convertible "
           "arbitrage simulator.")

res, g = headline(bond_kw, mkt_kw, *args)
floor = bond_floor(bond, mkt)

c = st.columns(6)
c[0].metric("Price (dirty)", f"{res['price']:.2f}")
c[1].metric("Parity", f"{res['parity']:.2f}")
c[2].metric("Premium to parity", f"{res['premium_pct']:.1%}")
c[3].metric("Bond floor", f"{floor:.2f}",
            help="Coupons and redemption only, discounted at rate + spread.")
c[4].metric("Delta", f"{g['delta_pct']:.0%}")
c[5].metric("Investment premium", f"{res['price'] / floor - 1:.1%}",
            help="How far the price sits above the bond floor — your downside "
                 "if the shares fall to zero and credit holds.")

c = st.columns(6)
c[0].metric("Gamma (Δ pts per 1%)",
            f"{g['gamma'] * mkt.spot * 0.01 / bond.conversion_ratio * 100:.2f}",
            help="How many percentage points delta rises if the share price goes up 1%.")
c[1].metric("Vega (per vol pt)", f"{g['vega']:.3f}")
c[2].metric("CS01 (per bp)", f"{g['cs01']:.4f}")
c[3].metric("Rho (per bp)", f"{g['rho_01']:.4f}")
c[4].metric("Cash part", f"{res['bond_part']:.2f}" if res["bond_part"] is not None else "—",
            help="Fixed-spread model only.")
c[5].metric("Equity part", f"{res['equity_part']:.2f}" if res["equity_part"] is not None else "—",
            help="Fixed-spread model only.")


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Price vs share price", "Sensitivities", "Stress test", "Arbitrage simulator",
     "Market check", "Path-dependent call", "Validation"])

# ---------- Tab 1: price curve ----------
with tab1:
    spots, prices, deltas = curve_vs_spot(bond_kw, mkt_kw, *args)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spots, y=prices, name="Convertible",
                             line=dict(color=BLUE, width=3)))
    fig.add_trace(go.Scatter(x=spots, y=bond.conversion_ratio * spots,
                             name="Parity", line=dict(color=ORANGE, width=2)))
    fig.add_trace(go.Scatter(x=spots, y=[floor] * len(spots), name="Bond floor",
                             line=dict(color=GREY, width=2, dash="dash")))
    if callable_:
        fig.add_hline(y=call_price, line=dict(color=GREY, width=1, dash="dot"),
                      annotation_text=f"Call {call_price:g}",
                      annotation_position="top left")
    fig.add_trace(go.Scatter(x=[spot], y=[res["price"]], mode="markers",
                             name="Today", marker=dict(color=BLUE, size=11,
                             line=dict(color="white", width=2))))
    st.plotly_chart(tidy(fig, 430, "Price (per 100 face)", "Share price"), width="stretch")
    if model == "jtd":
        st.caption("Under jump-to-default, the price falls below the fixed-spread bond floor "
                   "when the shares are low: default risk rises as the share price falls.")

    fig2 = go.Figure(go.Scatter(x=spots, y=deltas, name="Delta",
                                line=dict(color=BLUE, width=2.5)))
    fig2.add_vline(x=spot, line=dict(color=GREY, width=1, dash="dot"))
    if model == "tf":
        fig2.update_yaxes(range=[0, 105])
    st.plotly_chart(tidy(fig2, 240, "Delta (%)", "Share price"), width="stretch")
    if model == "jtd":
        st.caption("Delta can go above 100% when the shares are very low: default risk is "
                   "so high there that the bond falls faster than the shares it converts into.")

# ---------- Tab 2: sensitivities ----------
with tab2:
    left, right = st.columns(2)

    vols = tuple(np.round(np.linspace(0.05, 0.80, 31), 4))
    pv = curve_vs_input(bond_kw, mkt_kw, "vol", vols, *args)
    f = go.Figure(go.Scatter(x=np.array(vols) * 100, y=pv, name="Price",
                             line=dict(color=BLUE, width=2.5)))
    f.add_vline(x=vol, line=dict(color=GREY, width=1, dash="dot"))
    left.subheader("Price vs volatility")
    left.plotly_chart(tidy(f, 340, "Price", "Volatility (%)"), width="stretch")
    left.caption("Higher vol makes the conversion option worth more. "
                 "Switch the engine to the tree to see the jumps it produces.")

    sprs = tuple(np.round(np.linspace(0, 0.15, 31), 5))
    ps = curve_vs_input(bond_kw, mkt_kw, "credit_spread", sprs, *args)
    f = go.Figure(go.Scatter(x=np.array(sprs) * 1e4, y=ps, name="Price",
                             line=dict(color=BLUE, width=2.5)))
    f.add_vline(x=spread, line=dict(color=GREY, width=1, dash="dot"))
    right.subheader("Price vs credit spread")
    right.plotly_chart(tidy(f, 340, "Price", "Credit spread (bp)"), width="stretch")
    right.caption("Wider spreads hit the cash part of the bond. Under jump-to-default, very "
                  "high default risk also pushes up the conversion option (shares must "
                  "drift up faster to compensate), so the line can flatten or turn up.")

# ---------- Tab 3: stress test ----------
with tab3:
    st.subheader("Instant stress: what if this happened right now?")
    hedged = st.toggle("Delta-hedged (short shares against the bond)", value=True,
                       help="On: the convertible arbitrage position. Off: you just own the bond.")
    spot_shocks = (-0.5, -0.3, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3, 0.5)
    spread_shocks = (-0.01, 0.0, 0.01, 0.02, 0.03, 0.05, 0.08)
    vol_shocks = (-0.10, -0.05, 0.0, 0.05, 0.10, 0.20)
    xl = [f"{x:+.0%}" for x in spot_shocks]

    left, right = st.columns(2)
    z1 = cached_stress(bond_kw, mkt_kw, spot_shocks, spread_shocks, "spread", hedged,
                       model, mkw)
    left.markdown("**Share price × credit spread**")
    left.plotly_chart(heatmap(z1, xl, [f"{s*1e4:+.0f}bp" for s in spread_shocks],
                              "Share price move", "Spread move"), width="stretch")
    z2 = cached_stress(bond_kw, mkt_kw, spot_shocks, vol_shocks, "vol", hedged, model, mkw)
    right.markdown("**Share price × implied vol**")
    right.plotly_chart(heatmap(z2, xl, [f"{s*100:+.0f} pts" for s in vol_shocks],
                               "Share price move", "Vol move"), width="stretch")
    if hedged:
        st.caption("P&L per 100 face, bond plus share hedge. Look along the 'no change' row: "
                   "gains on big share moves in either direction (gamma). Look down a column: "
                   "spread widening and falling vol are what hurt a hedged book.")
    else:
        st.caption("P&L per 100 face, bond only. Losses on the left are cushioned by the bond "
                   "floor; gains on the right follow the shares.")

# ---------- Tab 4: arbitrage simulator ----------
with tab4:
    st.subheader("Convertible arbitrage: buy the bond, short the shares, rebalance")
    st.write("Simulates the trade along many random share-price paths. The bond is priced "
             "at the **implied vol** from the sidebar; the shares actually move with the "
             "**realised vol** you set here. You make money on average when realised beats "
             "implied, and lose it to credit, borrow costs and stress events.")

    scen_name = st.selectbox("Stress scenario", list(SCENARIOS) + ["Custom"],
                             help="A one-off shock on a chosen day. 'Custom' lets you set it.")
    with st.form("sim_form"):
        a, b_, c_ = st.columns(3)
        rv = a.slider("Realised vol (%)", 5.0, 120.0, float(vol + 5), 1.0)
        horizon_m = a.slider("Holding period (months)", 1, int(min(60, maturity * 12 - 1)),
                             int(min(12, maturity * 12 - 1)))
        n_paths = b_.select_slider("Number of paths", [100, 250, 500, 1000, 2000], 500)
        rebal = b_.radio("Rebalance hedge", ["Daily", "Weekly"], horizontal=True)
        borrow = c_.slider("Stock borrow fee (%/yr)", 0.0, 10.0, 0.5, 0.25)
        sim_default = c_.checkbox("Simulate defaults", value=True, disabled=(model != "jtd"),
                                  help="Jump-to-default model only.")
        if scen_name == "Custom":
            d1, d2, d3, d4, d5 = st.columns(5)
            sc_day = d1.number_input("Day", 1, 1260, 60)
            sc_share = d2.number_input("Share jump (%)", -90.0, 200.0, -20.0, 5.0)
            sc_spread = d3.number_input("Spread jump (bp)", -500, 3000, 300, 25)
            sc_vol = d4.number_input("Vol jump (pts)", -50.0, 100.0, 10.0, 1.0)
            sc_borrow = d5.number_input("Borrow jump (%)", 0.0, 50.0, 0.0, 0.5)
        st.form_submit_button("Run simulation", type="primary")

    if scen_name == "Custom":
        scenario_kw = dict(day=int(sc_day), share_jump=sc_share / 100,
                           spread_jump=sc_spread / 1e4, vol_jump=sc_vol / 100,
                           borrow_jump=sc_borrow / 100)
    else:
        sc = SCENARIOS[scen_name]
        scenario_kw = sc.__dict__ if sc else None
    sim_kw = dict(horizon=horizon_m / 12, n_paths=int(n_paths), realised_vol=rv / 100,
                  rebalance_every=1 if rebal == "Daily" else 5,
                  borrow_fee=borrow / 100, seed=42, simulate_default=sim_default)

    with st.spinner("Running paths..."):
        out = cached_sim(bond_kw, mkt_kw, sim_kw, scenario_kw, model, mkw)
    s = out["summary"]

    m = st.columns(6)
    m[0].metric("Average P&L", f"{s['mean']:+.2f}")
    m[1].metric("Median P&L", f"{s['median']:+.2f}")
    m[2].metric("Worst 5%", f"{s['p5']:+.2f}", help="5th percentile of final P&L.")
    m[3].metric("Best 5%", f"{s['p95']:+.2f}", help="95th percentile of final P&L.")
    m[4].metric("Profitable paths", f"{s['pct_profitable']:.0%}")
    m[5].metric("Defaulted paths", f"{s['pct_defaulted']:.1%}" if model == "jtd" else "—")
    st.caption("P&L per 100 face of bonds, over the whole holding period.")

    left, right = st.columns(2)
    comps = [k for k in COMPONENT_LABELS if abs(s["by_component"][k]) > 1e-4 or k in ("vol", "credit", "carry")]
    vals = [s["by_component"][k] for k in comps]
    f = go.Figure(go.Bar(
        x=vals, y=[COMPONENT_LABELS[k] for k in comps], orientation="h",
        marker_color=[COMPONENT_COLOURS[k] for k in comps],
        text=[f"{v:+.2f}" for v in vals], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
    f.update_yaxes(autorange="reversed")
    span = max(max(vals), 0) - min(min(vals), 0) or 1.0
    f.update_xaxes(range=[min(min(vals), 0) - 0.25 * span, max(max(vals), 0) + 0.25 * span])
    left.markdown("**Where the money came from (average across paths)**")
    left.plotly_chart(tidy(f, 320, "", "P&L"), width="stretch")

    final = out["total"][-1]
    f = go.Figure(go.Histogram(x=final, nbinsx=40, marker_color=BLUE,
                               marker_line=dict(color="white", width=1),
                               hovertemplate="P&L %{x}<br>Paths: %{y}<extra></extra>"))
    f.add_vline(x=0, line=dict(color=GREY, width=1))
    f.add_vline(x=s["mean"], line=dict(color=ORANGE, width=2, dash="dash"),
                annotation_text=f"Average {s['mean']:+.2f}", annotation_position="top")
    right.markdown("**Spread of outcomes**")
    right.plotly_chart(tidy(f, 320, "Paths", "Final P&L"), width="stretch")

    st.markdown("**One path in detail**")
    p_idx = st.slider("Path number", 1, int(n_paths), 1) - 1
    days = out["days"]
    left, right = st.columns(2)
    f = go.Figure(go.Scatter(x=days, y=out["paths"][:, p_idx], name="Share price",
                             line=dict(color=BLUE, width=2)))
    f.add_hline(y=bond.conversion_price, line=dict(color=GREY, width=1, dash="dot"),
                annotation_text="Conversion price", annotation_position="top left")
    if scenario_kw:
        f.add_vline(x=scenario_kw["day"], line=dict(color=YELLOW, width=1.5, dash="dash"),
                    annotation_text="Stress", annotation_position="top")
    left.plotly_chart(tidy(f, 320, "Share price", "Trading day"), width="stretch")

    f = go.Figure()
    for k in COMPONENT_LABELS:
        series = out["cum"][k][:, p_idx]
        if np.abs(series).max() > 1e-6:
            f.add_trace(go.Scatter(x=days, y=series, name=COMPONENT_LABELS[k],
                                   line=dict(color=COMPONENT_COLOURS[k], width=1.5)))
    f.add_trace(go.Scatter(x=days, y=out["total"][:, p_idx], name="Total",
                           line=dict(color="#0b0b0b", width=3)))
    right.plotly_chart(tidy(f, 320, "Cumulative P&L", "Trading day"), width="stretch")

    with st.expander("How the P&L is split"):
        st.markdown(
            "- **Vol (gamma net of theta):** each day you earn ½ × gamma × (share move)², "
            "and the bond's time decay charges you what that was *expected* to be at the "
            "implied vol. The difference is positive on average only if realised > implied.\n"
            "- **Credit carry:** the spread you earn for holding the issuer's default risk. "
            "Under the fixed-spread model no defaults ever happen in the simulation, so this "
            "looks like free money. Switch to jump-to-default with defaults on and the "
            "default losses offset it on average.\n"
            "- **Borrow & financing:** stock borrow fee, plus whatever is left of interest, "
            "dividends and coupons (these largely cancel by design).\n"
            "- **Stress shock:** instant repricing when spreads or implied vol jump.\n"
            "- **Other:** hedge slippage between rebalances, and large jumps the simple "
            "gamma formula doesn't fully capture.\n\n"
            "Check: with zero borrow fee, no defaults, and realised vol = implied vol, the "
            "average P&L comes out at zero.")

# ---------- Tab 5: market check ----------
with tab5:
    st.subheader("Is this convertible cheap or expensive?")
    st.write("Enter the bond's **clean** market price (as quoted, per 100 face). "
             "The model works out which volatility, and which credit spread, would "
             "explain that price, then compares the implied vol with how volatile "
             "the shares have actually been.")

    a, b_, c_ = st.columns(3)
    accrued = accrued_interest(bond)
    mkt_clean = a.number_input("Market price (clean)", 1.0, 1000.0,
                               float(round(res["price"] - accrued, 2)), 0.25)
    ticker = b_.text_input("Share ticker (Yahoo format, optional)", "",
                           help="e.g. AAPL, VOD.L for London, SAP.DE for Frankfurt")
    manual_rv = c_.number_input("Or enter realised vol yourself (%)", 0.0, 200.0, 0.0, 1.0,
                                help="Leave at 0 to use the ticker instead.")

    dirty = mkt_clean + accrued
    with st.spinner("Solving for implied vol and spread..."):
        iv, isp = solve_implied(bond_kw, mkt_kw, dirty, *args)

    rv_hist, hist = None, None
    if manual_rv > 0:
        rv_hist = manual_rv / 100
    elif ticker.strip():
        try:
            hist = fetch_history(ticker.strip().upper())
            if hist is not None and len(hist) > 30:
                rv_hist = realised_vol(hist.values)
            else:
                st.warning("No price history came back for that ticker.")
        except Exception as e:
            st.warning(f"Couldn't fetch {ticker}: {e}")

    m = st.columns(5)
    m[0].metric("Accrued interest", f"{accrued:.3f}")
    m[1].metric("Dirty price", f"{dirty:.2f}")
    m[2].metric("Implied vol", f"{iv:.1%}" if iv is not None else "n/a",
                help="Vol that makes the model match the market, holding the spread fixed.")
    m[3].metric("Implied spread", f"{isp * 1e4:.0f}bp" if isp is not None else "n/a",
                help="Spread that makes the model match the market, holding the vol fixed.")
    m[4].metric("Realised vol (1y)", f"{rv_hist:.1%}" if rv_hist is not None else "—")

    if iv is None:
        st.info("No volatility between 1% and 150% reproduces that price with the "
                "current spread. Check the terms, the share price and the spread.")
    elif rv_hist is not None:
        gap = iv - rv_hist
        st.metric("Implied minus realised", f"{gap * 100:+.1f} vol pts")
        if gap < -0.05:
            st.success("Implied vol is well **below** realised: on this measure the "
                       "convertible looks **cheap**. A delta-hedged long would earn more from "
                       "share-price swings than the bond is pricing in, if realised vol holds "
                       "up. Try it in the Arbitrage simulator tab.")
        elif gap > 0.05:
            st.error("Implied vol is well **above** realised: on this measure the "
                     "convertible looks **expensive** versus how the shares have moved.")
        else:
            st.info("Implied and realised vol are close: roughly fair on this measure.")
        st.caption("Before calling it a trade: the result depends on the credit spread "
                   "you've assumed (a wider spread lowers the bond floor, which raises "
                   "implied vol). Check listed option implied vols too, and whether "
                   "the past year's vol is a fair guide to the next.")

    if hist is not None and len(hist) > 30:
        logret = np.diff(np.log(hist.values))
        roll = [np.std(logret[i - 21:i], ddof=1) * np.sqrt(252) * 100
                for i in range(21, len(logret) + 1)]
        f = go.Figure(go.Scatter(x=hist.index[21:], y=roll, name="1m realised vol",
                                 line=dict(color=BLUE, width=2)))
        if iv is not None:
            f.add_hline(y=iv * 100, line=dict(color=ORANGE, width=2, dash="dash"),
                        annotation_text=f"Implied {iv:.0%}", annotation_position="top left")
        st.plotly_chart(tidy(f, 300, "Vol (%)", ""), width="stretch")
        st.caption("Rolling 1-month realised vol over the past year, against the "
                   "convertible's implied vol.")

# ---------- Tab 6: path-dependent soft call (Monte Carlo) ----------
with tab6:
    st.subheader("The real soft call: \"above the trigger on 20 of the last 30 days\"")
    st.write("The grid can only test today's share price against the trigger. Real "
             "clauses look back over a window of trading days, so whether the issuer may "
             "call depends on the **path** the shares took. This tab prices the real clause "
             "by simulation and shows how much value the simple shortcut misses.")
    if not (callable_ and soft_call):
        st.info("Turn on **Issuer can call** and **Soft call** in the sidebar to use this tab.")
    else:
        rec = mkw.get("recovery", 0.4)
        hp = mkw.get("hazard_power", 1.0)
        if model != "jtd":
            st.caption(f"Monte Carlo uses the jump-to-default credit model (recovery "
                       f"{rec:.0%}, sensitivity {hp:g}), so the grid price shown here is "
                       f"from that model too. Switch the sidebar to Jump-to-default to "
                       f"change those settings.")
        with st.form("mc_form"):
            a, b_, c_ = st.columns(3)
            days_needed = a.number_input("Days above trigger needed", 1, 60, 20)
            window = b_.number_input("...out of the last (trading days)", 1, 60, 30)
            n_mc = c_.select_slider("Number of paths", [5000, 10000, 20000], 10000)
            st.form_submit_button("Run Monte Carlo", type="primary")
        if days_needed > window:
            st.warning("Days needed can't be more than the window.")
        else:
            with st.spinner("Simulating paths..."):
                r = cached_mc(bond_kw, mkt_kw, int(days_needed), int(window), int(n_mc),
                              rec, hp)
            m = st.columns(4)
            m[0].metric("Grid price (today-only trigger)", f"{r['grid_today_only']:.2f}")
            m[1].metric(f"Extra value of {days_needed}-of-{window} rule",
                        f"{r['extra_value']:+.2f}",
                        help=f"± {r['extra_value_se']:.2f} (one standard error)")
            m[2].metric(f"Price with {days_needed}-of-{window} rule", f"{r['price']:.2f}")
            m[3].metric("Paths that defaulted", f"{r['pct_defaulted']:.1%}")
            st.caption(f"Extra value ± {r['extra_value_se']:.2f} (one standard error). "
                       f"Check: Monte Carlo with the today-only trigger gives "
                       f"{r['mc_today_only']:.2f} ± {r['mc_today_only_se']:.2f}, against "
                       f"the grid's {r['grid_today_only']:.2f}.")

        with st.expander("How this works"):
            st.markdown(
                "1. **Simulate** thousands of daily share-price paths, with defaults "
                "(more likely when the shares are low).\n"
                "2. **Count** along each path how many of the last N trading days closed "
                "above the trigger.\n"
                "3. **Work back from maturity.** On each day the holder and issuer decide "
                "using the grid's value of holding on: the holder puts or converts if that "
                "pays more; the issuer calls if the path's trigger count allows it and "
                "holding is worth more than the call price.\n"
                "4. **Accuracy trick (control variate):** price the bond twice on the "
                "*same* paths, once with the today-only trigger and once with the real "
                "rule. Noise is almost identical in both, so the *difference* is precise. "
                "Final price = grid price + that difference.\n\n"
                "Why the grid supplies the decisions: the textbook alternative "
                "(Longstaff–Schwartz regression) is noisy enough here to misjudge early "
                "conversion by about a point. The grid's decisions are nearly exact, and "
                "the trigger rule barely changes when the holder should convert, so this "
                "is a sound shortcut. Regression is still available in `mc_pricer.py` "
                "(`method=\"regression\"`).")

# ---------- Tab 7: validation ----------
with tab7:
    st.subheader("How do we know the numbers are right?")
    here = os.path.dirname(os.path.abspath(__file__))
    cf_path = os.path.join(here, "closed_form_checks.csv")
    bm_path = os.path.join(here, "benchmark_results.csv")
    if os.path.exists(cf_path):
        st.markdown("**1. Textbook formulas.** Cases with an exact answer.")
        cf = pd.read_csv(cf_path)
        st.dataframe(cf.style.format({"Exact": "{:.4f}", "Ours": "{:.4f}",
                                      "Difference": "{:+.4f}"}),
                     hide_index=True, width="stretch")
    if os.path.exists(bm_path):
        bm = pd.read_csv(bm_path)
        st.markdown("**2. QuantLib**, the industry-standard open-source pricing library, "
                    f"on {len(bm)} bonds (no call, hard call, soft call, put; different "
                    "share prices, vols, spreads, coupons and dividends).")
        c = st.columns(3)
        c[0].metric("Our replica of QuantLib's method vs QuantLib",
                    f"{bm['1. Replica - QuantLib'].abs().mean():.3f}",
                    help=f"Average gap per 100 face; largest "
                         f"{bm['1. Replica - QuantLib'].abs().max():.3f}")
        c[1].metric("Grid vs grid twice as fine",
                    f"{bm['2. FD convergence'].abs().mean():.3f}",
                    help=f"Average gap; largest {bm['2. FD convergence'].abs().max():.3f}")
        c[2].metric("Our model vs QuantLib's model",
                    f"{bm['3. TF - QuantLib (model gap)'].abs().mean():.2f}",
                    help=f"Average gap; largest "
                         f"{bm['3. TF - QuantLib (model gap)'].abs().max():.2f}. A real "
                         f"model difference, explained below.")
        st.markdown(
            "- **Replica vs QuantLib (≈ 0):** QuantLib's convertible engine doesn't use "
            "the Tsiveriotis–Fernandes cash/equity split. It tracks the probability of "
            "conversion at each tree node and discounts the whole value at a blend, "
            "*rate + (1 − probability) × spread*. Rebuilding that method separately and "
            "matching QuantLib to a few hundredths confirms both our tree code and what "
            "QuantLib does.\n"
            "- **Grid convergence (≈ 0):** halving the grid spacing barely moves the price.\n"
            "- **Model gap (up to ~1.8 points):** the two models genuinely differ. "
            "Tsiveriotis–Fernandes charges the credit spread only on the part of the value "
            "that depends on the issuer paying cash; the blended method spreads it across "
            "the whole value. A gap this size is bigger than many trading edges, which is "
            "why the choice of credit model matters.")
        show = bm[["Bond", "Share price", "Vol", "Spread (bp)", "QuantLib",
                   "Our replica of QuantLib", "1. Replica - QuantLib",
                   "Our TF finite diff", "3. TF - QuantLib (model gap)"]]
        with st.expander("Full benchmark table"):
            st.dataframe(show.style.format(precision=3), hide_index=True, width="stretch")
    if not (os.path.exists(cf_path) or os.path.exists(bm_path)):
        st.info("Run `python benchmark.py` once (needs `pip install QuantLib`) to create "
                "the validation results.")
    st.markdown("**3. Built-in checks** used while building the simulator and Monte Carlo "
                "pricer: the delta-hedged trade earns zero on average when realised vol "
                "equals implied and there's no credit or borrow cost; under jump-to-default, "
                "default losses exactly offset credit carry; and Monte Carlo with a "
                "today-only trigger reproduces the grid.")



st.divider()
st.caption("Model notes: fixed-spread model after Tsiveriotis & Fernandes (1998); "
           "jump-to-default with a hazard rate that rises as the share price falls, "
           "calibrated so today's spread = hazard × (1 − recovery). Crank–Nicolson finite "
           "differences on a fixed share-price grid, with Rannacher smoothing after coupons "
           "and events. The grid tests soft calls on the current share price; the "
           "Path-dependent call tab prices the real look-back clause by Monte Carlo. "
           "The simulator uses risk-neutral share drift and a constant "
           "realised vol, with no transaction costs.")
