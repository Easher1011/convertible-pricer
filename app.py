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


# Colours (colour-blind friendly)
BLUE = "#2a78d6"     # convertible
ORANGE = "#eb6834"   # parity / implied
GREY = "#8a8985"     # reference lines

st.set_page_config(page_title="Convertible Bond Pricer", layout="wide")


# ---------------------------------------------------------------------
# Helper calculations
# ---------------------------------------------------------------------
def bond_floor(bond: ConvertibleBond, mkt: Market) -> float:
    """Value with no conversion option: coupons + redemption at rate + spread."""
    y = mkt.rate + mkt.credit_spread
    c = bond.face * bond.coupon_rate / bond.coupon_freq
    n = int(round(bond.maturity * bond.coupon_freq))
    pv = sum(c * math.exp(-y * (bond.maturity - k / bond.coupon_freq))
             for k in range(n) if bond.maturity - k / bond.coupon_freq > 0)
    return pv + bond.face * math.exp(-y * bond.maturity)


# st.cache_data remembers results, so moving one slider back and forth
# doesn't redo calculations it has already done.
@st.cache_data
def curve_vs_spot(bond_kw: dict, mkt_kw: dict, steps: int, method: str):
    bond = ConvertibleBond(**bond_kw)
    lo, hi = bond.conversion_price * 0.02, bond.conversion_price * 2.4
    if method == "fd":
        # One grid solve gives the price at every share price at once
        S, V, _, _ = solve_fd_grid(bond, Market(**mkt_kw), steps)
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
def curve_vs_input(bond_kw: dict, mkt_kw: dict, field: str, values: tuple, steps: int,
                   method: str):
    bond = ConvertibleBond(**bond_kw)
    return np.array([
        price_convertible(bond, Market(**{**mkt_kw, field: v}), steps, method)["price"]
        for v in values
    ])


@st.cache_data
def headline(bond_kw: dict, mkt_kw: dict, steps: int, method: str):
    bond, mkt = ConvertibleBond(**bond_kw), Market(**mkt_kw)
    return price_convertible(bond, mkt, steps, method), greeks(bond, mkt, steps, method)


@st.cache_data
def solve_implied(bond_kw: dict, mkt_kw: dict, dirty: float, steps: int, method: str):
    bond, mkt = ConvertibleBond(**bond_kw), Market(**mkt_kw)
    return (implied_vol(bond, mkt, dirty, steps, method),
            implied_spread(bond, mkt, dirty, steps, method))


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

st.sidebar.header("Model")
method_label = st.sidebar.radio("Engine", ["Finite difference", "Binomial tree"],
                                help="Finite difference is smooth and the default. The tree is "
                                     "there for comparison: it's simpler but jumpy with calls.")
method = "fd" if method_label == "Finite difference" else "tree"
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


# ---------------------------------------------------------------------
# Headline numbers
# ---------------------------------------------------------------------
st.title("Convertible Bond Pricer")
st.caption("Tsiveriotis–Fernandes model: the cash part of the bond is discounted at "
           "rate + credit spread, the equity part at the risk-free rate. Solved by "
           "Crank–Nicolson finite differences (or a binomial tree, for comparison).")

res, g = headline(bond_kw, mkt_kw, steps, method)
floor = bond_floor(bond, mkt)

c = st.columns(6)
c[0].metric("Price (dirty)", f"{res['price']:.2f}")
c[1].metric("Parity", f"{res['parity']:.2f}")
c[2].metric("Premium to parity", f"{res['premium_pct']:.1%}")
c[3].metric("Bond floor", f"{floor:.2f}")
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
c[4].metric("Cash part", f"{res['bond_part']:.2f}")
c[5].metric("Equity part", f"{res['equity_part']:.2f}")


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Price vs share price", "Sensitivities", "Market check"])

with tab1:
    spots, prices, deltas = curve_vs_spot(bond_kw, mkt_kw, steps, method)

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

    fig2 = go.Figure(go.Scatter(x=spots, y=deltas, name="Delta",
                                line=dict(color=BLUE, width=2.5)))
    fig2.add_vline(x=spot, line=dict(color=GREY, width=1, dash="dot"))
    fig2.update_yaxes(range=[0, 105])
    st.plotly_chart(tidy(fig2, 240, "Delta (%)", "Share price"), width="stretch")

with tab2:
    left, right = st.columns(2)

    vols = tuple(np.round(np.linspace(0.05, 0.80, 31), 4))
    pv = curve_vs_input(bond_kw, mkt_kw, "vol", vols, steps, method)
    f = go.Figure(go.Scatter(x=np.array(vols) * 100, y=pv, name="Price",
                             line=dict(color=BLUE, width=2.5)))
    f.add_vline(x=vol, line=dict(color=GREY, width=1, dash="dot"))
    left.subheader("Price vs volatility")
    left.plotly_chart(tidy(f, 340, "Price", "Volatility (%)"), width="stretch")
    left.caption("Higher vol makes the conversion option worth more. "
                 "Switch the engine to the tree to see the jumps it produces.")

    sprs = tuple(np.linspace(0, 0.15, 31))
    ps = curve_vs_input(bond_kw, mkt_kw, "credit_spread", sprs, steps, method)
    f = go.Figure(go.Scatter(x=np.array(sprs) * 1e4, y=ps, name="Price",
                             line=dict(color=BLUE, width=2.5)))
    f.add_vline(x=spread, line=dict(color=GREY, width=1, dash="dot"))
    right.subheader("Price vs credit spread")
    right.plotly_chart(tidy(f, 340, "Price", "Credit spread (bp)"), width="stretch")
    right.caption("Wider spreads hit the cash part of the bond. The deeper in the "
                  "money the shares, the less this matters.")

with tab3:
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
        iv, isp = solve_implied(bond_kw, mkt_kw, dirty, steps, method)

    rv, hist = None, None
    if manual_rv > 0:
        rv = manual_rv / 100
    elif ticker.strip():
        try:
            hist = fetch_history(ticker.strip().upper())
            if hist is not None and len(hist) > 30:
                rv = realised_vol(hist.values)
            else:
                st.warning("No price history came back for that ticker.")
        except Exception as e:
            st.warning(f"Couldn't fetch {ticker}: {e}")

    m = st.columns(5)
    m[0].metric("Accrued interest", f"{accrued:.3f}")
    m[1].metric("Dirty price", f"{dirty:.2f}")
    m[2].metric("Implied vol", f"{iv:.1%}" if iv is not None else "n/a",
                help="Vol that makes the model match the market, holding the spread "
                     "slider fixed.")
    m[3].metric("Implied spread", f"{isp * 1e4:.0f}bp" if isp is not None else "n/a",
                help="Spread that makes the model match the market, holding the vol "
                     "slider fixed.")
    m[4].metric("Realised vol (1y)", f"{rv:.1%}" if rv is not None else "—")

    if iv is None:
        st.info("No volatility between 1% and 150% reproduces that price with the "
                "current spread. Check the terms, the share price and the spread.")
    elif rv is not None:
        gap = iv - rv
        st.metric("Implied minus realised", f"{gap * 100:+.1f} vol pts")
        if gap < -0.05:
            st.success("Implied vol is well **below** realised: on this measure the "
                       "convertible looks **cheap**. Buying it and shorting shares "
                       "against it (delta hedging) would earn more from share-price "
                       "swings than the bond is pricing in, if realised vol holds up.")
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


st.divider()
st.caption("Model notes: the finite-difference engine uses a fixed share-price grid "
           "with Crank–Nicolson time stepping and Rannacher smoothing after coupons and "
           "events. Delta and gamma come from the grid; other Greeks are bump-and-reprice. "
           "Soft calls are tested on the current share price rather than over a "
           "trading-day window. The binomial tree is kept for comparison: with a call "
           "feature, its price jumps as vol changes, because the cash/equity split flips "
           "for whole rows of nodes at once.")
