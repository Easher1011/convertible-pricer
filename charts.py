"""
Charts for the convertible bond pricer
======================================

Draws the classic convertible bond picture:
  - Top chart:    convertible price vs share price, with the bond floor and parity lines
  - Bottom chart: delta vs share price (how equity-like the bond is)

Run it with the ▶ button. A window pops up and a PNG is saved in this folder.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from convertible_pricer import ConvertibleBond, Market, price_convertible, solve_fd_grid


# Colours (picked to be colour-blind friendly)
BLUE = "#2a78d6"     # convertible price
ORANGE = "#eb6834"   # parity
GREY = "#8a8985"     # bond floor / reference lines
INK = "#0b0b0b"
INK_SOFT = "#52514e"


def bond_floor(bond: ConvertibleBond, mkt: Market) -> float:
    """
    Value of the bond if it could NOT be converted: just the coupons and
    redemption, discounted at risk-free rate + credit spread.
    """
    y = mkt.rate + mkt.credit_spread
    coupon = bond.face * bond.coupon_rate / bond.coupon_freq
    n = int(round(bond.maturity * bond.coupon_freq))
    pv = sum(coupon * math.exp(-y * k / bond.coupon_freq) for k in range(1, n + 1))
    pv += bond.face * math.exp(-y * bond.maturity)
    return pv


def price_curve(bond, mkt, spots, steps=400):
    """
    Price the convertible (and get its delta) at each share price in `spots`.
    One finite-difference solve gives every share price at once.
    """
    S, V, _, _ = solve_fd_grid(bond, mkt, steps)
    dV = np.gradient(V, S)
    return np.interp(spots, S, V), np.interp(spots, S, dV)


def style_axis(ax):
    """Light, quiet axes so the data stands out."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cfcec9")
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    ax.grid(True, color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)


def make_chart(bond: ConvertibleBond, mkt: Market, save_as="convertible_chart.png"):
    # 1. Range of share prices to test (from very low to well above conversion price)
    spots = np.linspace(1, bond.conversion_price * 2.4, 120)

    # 2. Calculate the three lines
    cb_price, delta_shares = price_curve(bond, mkt, spots)
    parity = bond.conversion_ratio * spots
    floor = bond_floor(bond, mkt)

    # 3. Delta as % of the conversion ratio (0% = pure bond, 100% = pure shares)
    delta_pct = delta_shares / bond.conversion_ratio * 100

    # Today's position (for the marker)
    today = price_convertible(bond, mkt)

    # ---------- Draw ----------
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 8), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    fig.patch.set_facecolor("white")

    # --- Top chart: price ---
    ax1.axhline(floor, color=GREY, linestyle="--", linewidth=1.5)
    ax1.plot(spots, parity, color=ORANGE, linewidth=2)
    ax1.plot(spots, cb_price, color=BLUE, linewidth=2.2)
    if bond.call_price is not None:
        ax1.axhline(bond.call_price, color=GREY, linestyle=":", linewidth=1)

    # Today's dot
    ax1.scatter([mkt.spot], [today["price"]], s=60, color=BLUE,
                edgecolor="white", linewidth=2, zorder=5)
    ax1.annotate(f"Today: {today['price']:.2f}\n({today['premium_pct']:.0%} premium)",
                 (mkt.spot, today["price"]), xytext=(-12, 22),
                 textcoords="offset points", ha="right", fontsize=9, color=INK)

    # Labels written on the lines themselves (easier to read than a legend)
    x_lab = spots[-1]
    ax1.text(x_lab, cb_price[-1] + 6, "Convertible", color=INK, fontsize=9,
             ha="right", va="bottom", fontweight="bold")
    k = int(len(spots) * 0.72)
    ax1.text(spots[k] + 2, parity[k] - 14, "Parity (convert now)", color=INK_SOFT,
             fontsize=9, ha="left", va="top")
    ax1.text(spots[3], floor + 3, f"Bond floor {floor:.1f}", color=INK_SOFT,
             fontsize=9, va="bottom")
    if bond.call_price is not None:
        ax1.text(spots[3], bond.call_price + 3, f"Call price {bond.call_price:.0f}",
                 color=INK_SOFT, fontsize=8, va="bottom")

    # Shade the three "zones" lightly with text only (no colour noise)
    ymax = max(cb_price.max(), parity.max()) * 1.05
    ax1.set_ylim(0, ymax)
    for x, label in [(bond.conversion_price * 0.35, "Bond-like"),
                     (bond.conversion_price * 1.0, "Balanced"),
                     (bond.conversion_price * 2.0, "Equity-like")]:
        ax1.text(x, ymax * 0.97, label, ha="center", va="top",
                 fontsize=9, color=INK_SOFT, style="italic")

    ax1.set_ylabel("Price (per 100 face)", color=INK_SOFT)
    ax1.set_title(
        f"Convertible bond: {bond.coupon_rate:.1%} coupon, {bond.maturity:g}y, "
        f"conv. price {bond.conversion_price:.0f}   |   "
        f"vol {mkt.vol:.0%}, spread {mkt.credit_spread*1e4:.0f}bp",
        loc="left", fontsize=11, color=INK,
    )
    style_axis(ax1)

    # --- Bottom chart: delta ---
    ax2.plot(spots, delta_pct, color=BLUE, linewidth=2)
    ax2.axvline(mkt.spot, color=GREY, linewidth=1, linestyle=":")
    today_delta = today["delta"] / bond.conversion_ratio * 100
    ax2.scatter([mkt.spot], [today_delta], s=50, color=BLUE,
                edgecolor="white", linewidth=2, zorder=5)
    ax2.annotate(f"{today_delta:.0f}%", (mkt.spot, today_delta),
                 xytext=(8, -4), textcoords="offset points", fontsize=9, color=INK)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Delta (%)", color=INK_SOFT)
    ax2.set_xlabel("Share price", color=INK_SOFT)
    style_axis(ax2)

    fig.tight_layout()
    fig.savefig(save_as, dpi=150)
    print(f"Saved chart to {save_as}")
    plt.show()


if __name__ == "__main__":
    bond = ConvertibleBond(
        face=100, coupon_rate=0.02, coupon_freq=2, maturity=5,
        conversion_ratio=2.0,
        call_price=103, call_start=3.0,
    )
    mkt = Market(spot=40, vol=0.30, rate=0.04, credit_spread=0.03)
    make_chart(bond, mkt)
