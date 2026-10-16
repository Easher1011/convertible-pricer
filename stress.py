"""
Stress testing
==============

"What happens to my position if the shares move X% and spreads move Y bp,
right now?" — answered for a whole grid of moves at once.

Two versions:
  - Unhedged: you just own the convertible.
  - Delta-hedged: you own the convertible AND are short the shares
    (delta-worth), like a convertible arbitrage fund. The share hedge cancels
    small share moves, so what's left shows gamma (big moves help you) and
    credit risk (spread widening hurts you).

All P&L is per 100 face of bonds, and assumes the move happens instantly
(no time passes).
"""

import numpy as np

from convertible_pricer import ConvertibleBond, Market, price_convertible, solve_fd_grid


def _value_at(bond, mkt, spots, steps, model="tf", **model_kw):
    """Price the bond at several share prices using one grid solve."""
    S, V, _, _ = solve_fd_grid(bond, mkt, steps, model=model, **model_kw)
    return np.interp(spots, S, V)


def stress_grid(bond: ConvertibleBond, mkt: Market, spot_shocks, other_shocks,
                shock_type: str = "spread", hedged: bool = True, steps: int = 300,
                model: str = "tf", **model_kw):
    """
    P&L grid.
      spot_shocks:  share price moves as fractions, e.g. [-0.3, ..., 0.3]
      other_shocks: spread moves in decimals (0.01 = 100bp) if shock_type="spread",
                    or vol moves in decimals (0.05 = 5 vol pts) if shock_type="vol"
    Returns a 2D array: rows = other_shocks, columns = spot_shocks.
    """
    base = price_convertible(bond, mkt, steps, model=model, **model_kw)
    p0, delta0 = base["price"], base["delta"]
    spot_shocks = np.asarray(spot_shocks)
    new_spots = mkt.spot * (1 + spot_shocks)

    # The share hedge: short delta0 shares, so it gains when shares fall
    hedge_pnl = -delta0 * mkt.spot * spot_shocks if hedged else 0.0

    grid = []
    for shock in other_shocks:
        changes = ({"credit_spread": max(mkt.credit_spread + shock, -0.005)}
                   if shock_type == "spread"
                   else {"vol": max(mkt.vol + shock, 0.01)})
        m = Market(**{**mkt.__dict__, **changes})
        values = _value_at(bond, m, new_spots, steps, model, **model_kw)
        grid.append(values - p0 + hedge_pnl)
    return np.array(grid)


if __name__ == "__main__":
    bond = ConvertibleBond(call_price=103, call_start=3.0)
    mkt = Market()
    spots = [-0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5]
    sprs = [-0.01, 0, 0.01, 0.03, 0.05]
    for hedged in (False, True):
        g = stress_grid(bond, mkt, spots, sprs, "spread", hedged)
        print("HEDGED" if hedged else "UNHEDGED", "rows = spread shock (bp), cols = share move")
        print("        " + "".join(f"{s:>8.0%}" for s in spots))
        for sh, row in zip(sprs, g):
            print(f"{sh*1e4:>+6.0f}bp" + "".join(f"{v:>8.2f}" for v in row))
        print()
