"""
Convertible arbitrage simulator
===============================

The classic convertible arbitrage trade:
  - BUY the convertible bond
  - SHORT shares against it (delta-worth), so small share moves cancel out
  - REBALANCE the share hedge as delta changes

You then make money when the shares move around MORE than the bond's price
assumed (realised vol > implied vol), and lose when they move LESS, or
when credit spreads widen, or when borrowing shares gets expensive.

This file simulates that trade along many random share-price paths and
splits the P&L into where it came from:

  vol     - profit from share swings MINUS what the bond's price charged
            for them. Positive on average only if realised vol > implied vol.
            (Desk name: "gamma P&L net of theta".)
  credit  - the credit spread you earn for taking the issuer's default risk.
  carry   - stock borrow fee and anything left from financing, dividends
            and coupons (these mostly cancel out by design).
  shock   - repricing when spreads / implied vol jump in a stress scenario.
  default - loss if the issuer defaults (jump-to-default model only).
  other   - the rest: hedge mismatch between rebalances, and big jumps the
            simple gamma formula doesn't fully capture.

A useful check: with zero borrow fee, no defaults priced or simulated, and
realised vol = implied vol, the average P&L comes out at zero.

All P&L is per 100 face of bonds. Speed trick: we solve the pricing grid
ONCE for every day and every share price, then just look values up.
"""

from dataclasses import dataclass, field
import numpy as np

from convertible_pricer import ConvertibleBond, Market, solve_fd_grid, hazard_rate

TRADING_DAYS = 252


@dataclass
class Scenario:
    """A one-off stress event on a given day. Everything defaults to 'no change'."""
    day: int = 60
    share_jump: float = 0.0       # e.g. -0.30 = shares drop 30% that day
    spread_jump: float = 0.0      # e.g. 0.05 = spreads +500bp from that day on
    vol_jump: float = 0.0         # implied (pricing) vol change, e.g. 0.10 = +10 pts
    borrow_jump: float = 0.0      # extra stock borrow fee from that day on


SCENARIOS = {
    "None": None,
    "Credit crunch (2008-style)": Scenario(day=60, share_jump=-0.30, spread_jump=0.05,
                                           vol_jump=0.15, borrow_jump=0.03),
    "Downgrade while shares rally (GM 2005-style)": Scenario(day=60, share_jump=0.15,
                                                             spread_jump=0.04),
    "Implied vol crush": Scenario(day=60, vol_jump=-0.10),
}


@dataclass
class SimSettings:
    horizon: float = 1.0          # years to run the trade
    n_paths: int = 500            # number of simulated share-price paths
    realised_vol: float = 0.35    # how volatile the shares ACTUALLY are
    rebalance_every: int = 1      # days between hedge rebalances
    borrow_fee: float = 0.005     # annual fee to borrow shares for the short
    seed: int = 42
    scenario: Scenario | None = None
    simulate_default: bool = True  # only used with the jump-to-default model


class _Surface:
    """Bond value, delta and gamma for any (time, share price), by lookup."""

    def __init__(self, bond, mkt, s_max, nt, model, model_kw):
        S, _, _, _, times, surf, surf_B = solve_fd_grid(
            bond, mkt, nt, 600, model=model, keep_surface=True, s_max=s_max, **model_kw)
        self.S, self.times, self.V, self.B = S, times, surf, surf_B
        self.mkt = mkt
        self.dS = S[1] - S[0]
        self.D = np.gradient(surf, S, axis=1)
        self.G = np.gradient(self.D, S, axis=1)
        self.CR = bond.conversion_ratio

    def _lookup(self, arr, t, s, extrapolate_slope=None):
        # time: blend the two nearest rows
        j = int(np.clip(np.searchsorted(self.times, t) - 1, 0, len(self.times) - 2))
        w_t = (t - self.times[j]) / (self.times[j + 1] - self.times[j])
        row = (1 - w_t) * arr[j] + w_t * arr[j + 1]
        # share price: straight-line between grid points (grid is evenly spaced)
        x = np.clip(s / self.dS, 0, len(self.S) - 1 - 1e-9)
        i = x.astype(int)
        w = x - i
        val = (1 - w) * row[i] + w * row[i + 1]
        if extrapolate_slope is not None:          # beyond the grid: behaves like shares
            beyond = s > self.S[-1]
            val = np.where(beyond, row[-1] + extrapolate_slope * (s - self.S[-1]), val)
        return val

    def value(self, t, s):
        return self._lookup(self.V, t, s, extrapolate_slope=self.CR)

    def delta(self, t, s):
        d = self._lookup(self.D, t, s)
        return np.where(s > self.S[-1], self.CR, d)

    def cash_part(self, t, s):
        b = self._lookup(self.B, t, s)
        return np.where(s > self.S[-1], 0.0, b)

    def gamma(self, t, s):
        g = self._lookup(self.G, t, s)
        return np.where(s > self.S[-1], 0.0, g)


def run_simulation(bond: ConvertibleBond, mkt: Market, sim: SimSettings,
                   model: str = "tf", nt: int = 400, **model_kw) -> dict:
    """
    Run the delta-hedged trade over many paths.
    Returns per-day cumulative P&L by component (for every path), the share
    paths, and a summary.
    """
    rng = np.random.default_rng(sim.seed)
    n_days = int(round(min(sim.horizon, bond.maturity - 1e-6) * TRADING_DAYS))
    dt = 1.0 / TRADING_DAYS
    N = sim.n_paths
    sc = sim.scenario

    # Pricing grids: one for normal times, one for after the stress event
    s_max = 6.0 * max(mkt.spot, bond.conversion_price)
    base = _Surface(bond, mkt, s_max, nt, model, model_kw)
    if sc and (sc.spread_jump or sc.vol_jump):
        m2 = Market(**{**mkt.__dict__,
                       "credit_spread": max(mkt.credit_spread + sc.spread_jump, 0.0),
                       "vol": max(mkt.vol + sc.vol_jump, 0.01)})
        stressed = _Surface(bond, m2, s_max, nt, model, model_kw)
    else:
        m2, stressed = mkt, base

    # Coupon payment times (to pay coupons into the cash account)
    period = 1.0 / bond.coupon_freq
    coupon = bond.face * bond.coupon_rate / bond.coupon_freq
    coupon_times = [bond.maturity - k * period
                    for k in range(int(bond.maturity * bond.coupon_freq) + 1)
                    if bond.maturity - k * period > 0]

    # --- Starting positions ---
    S = np.full(N, mkt.spot)
    V = base.value(0.0, S)
    h = base.delta(0.0, S)                  # shares shorted
    cash = -V + h * S                       # paid for bond, received for short sale
    alive = np.ones(N, dtype=bool)          # False once the issuer has defaulted
    bond_held = np.ones(N, dtype=bool)

    names = ["vol", "credit", "carry", "shock", "default", "other"]
    cum = {k: np.zeros((n_days + 1, N)) for k in names}
    paths = np.zeros((n_days + 1, N))
    paths[0] = S
    surf = base
    borrow = sim.borrow_fee
    jtd = model == "jtd"
    recovery = model_kw.get("recovery", 0.4)
    hazard_power = model_kw.get("hazard_power", 1.0)

    for d in range(1, n_days + 1):
        t0, t1 = round((d - 1) * dt, 12), round(d * dt, 12)
        step = {k: np.zeros(N) for k in names}

        # 1. Stress event: reprice instantly on the new spread / vol
        if sc and d == sc.day:
            if stressed is not base:
                step["shock"] += stressed.value(t0, S) - surf.value(t0, S)
                surf = stressed
            borrow = sim.borrow_fee + sc.borrow_jump
        m_now = surf.mkt

        # 2. Cash flows over the day: interest on cash, borrow fee and
        #    dividends on the short, coupons received
        cash_flow = cash * mkt.rate * dt - h * S * (borrow + mkt.div_yield) * dt
        for tc in coupon_times:
            # small tolerance: day 126 is 0.49999999 years in computer arithmetic
            if t0 + 1e-9 < tc <= t1 + 1e-9:
                cash_flow = cash_flow + coupon
        cash = cash + cash_flow

        # 3. Default intensity today (jump-to-default model only)
        lam = hazard_rate(S, m_now, recovery, hazard_power) if jtd else np.zeros(N)

        # 4. Share price move. Under jump-to-default the shares drift up by the
        #    hazard rate (that's what compensates shareholders for default risk).
        z = rng.standard_normal(N)
        growth = np.exp((mkt.rate - mkt.div_yield + lam - 0.5 * sim.realised_vol**2) * dt
                        + sim.realised_vol * np.sqrt(dt) * z)
        if sc and d == sc.day:
            growth = growth * (1 + sc.share_jump)
        S_new = S * growth

        defaulted = np.zeros(N, dtype=bool)
        if jtd and sim.simulate_default:
            defaulted = alive & (rng.random(N) < 1 - np.exp(-lam * dt))

        # 5. Revalue and split the P&L
        V0 = surf.value(t0, S)
        V_new = surf.value(t1, S_new)
        dS = S_new - S
        G = surf.gamma(t0, S)
        theta = surf.value(t1, S) - V0
        gamma_full = 0.5 * G * dS**2
        gamma_cost = 0.5 * G * m_now.vol**2 * S**2 * dt      # what the price assumed
        if jtd:
            # Fair payment for expected default loss on the hedged position
            credit = lam * (V0 - recovery * bond.face - h * S) * dt
        else:
            credit = m_now.credit_spread * surf.cash_part(t0, S) * dt
        vol = gamma_full - gamma_cost
        carry = theta + gamma_cost - credit + cash_flow
        other = (V_new - V0) - h * dS - theta - gamma_full

        default_pnl = (recovery * bond.face - V0) + h * S   # bond recovers, shares -> 0
        step["vol"] += np.where(defaulted, 0.0, vol)
        step["credit"] += np.where(defaulted, 0.0, credit)
        step["carry"] += np.where(defaulted, cash_flow, carry)
        step["other"] += np.where(defaulted, 0.0, other)
        step["default"] += np.where(defaulted, default_pnl, 0.0)

        # Paths that defaulted on an earlier day: nothing more happens
        for k in names:
            step[k] = np.where(alive, step[k], 0.0)
            cum[k][d] = cum[k][d - 1] + step[k]

        cash = np.where(defaulted, cash + recovery * bond.face, cash)
        alive = alive & ~defaulted
        S = np.where(alive, S_new, 0.0)
        paths[d] = S

        # 6. Rebalance the hedge (trading shares changes cash, not P&L)
        if d % sim.rebalance_every == 0:
            h_new = np.where(alive, surf.delta(t1, S), 0.0)
            cash = cash + (h_new - h) * S
            h = h_new
        h = np.where(alive, h, 0.0)

    total = sum(cum[k] for k in names)
    final = total[-1]
    summary = {
        "mean": float(final.mean()),
        "median": float(np.median(final)),
        "std": float(final.std()),
        "p5": float(np.percentile(final, 5)),
        "p95": float(np.percentile(final, 95)),
        "pct_profitable": float((final > 0).mean()),
        "pct_defaulted": float((~alive).mean()),
        "by_component": {k: float(cum[k][-1].mean()) for k in names},
        "start_price": float(base.value(0.0, np.array([mkt.spot]))[0]),
        "start_delta": float(base.delta(0.0, np.array([mkt.spot]))[0]),
    }
    return {"days": np.arange(n_days + 1), "paths": paths, "cum": cum,
            "total": total, "summary": summary}


if __name__ == "__main__":
    bond = ConvertibleBond(call_price=103, call_start=3.0)
    mkt = Market(vol=0.30)
    for rv in (0.20, 0.30, 0.40):
        out = run_simulation(bond, mkt, SimSettings(realised_vol=rv, n_paths=1000))
        s = out["summary"]
        comp = ", ".join(f"{k} {v:+.2f}" for k, v in s["by_component"].items())
        print(f"realised {rv:.0%} vs implied 30%: mean P&L {s['mean']:+.2f} "
              f"(5%-95%: {s['p5']:+.2f} to {s['p95']:+.2f}), "
              f"{s['pct_profitable']:.0%} profitable | {comp}")
