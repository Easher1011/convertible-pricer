"""
Convertible Bond Pricer — Tsiveriotis-Fernandes (1998) binomial tree
=====================================================================

The big idea, in plain English
------------------------------
A convertible bond is two things glued together:
  1. A normal bond (pays coupons, gives your money back at the end), and
  2. An option to swap the bond for shares instead.

The tricky part is choosing the right discount rate:
  - The cash part (coupons, redemption) can fail if the company defaults,
    so we discount it with  risk-free rate + credit spread.
  - The share part doesn't depend on the company paying you cash,
    so we discount it with  the risk-free rate only.

Tsiveriotis-Fernandes handles this by keeping TWO values at every point
in the tree:
  - B = the "cash-only" part of the bond's value
  - E = the "equity" part of the bond's value
  Total convertible value V = B + E

We build a tree of possible share prices, start at maturity, and work
backwards to today. At each point the holder and issuer make their best
choice (convert, call, put).
"""

from dataclasses import dataclass, field
import math
import numpy as np
from scipy.linalg import solve_banded


# ---------------------------------------------------------------------
# 1. The bond's terms (everything you'd find in the prospectus)
# ---------------------------------------------------------------------
@dataclass
class ConvertibleBond:
    face: float = 100.0              # repaid at maturity
    coupon_rate: float = 0.02        # annual coupon, e.g. 0.02 = 2%
    coupon_freq: int = 2             # payments per year (2 = semi-annual)
    maturity: float = 5.0            # years to maturity
    conversion_ratio: float = 2.0    # shares you get per bond
    # Issuer call: from call_start onwards, the issuer can redeem at call_price
    call_price: float | None = None
    call_start: float = 0.0
    # Soft call: issuer can only call if parity is at least this % of face.
    # E.g. 1.30 = shares must be at 130% of the conversion price.
    # None = "hard" call (can call whenever it's worth it).
    # Real soft calls are tested over e.g. 20 of 30 trading days; we
    # approximate with today's share price, which is the standard shortcut.
    call_trigger: float | None = None
    # Holder put: on these dates (in years), holder can sell back at put_price
    put_price: float | None = None
    put_dates: list[float] = field(default_factory=list)

    @property
    def conversion_price(self) -> float:
        """Share price at which converting is worth exactly face value."""
        return self.face / self.conversion_ratio


# ---------------------------------------------------------------------
# 2. Market inputs
# ---------------------------------------------------------------------
@dataclass
class Market:
    spot: float = 40.0          # current share price
    vol: float = 0.30           # share price volatility (0.30 = 30%)
    rate: float = 0.04          # risk-free rate
    credit_spread: float = 0.03 # issuer's credit spread
    div_yield: float = 0.0      # dividend yield


# ---------------------------------------------------------------------
# 3. The pricer
# ---------------------------------------------------------------------
def price_convertible_tree(bond: ConvertibleBond, mkt: Market, steps: int = 500) -> dict:
    """
    Price a convertible bond on a binomial tree. Returns a dict with the total price and
    its split into cash (bond) and equity parts.
    """
    T = bond.maturity
    dt = T / steps

    # --- Tree set-up (Cox-Ross-Rubinstein) ---
    u = math.exp(mkt.vol * math.sqrt(dt))   # up move
    d = 1.0 / u                              # down move
    p = (math.exp((mkt.rate - mkt.div_yield) * dt) - d) / (u - d)  # up probability
    if not 0 < p < 1:
        raise ValueError("Tree probability out of range — increase steps.")

    disc_rf = math.exp(-mkt.rate * dt)                        # for equity part
    disc_risky = math.exp(-(mkt.rate + mkt.credit_spread) * dt)  # for cash part

    # --- Work out which tree steps have a coupon payment ---
    coupon_amt = bond.face * bond.coupon_rate / bond.coupon_freq
    n_coupons = int(round(T * bond.coupon_freq))
    coupon_steps = {}
    for k in range(1, n_coupons + 1):
        t = T - (k - 1) / bond.coupon_freq        # count back from maturity
        s = int(round(t / dt))
        coupon_steps[s] = coupon_steps.get(s, 0.0) + coupon_amt

    put_steps = {int(round(t / dt)) for t in bond.put_dates}

    # --- Share prices at maturity (all the end points of the tree) ---
    j = np.arange(steps + 1)
    S = mkt.spot * u ** (steps - j) * d ** j

    # --- At maturity: take the better of shares or cash ---
    redemption = bond.face + coupon_steps.get(steps, 0.0)
    conv_value = bond.conversion_ratio * S
    convert = conv_value > redemption
    E = np.where(convert, conv_value, 0.0)
    B = np.where(convert, 0.0, redemption)

    # --- Step backwards through the tree to today ---
    for i in range(steps - 1, -1, -1):
        t = i * dt
        S = mkt.spot * u ** (i - np.arange(i + 1)) * d ** np.arange(i + 1)

        # Value of holding on: average of the two next nodes, discounted
        E = disc_rf * (p * E[:-1] + (1 - p) * E[1:])
        B = disc_risky * (p * B[:-1] + (1 - p) * B[1:])

        V = B + E
        conv_value = bond.conversion_ratio * S

        # Holder put: holder can force the issuer to buy back
        if bond.put_price is not None and i in put_steps:
            put_now = bond.put_price > V
            B = np.where(put_now, bond.put_price, B)
            E = np.where(put_now, 0.0, E)
            V = B + E

        # Issuer call: issuer redeems if bond is worth more than call price.
        # Holder then picks the better of cash (call price) or shares.
        if bond.call_price is not None and t >= bond.call_start:
            called = V > bond.call_price
            if bond.call_trigger is not None:
                called &= conv_value >= bond.call_trigger * bond.face
            # Called, and cash is worth more than shares: holder takes the cash
            take_cash = called & (bond.call_price >= conv_value)
            B = np.where(take_cash, bond.call_price, B)
            E = np.where(take_cash, 0.0, E)
            # Called, and shares are worth more: holder is FORCED to convert now
            # (they lose the extra value of waiting — that's why issuers call)
            forced = called & (conv_value > bond.call_price)
            E = np.where(forced, conv_value, E)
            B = np.where(forced, 0.0, B)
            V = B + E

        # Voluntary conversion: if shares beat holding, convert
        convert = conv_value > V
        E = np.where(convert, conv_value, E)
        B = np.where(convert, 0.0, B)

        # Coupon paid on this date (cash part). Decisions above were made on
        # the value *excluding* today's coupon, which the holder gets anyway.
        if i in coupon_steps and i > 0:
            B = B + coupon_steps[i]

        # Remember the 3 values two steps in — used for delta and gamma below
        if i == 2:
            V2, S2 = (B + E).copy(), S.copy()

    price = float(B[0] + E[0])

    # Delta and gamma read straight off the tree (much smoother than
    # re-pricing with a bumped share price, which moves every node)
    if steps >= 2:
        d_up = (V2[0] - V2[1]) / (S2[0] - S2[1])
        d_dn = (V2[1] - V2[2]) / (S2[1] - S2[2])
        tree_delta = float((V2[0] - V2[2]) / (S2[0] - S2[2]))
        tree_gamma = float((d_up - d_dn) / ((S2[0] - S2[2]) / 2))
    else:
        tree_delta = tree_gamma = float("nan")

    return {
        "price": price,
        "delta": tree_delta,      # in shares
        "gamma": tree_gamma,
        "bond_part": float(B[0]),
        "equity_part": float(E[0]),
        "parity": bond.conversion_ratio * mkt.spot,     # value if converted now
        "premium_pct": price / (bond.conversion_ratio * mkt.spot) - 1,
    }


# ---------------------------------------------------------------------
# 3b. Finite-difference pricer (the default)
# ---------------------------------------------------------------------
# Same model, different engine. Instead of a tree of share prices that
# moves every time vol changes, we use a FIXED grid of share prices
# (e.g. 0, 0.4, 0.8, ... 200) and step backwards in time, solving the
# Black-Scholes equation for the cash part and the equity part separately.
#
# Why bother: on the tree, a change in vol shifts every node, and when a
# row of nodes crosses a fixed level (like "conversion value = call
# price") the price jumps. A fixed grid doesn't have that problem, so
# the price moves smoothly with vol and spread — which matters when we
# solve for implied vol.
#
# Method: Crank-Nicolson (accurate), with a few fully implicit steps at the
# start and after each coupon/event ("Rannacher smoothing") to stop the
# kinks in the payoff causing wiggles.

def _event_times(bond: ConvertibleBond):
    """Coupon dates (counted back from maturity), call start and put dates."""
    T = bond.maturity
    period = 1.0 / bond.coupon_freq
    coupons = {}
    k = 0
    while T - k * period > 1e-9:
        coupons[round(T - k * period, 10)] = bond.face * bond.coupon_rate / bond.coupon_freq
        k += 1
    events = set(coupons)
    if 0 < bond.call_start < T:
        events.add(round(bond.call_start, 10))
    for t in bond.put_dates:
        if 0 < t < T:
            events.add(round(t, 10))
    return coupons, events


def hazard_rate(S, mkt: Market, recovery: float = 0.4, hazard_power: float = 1.0,
                cap: float = 5.0):
    """
    Default intensity (chance of default per year) at each share price.

    Calibrated so that at TODAY's share price the bond's credit spread equals
    the spread input:  spread = hazard x (1 - recovery)  ("credit triangle").
    It then rises as the share price falls:
        hazard(S) = hazard_today x (spot / S) ^ hazard_power
    hazard_power = 0 means default risk doesn't depend on the share price.
    Capped (default 500%/yr) so it stays finite as the share price nears zero.
    """
    lam0 = max(mkt.credit_spread, 0.0) / (1 - recovery)
    S_safe = np.maximum(np.asarray(S, dtype=float), 1e-8)
    return np.minimum(lam0 * (mkt.spot / S_safe) ** hazard_power, cap)


def _operator(alpha, S, dS, drift, disc):
    """Grid version of the Black-Scholes operator: links to node below, itself, above."""
    beta = drift * S / (2 * dS)
    return alpha - beta, -2 * alpha - disc, alpha + beta


def _cn_step(V_next, dt, op, theta, top_value, src=None):
    """
    One step back in time (Crank-Nicolson when theta = 0.5, fully implicit at 1):
        (I - theta dt L) V = (I + (1 - theta) dt L) V_next + dt * src
    """
    lo, mid, up = op
    n = len(V_next)
    LV = mid * V_next
    LV[1:] += lo[1:] * V_next[:-1]
    LV[:-1] += up[:-1] * V_next[1:]
    rhs = V_next + (1 - theta) * dt * LV
    if src is not None:
        rhs = rhs + dt * src
    diag = 1 - theta * dt * mid
    lower = -theta * dt * lo
    upper = -theta * dt * up
    diag[-1], lower[-1], rhs[-1] = 1.0, 0.0, top_value   # top row: value fixed
    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, rhs)


def solve_fd_grid(bond: ConvertibleBond, mkt: Market, nt: int = 400, ns: int = 500,
                  model: str = "tf", recovery: float = 0.4, hazard_power: float = 1.0,
                  keep_surface: bool = False, s_max: float | None = None):
    """
    Solve on the whole share-price grid, stepping back from maturity to today.

    model = "tf"  -> Tsiveriotis-Fernandes: fixed credit spread, value split into
                     a cash part (discounted at rate + spread) and an equity part
                     (discounted at the risk-free rate).
    model = "jtd" -> Jump-to-default: default can happen at any moment, more
                     likely when the share price is low. On default the shares
                     drop to zero and the bond pays recovery x face.

    Returns (S, V, B, E): share-price grid, total value, cash part, equity part
    (B and E are None for the jump-to-default model, where there's no split).
    With keep_surface=True also returns (times, surface, surface_B): the value
    (and its cash part, fixed-spread model only) at every time step, rows
    ordered from today to maturity, recorded just after any coupon is paid.
    The simulator uses this.
    """
    T = bond.maturity
    CR = bond.conversion_ratio
    if s_max is None:
        s_max = 4.0 * max(mkt.spot, bond.conversion_price)
    S = np.linspace(0.0, s_max, ns + 1)
    dS = S[1] - S[0]

    coupons, events = _event_times(bond)
    times = np.unique(np.round(np.concatenate([np.linspace(0, T, nt + 1),
                                               np.array(sorted(events))]), 10))
    put_set = {round(t, 10) for t in bond.put_dates}
    jtd = model == "jtd"

    # --- Payoff at maturity ---
    redemption = bond.face + coupons.get(round(T, 10), 0.0)
    conv = CR * S
    convert = conv > redemption
    E = np.where(convert, conv, 0.0)
    B = np.where(convert, 0.0, redemption)
    if jtd:                                         # no split in this model
        B, E = B + E, np.zeros_like(S)

    # --- Operators (fixed through time) ---
    alpha = 0.5 * mkt.vol**2 * S**2 / dS**2
    carry = mkt.rate - mkt.div_yield
    if jtd:
        lam = hazard_rate(S, mkt, recovery, hazard_power)
        # Shares must drift up by the hazard rate to make up for the default jump
        op_B = _operator(alpha, S, dS, carry + lam, mkt.rate + lam)
        src_B = lam * recovery * bond.face          # what you get paid on default
    else:
        op_E = _operator(alpha, S, dS, carry, mkt.rate)
        op_B = _operator(alpha, S, dS, carry, mkt.rate + mkt.credit_spread)
        src_B = None

    surface = [B + E] if keep_surface else None
    surface_B = [B.copy()] if (keep_surface and not jtd) else None
    surface_t = [T]
    implicit_left = 4                               # Rannacher steps remaining
    for n in range(len(times) - 2, -1, -1):
        t, dt = times[n], times[n + 1] - times[n]
        theta = 1.0 if implicit_left > 0 else 0.5
        implicit_left -= 1

        if jtd:
            B = _cn_step(B, dt, op_B, theta, CR * s_max, src_B)
        else:
            E = _cn_step(E, dt, op_E, theta, CR * s_max)
            B = _cn_step(B, dt, op_B, theta, 0.0, src_B)
        V = B + E
        tk = round(t, 10)

        # Holder put
        if bond.put_price is not None and tk in put_set:
            put_now = bond.put_price > V
            B = np.where(put_now, bond.put_price, B)
            E = np.where(put_now, 0.0, E)
            V = B + E

        # Issuer call (with optional soft-call trigger)
        if bond.call_price is not None and t >= bond.call_start - 1e-10:
            called = V > bond.call_price
            if bond.call_trigger is not None:
                called &= conv >= bond.call_trigger * bond.face
            take_cash = called & (bond.call_price >= conv)
            forced = called & (conv > bond.call_price)
            B = np.where(take_cash, bond.call_price, np.where(forced, 0.0, B))
            E = np.where(take_cash, 0.0, np.where(forced, conv, E))
            V = B + E

        # Voluntary conversion
        convert = conv > V
        E = np.where(convert, conv, E)
        B = np.where(convert, 0.0, B)

        # The simulator needs values just AFTER any coupon is paid (it pays the
        # coupon into cash separately), so record the surface before adding it
        if keep_surface:
            surface.append(B + E)
            surface_t.append(t)
            if surface_B is not None:
                surface_B.append(B.copy())

        # Coupon paid today (holder gets it whatever they decided)
        if 0 < t and tk in coupons and tk < round(T, 10):
            B = B + coupons[tk]
            if keep_surface:
                # ...and a row a hair BEFORE the coupon date, still including it,
                # so the value drops in one step exactly when the coupon is paid
                surface.append(B + E)
                surface_t.append(t - 1e-9)
                if surface_B is not None:
                    surface_B.append(B.copy())

        if jtd:
            B, E = B + E, np.zeros_like(S)
        if tk in events:
            implicit_left = 2                       # smooth again after a jump

    V = B + E
    out_B, out_E = (None, None) if jtd else (B, E)
    if keep_surface:
        surf_B = np.array(surface_B[::-1]) if surface_B is not None else None
        return (S, V, out_B, out_E, np.array(surface_t[::-1]),
                np.array(surface[::-1]), surf_B)
    return S, V, out_B, out_E


def price_convertible_fd(bond: ConvertibleBond, mkt: Market, steps: int = 400,
                         ns: int = 500, model: str = "tf", **model_kw) -> dict:
    """Finite-difference price, same outputs as the tree version."""
    S, V, B, E = solve_fd_grid(bond, mkt, steps, ns, model=model, **model_kw)
    dV = np.gradient(V, S)
    d2V = np.gradient(dV, S)
    price = float(np.interp(mkt.spot, S, V))
    return {
        "price": price,
        "delta": float(np.interp(mkt.spot, S, dV)),
        "gamma": float(np.interp(mkt.spot, S, d2V)),
        "bond_part": float(np.interp(mkt.spot, S, B)) if B is not None else None,
        "equity_part": float(np.interp(mkt.spot, S, E)) if E is not None else None,
        "parity": bond.conversion_ratio * mkt.spot,
        "premium_pct": price / (bond.conversion_ratio * mkt.spot) - 1,
    }


def price_convertible(bond: ConvertibleBond, mkt: Market, steps: int = 400,
                      method: str = "fd", model: str = "tf", **model_kw) -> dict:
    """
    Price a convertible bond.
    method = "fd"   -> finite-difference grid (default, smooth)
    method = "tree" -> binomial tree (simpler, a bit noisy with calls)
    model  = "tf"   -> fixed credit spread (Tsiveriotis-Fernandes)
    model  = "jtd"  -> jump-to-default; extra settings: recovery, hazard_power
    """
    if method == "tree":
        if model != "tf":
            raise ValueError("The tree only supports the fixed-spread model.")
        return price_convertible_tree(bond, mkt, steps)
    return price_convertible_fd(bond, mkt, steps, model=model, **model_kw)


# ---------------------------------------------------------------------
# 4. Greeks — how the price reacts to changes (simple "bump and reprice")
# ---------------------------------------------------------------------
def greeks(bond: ConvertibleBond, mkt: Market, steps: int = 400,
           method: str = "fd", model: str = "tf", **model_kw) -> dict:
    """
    Delta and gamma come straight from the grid (or tree).
    The others use "bump and reprice": nudge one input, price again, compare.

    On the tree, vega needs a wide +/-5 vol bump: changing vol reshapes the
    tree and adds noise that a small bump would be swamped by. The grid
    doesn't have that problem, so it uses a normal +/-1 vol bump.
    """
    def px(**changes):
        m = Market(**{**mkt.__dict__, **changes})
        return price_convertible(bond, m, steps, method, model, **model_kw)["price"]

    res = price_convertible(bond, mkt, steps, method, model, **model_kw)
    vb = 0.05 if method == "tree" else 0.01
    bp = 0.0001

    return {
        # Delta in shares: how many shares the bond behaves like
        "delta_shares": res["delta"],
        # Delta as % of conversion ratio (0% = pure bond, 100% = pure shares)
        "delta_pct": res["delta"] / bond.conversion_ratio,
        "gamma": res["gamma"],
        # Vega: price change per +1 vol point
        "vega": (px(vol=mkt.vol + vb) - px(vol=mkt.vol - vb)) / (2 * vb * 100),
        # Credit sensitivity: price change for +1bp of spread
        "cs01": (px(credit_spread=mkt.credit_spread + bp)
                 - px(credit_spread=mkt.credit_spread - bp)) / 2,
        # Rate sensitivity: price change for +1bp of rates
        "rho_01": (px(rate=mkt.rate + bp) - px(rate=mkt.rate - bp)) / 2,
    }


# ---------------------------------------------------------------------
# 5. Working backwards: what input makes the model match the market?
# ---------------------------------------------------------------------
def accrued_interest(bond: ConvertibleBond) -> float:
    """
    Coupon earned since the last payment date. Market prices are quoted
    "clean" (without this), but the model gives the full "dirty" value,
    so we add this to the market price before comparing.
    Coupon dates are counted back from maturity in whole periods.
    """
    period = 1.0 / bond.coupon_freq
    time_to_next = bond.maturity % period
    if time_to_next < 1e-9:
        time_to_next = period            # today is a coupon date
    fraction_elapsed = 1 - time_to_next / period
    return bond.face * bond.coupon_rate / bond.coupon_freq * fraction_elapsed


def _solve(fn, target, lo, hi, tol=1e-4, max_iter=60, scan=12):
    """
    Find x in [lo, hi] where fn(x) = target.

    First a coarse scan from low to high, to find the FIRST place the price
    crosses the target. (Under the jump-to-default model, price isn't always
    one-directional in spread: very high default risk pushes up the value of
    the conversion option. We want the sensible, lower answer.)
    Then bisection: keep halving that bracket until it's tiny.
    Returns None if the target is outside what the range can produce.
    """
    xs = np.linspace(lo, hi, scan + 1)
    fs = [fn(x) - target for x in xs]
    bracket = None
    for i in range(scan):
        if fs[i] == 0:
            return float(xs[i])
        if fs[i] * fs[i + 1] < 0:
            bracket = i
            break
    if bracket is None:
        return None
    lo, hi = xs[bracket], xs[bracket + 1]
    f_lo, f_hi = fs[bracket], fs[bracket + 1]
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        f_mid = fn(mid) - target
        if abs(hi - lo) < tol:
            break
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def implied_vol(bond: ConvertibleBond, mkt: Market, dirty_price: float,
                steps: int = 400, method: str = "fd", model: str = "tf",
                **model_kw) -> float | None:
    """Volatility that makes the model price equal the market (dirty) price."""
    def px(v):
        return price_convertible(bond, Market(**{**mkt.__dict__, "vol": v}),
                                 steps, method, model, **model_kw)["price"]
    return _solve(px, dirty_price, 0.01, 1.50)


def implied_spread(bond: ConvertibleBond, mkt: Market, dirty_price: float,
                   steps: int = 400, method: str = "fd", model: str = "tf",
                   **model_kw) -> float | None:
    """Credit spread that makes the model price equal the market (dirty) price."""
    def px(s):
        return price_convertible(bond, Market(**{**mkt.__dict__, "credit_spread": s}),
                                 steps, method, model, **model_kw)["price"]
    lo = 0.0 if model == "jtd" else -0.01     # default risk can't be negative
    return _solve(px, dirty_price, lo, 0.40)


def realised_vol(prices) -> float:
    """
    Annualised volatility from a list of daily closing prices:
    standard deviation of daily log returns x sqrt(252 trading days).
    """
    p = np.asarray(prices, dtype=float)
    rets = np.diff(np.log(p))
    return float(np.std(rets, ddof=1) * math.sqrt(252))


# ---------------------------------------------------------------------
# 6. Run an example when you press "Run" in VS Code
# ---------------------------------------------------------------------
if __name__ == "__main__":
    bond = ConvertibleBond(
        face=100, coupon_rate=0.02, coupon_freq=2, maturity=5,
        conversion_ratio=2.0,             # conversion price = 50
        call_price=103, call_start=3.0,   # callable after 3 years
    )
    mkt = Market(spot=40, vol=0.30, rate=0.04, credit_spread=0.03)

    res = price_convertible(bond, mkt)
    g = greeks(bond, mkt)

    print(f"Conversion price : {bond.conversion_price:.2f}")
    print(f"Parity           : {res['parity']:.2f}")
    print(f"Convertible price: {res['price']:.2f}")
    print(f"  cash part      : {res['bond_part']:.2f}")
    print(f"  equity part    : {res['equity_part']:.2f}")
    print(f"Premium to parity: {res['premium_pct']:.1%}")
    print()
    for k, v in g.items():
        print(f"{k:13s}: {v:.4f}")
