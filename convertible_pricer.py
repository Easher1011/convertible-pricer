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


def solve_fd_grid(bond: ConvertibleBond, mkt: Market, nt: int = 400, ns: int = 500):
    """
    Solve on the whole share-price grid. Returns (S, V, B, E) at today:
    the grid of share prices, total value, cash part and equity part.
    """
    T = bond.maturity
    CR = bond.conversion_ratio
    s_max = 4.0 * max(mkt.spot, bond.conversion_price)
    S = np.linspace(0.0, s_max, ns + 1)
    dS = S[1] - S[0]

    coupons, events = _event_times(bond)
    times = np.unique(np.round(np.concatenate([np.linspace(0, T, nt + 1),
                                               np.array(sorted(events))]), 10))
    put_set = {round(t, 10) for t in bond.put_dates}

    # --- Payoff at maturity ---
    redemption = bond.face + coupons.get(round(T, 10), 0.0)
    conv = CR * S
    convert = conv > redemption
    E = np.where(convert, conv, 0.0)
    B = np.where(convert, 0.0, redemption)

    # --- The Black-Scholes operator on the grid (one row per share price) ---
    alpha = 0.5 * mkt.vol**2 * S**2 / dS**2
    beta = (mkt.rate - mkt.div_yield) * S / (2 * dS)
    lo, up = alpha - beta, alpha + beta             # links to the node below / above

    def L_apply(V, disc):
        """Black-Scholes operator applied to V, for a given discount rate."""
        out = (-2 * alpha - disc) * V
        out[1:] += lo[1:] * V[:-1]
        out[:-1] += up[:-1] * V[1:]
        return out

    def step(V_next, dt, disc, theta, top_value):
        """One step back in time: (I - theta dt L) V = (I + (1-theta) dt L) V_next."""
        rhs = V_next + (1 - theta) * dt * L_apply(V_next, disc)
        diag = 1 + theta * dt * (2 * alpha + disc)
        lower = -theta * dt * lo
        upper = -theta * dt * up
        # Top row: fix the value (deep in the money = converted)
        diag[-1], lower[-1], rhs[-1] = 1.0, 0.0, top_value
        # S = 0 row has no link below
        ab = np.zeros((3, ns + 1))
        ab[0, 1:] = upper[:-1]
        ab[1, :] = diag
        ab[2, :-1] = lower[1:]
        return solve_banded((1, 1), ab, rhs)

    implicit_left = 4                               # Rannacher steps remaining
    for n in range(len(times) - 2, -1, -1):
        t, dt = times[n], times[n + 1] - times[n]
        theta = 1.0 if implicit_left > 0 else 0.5
        implicit_left -= 1

        E = step(E, dt, mkt.rate, theta, CR * s_max)
        B = step(B, dt, mkt.rate + mkt.credit_spread, theta, 0.0)
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

        # Coupon paid today (holder gets it whatever they decided)
        if 0 < t and tk in coupons and tk < round(T, 10):
            B = B + coupons[tk]

        if tk in events:
            implicit_left = 2                       # smooth again after a jump

    return S, B + E, B, E


def price_convertible_fd(bond: ConvertibleBond, mkt: Market, steps: int = 400,
                         ns: int = 500) -> dict:
    """Finite-difference price, same outputs as the tree version."""
    S, V, B, E = solve_fd_grid(bond, mkt, steps, ns)
    dV = np.gradient(V, S)
    d2V = np.gradient(dV, S)
    price = float(np.interp(mkt.spot, S, V))
    return {
        "price": price,
        "delta": float(np.interp(mkt.spot, S, dV)),
        "gamma": float(np.interp(mkt.spot, S, d2V)),
        "bond_part": float(np.interp(mkt.spot, S, B)),
        "equity_part": float(np.interp(mkt.spot, S, E)),
        "parity": bond.conversion_ratio * mkt.spot,
        "premium_pct": price / (bond.conversion_ratio * mkt.spot) - 1,
    }


def price_convertible(bond: ConvertibleBond, mkt: Market, steps: int = 400,
                      method: str = "fd") -> dict:
    """
    Price a convertible bond.
    method = "fd"   -> finite-difference grid (default, smooth)
    method = "tree" -> binomial tree (simpler, a bit noisy with calls)
    """
    if method == "tree":
        return price_convertible_tree(bond, mkt, steps)
    return price_convertible_fd(bond, mkt, steps)



# ---------------------------------------------------------------------
# 4. Greeks — how the price reacts to changes (simple "bump and reprice")
# ---------------------------------------------------------------------
def greeks(bond: ConvertibleBond, mkt: Market, steps: int = 400,
           method: str = "fd") -> dict:
    """
    Delta and gamma come straight from the grid (or tree).
    The others use "bump and reprice": nudge one input, price again, compare.

    On the tree, vega needs a wide +/-5 vol bump: changing vol reshapes the
    tree and adds noise that a small bump would be swamped by. The grid
    doesn't have that problem, so it uses a normal +/-1 vol bump.
    """
    def px(**changes):
        m = Market(**{**mkt.__dict__, **changes})
        return price_convertible(bond, m, steps, method)["price"]

    res = price_convertible(bond, mkt, steps, method)
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


def _solve(fn, target, lo, hi, tol=1e-4, max_iter=60):
    """
    Bisection: keep halving the range [lo, hi] until fn(x) hits target.
    Slower than fancier methods but never gets confused by tree noise.
    Returns None if the target is outside what the range can produce.
    """
    f_lo, f_hi = fn(lo) - target, fn(hi) - target
    if f_lo * f_hi > 0:
        return None
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
                steps: int = 400, method: str = "fd") -> float | None:
    """Volatility that makes the model price equal the market (dirty) price."""
    def px(v):
        return price_convertible(bond, Market(**{**mkt.__dict__, "vol": v}),
                                 steps, method)["price"]
    return _solve(px, dirty_price, 0.01, 1.50)


def implied_spread(bond: ConvertibleBond, mkt: Market, dirty_price: float,
                   steps: int = 400, method: str = "fd") -> float | None:
    """Credit spread that makes the model price equal the market (dirty) price."""
    def px(s):
        return price_convertible(bond, Market(**{**mkt.__dict__, "credit_spread": s}),
                                 steps, method)["price"]
    return _solve(px, dirty_price, -0.01, 0.40)


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
