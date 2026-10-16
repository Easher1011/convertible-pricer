"""
Monte Carlo pricer with a path-dependent soft call
==================================================

Real soft calls say something like: "the issuer may call only if the share
price has closed above 130% of the conversion price on at least 20 of the
last 30 trading days". Whether the call is allowed depends on the PATH the
share price took, not just where it is today. The finite-difference grid
can't see the path, so it approximates with "share price above 130% today".

This file prices the real clause by simulation (jump-to-default model:
default is more likely when the shares are low; on default the shares go
to zero and the bond pays recovery):

1. Simulate many daily share-price paths, with defaults.
2. Along each path, count the days above the trigger in the last 30.
3. Work backwards from maturity. At each decision date, estimate the value
   of holding on by regressing the realised future value on the share price
   and trigger count (Longstaff-Schwartz). Then apply the same rules as the
   grid: holder puts if that pays more; issuer calls when the trigger is met
   and calling is cheaper; holder converts if shares are worth more.
   (With no dividend, early voluntary conversion is never optimal, so it is
   switched off; that removes the regression's noise from that decision.)

The accuracy trick (control variate):
Monte Carlo is noisy, and the regression step adds a small bias. Both are
almost identical whether the trigger is "today only" or "20 of 30", as long
as we use the SAME simulated paths. So we price both on the same paths and
keep only the DIFFERENCE, which is precise. Then:

    price (20 of 30) = finite-difference price (today only)
                       + Monte Carlo difference (20 of 30 minus today only)
"""

from dataclasses import dataclass
import numpy as np

from convertible_pricer import ConvertibleBond, Market, hazard_rate, price_convertible

TRADING_DAYS = 252


@dataclass
class TriggerRule:
    days_needed: int = 20     # days above the trigger...
    window: int = 30          # ...within this many most recent trading days


class GridPolicy:
    """
    Decision rule taken from the finite-difference solution (today-only trigger).
    value(t, S) is the grid's value of HOLDING ON at that date and share price,
    before any decision. Monte Carlo uses it exactly where Longstaff-Schwartz
    would use its regression estimate: holder puts if the put pays more,
    issuer calls (when the path's trigger allows) if holding is worth more
    than the call price, holder converts if shares are worth more.
    """

    def __init__(self, bond, mkt, recovery, hazard_power, nt=600, ns=1200):
        from convertible_pricer import solve_fd_grid
        S, _, _, _, times, surf, _ = solve_fd_grid(
            bond, mkt, nt, ns, model="jtd", recovery=recovery,
            hazard_power=hazard_power, keep_surface=True,
            s_max=6.0 * max(mkt.spot, bond.conversion_price),
            surface_kind="continuation")
        self.S, self.times, self.V = S, times, surf
        self.dS = S[1] - S[0]
        self.CR = bond.conversion_ratio

    def value(self, t, s):
        j = int(np.clip(np.searchsorted(self.times, t, side="right") - 1, 0,
                        len(self.times) - 2))
        w_t = (t - self.times[j]) / (self.times[j + 1] - self.times[j])
        row = (1 - w_t) * self.V[j] + w_t * self.V[j + 1]
        x = np.clip(s / self.dS, 0, len(self.S) - 1 - 1e-9)
        i = x.astype(int)
        w = x - i
        val = (1 - w) * row[i] + w * row[i + 1]
        return np.where(s > self.S[-1], self.CR * s, val)   # beyond grid: like shares


def _basis(x, c):
    """Regression inputs: share price (scaled) and fraction of window above trigger."""
    return np.column_stack([np.ones_like(x), x, x**2, x**3, c, c * x, c**2])


def simulate(bond: ConvertibleBond, mkt: Market, n_paths: int, rule: TriggerRule,
             recovery: float, hazard_power: float, seed: int) -> dict:
    """Daily share paths with defaults, plus trigger counts for 'today only' and `rule`."""
    rng = np.random.default_rng(seed)
    T = bond.maturity
    n_steps = int(round(T * TRADING_DAYS))
    dt = T / n_steps
    half = n_paths // 2
    N = 2 * half                                          # antithetic pairs: z and -z
    level = (bond.call_trigger * bond.face / bond.conversion_ratio
             if bond.call_trigger is not None else np.inf)

    S = np.empty((n_steps + 1, N), dtype=np.float32)
    above_today = np.zeros((n_steps + 1, N), dtype=bool)
    count = np.zeros((n_steps + 1, N), dtype=np.uint8)   # days above trigger in window
    ring = np.zeros((rule.window, N), dtype=bool)         # the last `window` days
    S[0] = mkt.spot
    above_today[0] = mkt.spot >= level
    ring[0] = above_today[0]
    running = ring[0].astype(np.int16)
    count[0] = running
    default_step = np.full(N, n_steps + 10)               # "never"
    s_now = np.full(N, mkt.spot)
    alive = np.ones(N, dtype=bool)

    for i in range(1, n_steps + 1):
        lam = hazard_rate(s_now, mkt, recovery, hazard_power)
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])
        # Shares drift up by the hazard rate: compensation for the default jump
        s_new = s_now * np.exp((mkt.rate - mkt.div_yield + lam - 0.5 * mkt.vol**2) * dt
                               + mkt.vol * np.sqrt(dt) * z)
        dflt = alive & (rng.random(N) < 1 - np.exp(-lam * dt))
        default_step[dflt] = i
        alive &= ~dflt
        s_now = np.where(alive, s_new, 0.0)
        S[i] = s_now

        is_above = s_now >= level
        above_today[i] = is_above
        slot = i % rule.window
        running += is_above.astype(np.int16) - ring[slot].astype(np.int16)
        ring[slot] = is_above
        count[i] = np.minimum(running, 255)

    return {"S": S, "above_today": above_today, "count": count,
            "default_step": default_step, "n_steps": n_steps, "dt": dt, "N": N,
            "half": half}


def _backward(bond: ConvertibleBond, mkt: Market, sim: dict, allowed_by_trigger,
              frac_for_regression, recovery: float, decide_every: int,
              policy: "GridPolicy | None" = None):
    """
    Work back from maturity. Returns each path's value today (so two versions
    on the same paths can be compared path by path).
    allowed_by_trigger[i, path]: is a soft call permitted on day i?
    policy: a GridPolicy to take decisions from the grid (default in
    soft_call_value), or None for Longstaff-Schwartz regression.
    """
    S, default_step = sim["S"], sim["default_step"]
    n_steps, dt, N = sim["n_steps"], sim["dt"], sim["N"]
    T, CR, F = bond.maturity, bond.conversion_ratio, bond.face
    times = np.arange(n_steps + 1) * dt

    period = 1.0 / bond.coupon_freq
    coupon_amt = F * bond.coupon_rate / bond.coupon_freq
    coupon_steps = sorted({int(round((T - k * period) / dt))
                           for k in range(int(round(T * bond.coupon_freq)))
                           if T - k * period > 1e-9})
    put_steps = {int(round(t / dt)) for t in bond.put_dates}
    call_start_step = (int(np.ceil(bond.call_start / dt - 1e-9))
                       if bond.call_price is not None else None)
    decision_steps = sorted(set(range(0, n_steps, decide_every)) | put_steps)

    # Value at maturity for paths still alive
    last_coupon = coupon_amt if n_steps in coupon_steps else 0.0
    value = np.where(default_step > n_steps,
                     np.maximum(F + last_coupon, CR * S[n_steps].astype(float)), 0.0)
    later = n_steps

    for a in reversed(decision_steps):
        # 1. Bring values back from `later` to `a`: discount, add coupons paid in
        #    between (only while alive), and recovery for paths that defaulted.
        ta = times[a]
        value = value * np.exp(-mkt.rate * (times[later] - ta))
        dflt_here = (default_step > a) & (default_step <= later)
        for c in coupon_steps:
            if a < c <= later:
                paid = default_step > c
                value += np.where(paid, coupon_amt * np.exp(-mkt.rate * (times[c] - ta)), 0.0)
        if dflt_here.any():
            td = times[np.minimum(default_step, n_steps)]
            coupons_before = np.zeros(N)
            for c in coupon_steps:
                if a < c <= later:
                    coupons_before += np.where(default_step > c,
                                               coupon_amt * np.exp(-mkt.rate * (times[c] - ta)), 0.0)
            value = np.where(dflt_here,
                             recovery * F * np.exp(-mkt.rate * (td - ta)) + coupons_before,
                             value)
        later = a

        # 2. Decisions at date a, for paths still alive
        live = default_step > a
        if not live.any():
            continue
        s_a = S[a].astype(float)
        conv = CR * s_a
        x = s_a / bond.conversion_price
        frac = frac_for_regression[a]

        if policy is not None:
            # ---- Decisions from the grid's value of holding on ----
            est = policy.value(ta, s_a)
            if bond.put_price is not None and a in put_steps:
                put_now = live & (bond.put_price > est)
                value = np.where(put_now, bond.put_price, value)
                est = np.where(put_now, bond.put_price, est)
            if bond.call_price is not None and a >= call_start_step:
                called = live & allowed_by_trigger[a] & (est > bond.call_price)
                pay = np.maximum(bond.call_price, conv)
                value = np.where(called, pay, value)
                est = np.where(called, pay, est)
            convert = live & (conv > est)
            value = np.where(convert, conv, value)
            continue

        # ---- Decisions by Longstaff-Schwartz regression ----
        if a > 0:
            X = _basis(x[live], frac[live])
            coef, *_ = np.linalg.lstsq(X, value[live], rcond=None)
            est = _basis(x, frac) @ coef                  # estimated value of holding on
        else:
            est = np.full(N, value[live].mean())          # today: every path identical

        if bond.put_price is not None and a in put_steps:
            put_now = live & (bond.put_price > est)
            value = np.where(put_now, bond.put_price, value)
            est = np.where(put_now, bond.put_price, est)

        if bond.call_price is not None and a >= call_start_step:
            ok = live & allowed_by_trigger[a]
            called = ok & (est > bond.call_price)
            pay = np.maximum(bond.call_price, conv)
            value = np.where(called, pay, value)
            est = np.where(called, pay, est)

        # Voluntary early conversion. With no dividend it is never optimal:
        # you'd give up coupons, recovery in default and the option's time
        # value for shares that pay nothing. So we only let the (noisy)
        # regression decide it when there IS a dividend.
        if mkt.div_yield > 0:
            convert = live & (conv > est)
            value = np.where(convert, conv, value)

    return value


def soft_call_value(bond: ConvertibleBond, mkt: Market, rule: TriggerRule = TriggerRule(),
                    n_paths: int = 20000, recovery: float = 0.4, hazard_power: float = 1.0,
                    decide_every: int = 1, seed: int = 7, fd_steps: int = 800,
                    method: str = "grid") -> dict:
    """
    Price a soft-callable convertible with the real path-dependent trigger.
    Returns the grid price (today-only trigger), the Monte Carlo estimate of the
    extra value from the real clause (with its standard error), the final price,
    and a raw Monte Carlo check of the grid price.
    method: "grid" (default) takes the convert / call / put decisions from the
    finite-difference solution; "regression" estimates them by Longstaff-Schwartz.
    """
    if bond.call_price is None or bond.call_trigger is None:
        raise ValueError("Needs a soft call: set call_price and call_trigger.")
    sim = simulate(bond, mkt, n_paths, rule, recovery, hazard_power, seed)
    policy = GridPolicy(bond, mkt, recovery, hazard_power) if method == "grid" else None

    today_only = _backward(bond, mkt, sim, sim["above_today"],
                           sim["above_today"].astype(float), recovery, decide_every, policy)
    real = _backward(bond, mkt, sim, sim["count"] >= rule.days_needed,
                     sim["count"].astype(float) / rule.window, recovery, decide_every,
                     policy)

    half = sim["half"]

    def pair_se(v):
        pairs = 0.5 * (v[:half] + v[half:])
        return float(pairs.std(ddof=1) / np.sqrt(half))

    diff = real - today_only
    fd = price_convertible(bond, mkt, fd_steps, model="jtd", recovery=recovery,
                           hazard_power=hazard_power)["price"]
    return {
        "grid_today_only": fd,
        "mc_today_only": float(today_only.mean()), "mc_today_only_se": pair_se(today_only),
        "mc_real": float(real.mean()),
        "extra_value": float(diff.mean()), "extra_value_se": pair_se(diff),
        "price": fd + float(diff.mean()),
        "pct_defaulted": float((sim["default_step"] <= sim["n_steps"]).mean()),
        "n_paths": sim["N"],
    }


if __name__ == "__main__":
    import time
    mkt = Market(spot=50.0, vol=0.30, rate=0.04, credit_spread=0.03)
    b = ConvertibleBond(call_price=100.0, call_start=2.0, call_trigger=1.3)
    print("Soft call at 130%, callable from year 2, jump-to-default model\n")
    print(f"{'Spot':>5} | {'Grid (today only)':>17} | {'MC check (today only)':>22} | "
          f"{'Extra value of 20-of-30':>24} | {'Price, 20-of-30':>15}")
    for spot in (35.0, 50.0, 60.0, 65.0, 75.0, 90.0):
        t = time.time()
        m = Market(**{**mkt.__dict__, "spot": spot})
        r = soft_call_value(b, m, TriggerRule(20, 30), n_paths=20000)
        print(f"{spot:5.0f} | {r['grid_today_only']:17.3f} | "
              f"{r['mc_today_only']:13.3f} ± {r['mc_today_only_se']:.3f} | "
              f"{r['extra_value']:+15.3f} ± {r['extra_value_se']:.3f} | {r['price']:15.3f}"
              f"   ({time.time() - t:.0f}s)")
