"""
Benchmark against QuantLib
==========================

QuantLib is the industry-standard open-source pricing library. This script
prices a range of convertibles in QuantLib and in our code, and runs three
separate checks:

1. IMPLEMENTATION: QuantLib's convertible engine doesn't use the
   Tsiveriotis-Fernandes cash/equity split. It tracks the probability of
   conversion at each node and discounts at a blended rate,
   r + (1 - p) x spread. We built that same method as a separate tree
   (price_convertible_tree_cp). If it matches QuantLib, our tree code is right.

2. NUMERICS: our finite-difference price vs the same calculation on a grid
   twice as fine. If they agree, the grid is fine enough. (We also show our
   own Tsiveriotis-Fernandes binomial tree for comparison; with a call
   feature the tree is the noisier of the two.)

3. MODEL DIFFERENCE: our Tsiveriotis-Fernandes price vs QuantLib's blended
   price. This gap is real and expected: they are different models.
   Tsiveriotis-Fernandes keeps the cash and equity parts separate, so it
   charges the credit spread only on value that actually depends on the
   issuer paying cash.

Plus closed-form checks against textbook formulas.

Run:  python benchmark.py
It prints a comparison table and saves it to benchmark_results.csv, which
the website's Validation tab reads. (Only this script needs QuantLib:
    pip install QuantLib
the website itself doesn't.)
"""

import math
import time
import numpy as np
import pandas as pd
import QuantLib as ql

from convertible_pricer import (ConvertibleBond, Market, price_convertible,
                                price_convertible_fd, price_convertible_tree,
                                price_convertible_tree_cp)


TODAY = ql.Date(16, 10, 2026)


def quantlib_price(bond: ConvertibleBond, mkt: Market, steps: int = 2000) -> float:
    """Price the same bond in QuantLib (dirty value per 100 face)."""
    ql.Settings.instance().evaluationDate = TODAY
    cal = ql.NullCalendar()                       # no holidays: keeps dates exact
    dc = ql.Actual365Fixed()

    months = int(round(bond.maturity * 12))
    maturity = TODAY + ql.Period(months, ql.Months)
    tenor = ql.Period(12 // bond.coupon_freq, ql.Months)
    schedule = ql.Schedule(TODAY, maturity, tenor, cal, ql.Unadjusted, ql.Unadjusted,
                           ql.DateGeneration.Backward, False)

    # Issuer call: QuantLib needs explicit dates, so we allow a call every day
    # from the call start date (our model treats it as continuous).
    callability = ql.CallabilitySchedule()
    if bond.call_price is not None:
        # Anchor in months (not days) so the call start lines up with coupon dates
        d = TODAY + ql.Period(int(round(bond.call_start * 12)), ql.Months)
        price = ql.BondPrice(bond.call_price, ql.BondPrice.Dirty)
        while d < maturity:
            if bond.call_trigger is None:
                callability.append(ql.Callability(price, ql.Callability.Call, d))
            else:
                callability.append(ql.SoftCallability(price, d, bond.call_trigger))
            d += 1
    for t in bond.put_dates:
        callability.append(ql.Callability(ql.BondPrice(bond.put_price, ql.BondPrice.Dirty),
                                          ql.Callability.Put,
                                          TODAY + ql.Period(int(round(t * 12)), ql.Months)))

    cb = ql.ConvertibleFixedCouponBond(
        ql.AmericanExercise(TODAY, maturity), bond.conversion_ratio, callability,
        TODAY, 0, [bond.coupon_rate], ql.Thirty360(ql.Thirty360.BondBasis), schedule,
        bond.face)

    spot = ql.QuoteHandle(ql.SimpleQuote(mkt.spot))
    rf = ql.YieldTermStructureHandle(ql.FlatForward(TODAY, mkt.rate, dc, ql.Continuous))
    div = ql.YieldTermStructureHandle(ql.FlatForward(TODAY, mkt.div_yield, dc, ql.Continuous))
    vol = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(TODAY, cal, mkt.vol, dc))
    process = ql.BlackScholesMertonProcess(spot, div, rf, vol)
    spread = ql.QuoteHandle(ql.SimpleQuote(mkt.credit_spread))

    # Average two neighbouring step counts: trees wobble with step count
    prices = []
    for n in (steps, steps + 1):
        cb.setPricingEngine(ql.BinomialCRRConvertibleEngine(process, n, spread))
        prices.append(cb.NPV())
    return float(np.mean(prices))


def test_cases():
    """A spread of bonds: deep out of the money to deep in, different features."""
    cases = []
    features = {
        "No call": dict(),
        "Hard call 103 from yr 3": dict(call_price=103.0, call_start=3.0),
        "Soft call 130% from yr 2": dict(call_price=100.0, call_start=2.0, call_trigger=1.3),
        "Put at 100 in yr 3": dict(put_price=100.0, put_dates=[3.0]),
    }
    for feat_name, feat in features.items():
        for spot in (20.0, 40.0, 60.0, 90.0):
            cases.append((feat_name, dict(coupon_rate=0.02, maturity=5.0,
                                          conversion_ratio=2.0, **feat),
                          dict(spot=spot, vol=0.30, rate=0.04, credit_spread=0.03)))
    # Vary vol, spread, coupon and dividends on the hard-call bond
    base = dict(coupon_rate=0.02, maturity=5.0, conversion_ratio=2.0,
                call_price=103.0, call_start=3.0)
    for v in (0.15, 0.50):
        cases.append(("Hard call, vol varied", base, dict(spot=40.0, vol=v, rate=0.04,
                                                          credit_spread=0.03)))
    for s in (0.01, 0.08):
        cases.append(("Hard call, spread varied", base, dict(spot=40.0, vol=0.30, rate=0.04,
                                                             credit_spread=s)))
    cases.append(("Hard call, 5% coupon", {**base, "coupon_rate": 0.05},
                  dict(spot=40.0, vol=0.30, rate=0.04, credit_spread=0.03)))
    cases.append(("Hard call, 2% dividend", base, dict(spot=40.0, vol=0.30, rate=0.04,
                                                       credit_spread=0.03, div_yield=0.02)))
    return cases


def averaged(fn, steps_list):
    """Trees wobble with step count: average several to get a steady number."""
    return float(np.mean([fn(n) for n in steps_list]))


def run():
    rows = []
    tree_steps = range(1500, 2500, 100)
    for name, b_kw, m_kw in test_cases():
        bond, mkt = ConvertibleBond(**b_kw), Market(**m_kw)
        qlp = quantlib_price(bond, mkt)
        replica = averaged(lambda n: price_convertible_tree_cp(bond, mkt, n)["price"],
                           range(1500, 2500, 250))
        fd = price_convertible(bond, mkt, 800)["price"]
        fd_fine = price_convertible_fd(bond, mkt, 1600, ns=2000)["price"]
        tf_tree = averaged(lambda n: price_convertible_tree(bond, mkt, n)["price"],
                           tree_steps)
        rows.append({
            "Bond": name, "Share price": mkt.spot, "Vol": mkt.vol,
            "Spread (bp)": round(mkt.credit_spread * 1e4),
            "QuantLib": qlp, "Our replica of QuantLib": replica,
            "1. Replica - QuantLib": replica - qlp,
            "Our TF finite diff": fd, "TF finite diff, 2x finer grid": fd_fine,
            "2. FD convergence": fd - fd_fine,
            "Our TF tree": tf_tree, "FD - tree": fd - tf_tree,
            "3. TF - QuantLib (model gap)": fd - qlp,
        })
    df = pd.DataFrame(rows)
    df.to_csv("benchmark_results.csv", index=False)
    return df


def closed_form_checks():
    """Cases with a textbook answer, priced by our finite-difference engine."""
    from statistics import NormalDist
    N = NormalDist().cdf
    out = []

    # (a) Zero coupon, no spread, no call: bond + 2 European calls (Black-Scholes)
    S, K, r, v, T = 40.0, 50.0, 0.04, 0.30, 5.0
    d1 = (math.log(S / K) + (r + v * v / 2) * T) / (v * math.sqrt(T))
    d2 = d1 - v * math.sqrt(T)
    exact = 100 * math.exp(-r * T) + 2 * (S * N(d1) - K * math.exp(-r * T) * N(d2))
    ours = price_convertible(ConvertibleBond(coupon_rate=0.0, conversion_ratio=2.0),
                             Market(spot=S, credit_spread=0.0), 800)["price"]
    out.append(("Zero coupon, no spread = bond + 2 Black-Scholes calls", exact, ours))

    # (b) Shares worthless: plain bond discounted at rate + spread
    y = 0.07
    exact = sum(math.exp(-y * k / 2) for k in range(1, 11)) + 100 * math.exp(-y * 5)
    ours = price_convertible(ConvertibleBond(), Market(spot=0.5), 800)["price"]
    out.append(("Shares near zero = straight bond at rate + spread", exact, ours))

    # (c) Jump-to-default, constant hazard, shares worthless: risky bond formula
    rr, sp, R = 0.04, 0.03, 0.4
    lam = sp / (1 - R)
    k = rr + lam
    exact = (sum(math.exp(-k * j / 2) for j in range(1, 11)) + 100 * math.exp(-k * 5)
             + lam * R * 100 * (1 - math.exp(-k * 5)) / k)
    ours = price_convertible(ConvertibleBond(conversion_ratio=0.001), Market(spot=40.0),
                             800, model="jtd", hazard_power=0.0)["price"]
    out.append(("Jump-to-default, no conversion value = risky bond formula", exact, ours))

    # (d) Jump-to-default with zero default risk = fixed-spread model with zero spread
    b = ConvertibleBond(call_price=103.0, call_start=3.0)
    exact = price_convertible(b, Market(credit_spread=0.0), 800)["price"]
    ours = price_convertible(b, Market(credit_spread=0.0), 800, model="jtd")["price"]
    out.append(("Jump-to-default with no default risk = fixed-spread at zero spread",
                exact, ours))

    df = pd.DataFrame(out, columns=["Check", "Exact", "Ours"])
    df["Difference"] = df["Ours"] - df["Exact"]
    df.to_csv("closed_form_checks.csv", index=False)
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    t = time.time()
    df = run()
    print(df.round(3).to_string(index=False))
    print()
    for col in ("1. Replica - QuantLib", "2. FD convergence", "FD - tree",
                "3. TF - QuantLib (model gap)"):
        print(f"{col:32s} average |gap| {df[col].abs().mean():.3f}, "
              f"largest {df[col].abs().max():.3f}")
    print()
    print(closed_form_checks().round(4).to_string(index=False))
    print(f"\nDone in {time.time() - t:.0f}s. Saved benchmark_results.csv "
          f"and closed_form_checks.csv")
