"""Investigate piecewise-linear structure of cadence → watts mapping.

Key observation from first differences: slope jumps at cadence 40, 50, 60, 70, 80.
Within each decade, deltas alternate between two adjacent integers — the signature of
floor() applied to a linear function with non-integer slope.

For each decade [D, D+10):
    watts = floor(base + slope * (c - D))

Find (base, slope) feasibility region per decade, then check for patterns.
"""

import numpy as np
from fractions import Fraction
from scipy.optimize import linprog

# Ground truth (cadence, watts) — non-zero only
data = [
    (23, 23),
    (25, 28),
    (27, 33),
    (28, 36),
    (29, 38),
    (32, 49),
    (33, 53),
    (34, 57),
    (35, 62),
    (37, 70),
    (38, 74),
    (39, 78),
    (40, 82),
    (41, 89),
    (42, 96),
    (43, 102),
    (44, 109),
    (45, 116),
    (46, 123),
    (47, 130),
    (48, 136),
    (49, 143),
    (50, 150),
    (51, 159),
    (52, 168),
    (53, 178),
    (54, 187),
    (55, 196),
    (56, 205),
    (57, 214),
    (58, 224),
    (59, 233),
    (60, 242),
    (61, 256),
    (62, 270),
    (63, 284),
    (64, 298),
    (65, 313),
    (66, 327),
    (67, 341),
    (68, 355),
    (69, 369),
    (70, 383),
    (71, 400),
    (72, 416),
    (73, 433),
    (74, 449),
    (75, 466),
    (76, 483),
    (77, 499),
    (78, 516),
    (79, 532),
    (80, 549),
    (81, 570),
    (83, 612),
    (84, 633),
    (87, 697),
    (88, 718),
    (89, 739),
]

c_all = np.array([d[0] for d in data], dtype=float)
w_all = np.array([d[1] for d in data], dtype=float)

# Also include the decade boundary transitions
# watts(40)=82 constrains the 30s decade, watts(50)=150 constrains the 40s, etc.

decades = [20, 30, 40, 50, 60, 70, 80]

print("=" * 70)
print("Piecewise-linear analysis: watts = floor(base + slope*(c - D))")
print("=" * 70)

results = {}

for dec in decades:
    dec_end = dec + 10
    # Data points in this decade (inclusive of both start and end)
    mask = (c_all >= dec) & (c_all <= dec_end)
    c_dec = c_all[mask]
    w_dec = w_all[mask]

    if len(c_dec) < 2:
        print(f"\nDecade {dec}s: insufficient data ({len(c_dec)} points)")
        continue

    print(
        f"\nDecade {dec}s: {len(c_dec)} points, cadence {int(c_dec.min())}–{int(c_dec.max())}"
    )

    # 2D LP: find feasible (base, slope) where
    # w_i <= base + slope * (c_i - dec) < w_i + 1
    # Variables: x = [base, slope]
    offsets = c_dec - dec
    X = np.column_stack([np.ones_like(offsets), offsets])  # [1, c-dec]
    EPS = 1e-9

    # Constraints: w <= X@x < w+1
    A_ub = np.vstack([-X, X])
    b_ub = np.concatenate([-w_dec, w_dec + 1 - EPS])

    # Minimize base (to find the feasible region)
    result_lo = linprog([1, 0], A_ub=A_ub, b_ub=b_ub, method="highs")
    result_hi = linprog([-1, 0], A_ub=A_ub, b_ub=b_ub, method="highs")

    if not result_lo.success or not result_hi.success:
        print("  INFEASIBLE!")
        continue

    base_lo = result_lo.x[0]
    base_hi = -result_hi.fun  # maximize = -minimize(-f)

    # Find slope range by minimizing/maximizing slope
    result_slo = linprog([0, 1], A_ub=A_ub, b_ub=b_ub, method="highs")
    result_shi = linprog([0, -1], A_ub=A_ub, b_ub=b_ub, method="highs")

    slope_lo = result_slo.x[1]
    slope_hi = result_shi.x[1]

    print(f"  base ∈ [{base_lo:.6f}, {base_hi:.6f}]")
    print(f"  slope ∈ [{slope_lo:.6f}, {slope_hi:.6f}]")

    # Try to find simplest rational slope
    # Use Stern-Brocot tree / mediants to find simplest fraction in interval
    def simplest_fraction(lo, hi, max_denom=1000):
        """Find simplest fraction in (lo, hi) using Stern-Brocot tree."""
        best = None
        for d in range(1, max_denom + 1):
            n_lo = int(np.ceil(lo * d))
            n_hi = int(np.floor(hi * d - 1e-12))
            if n_lo <= n_hi:
                for n in range(n_lo, n_hi + 1):
                    from math import gcd

                    g = gcd(n, d)
                    if g > 1:
                        continue
                    if best is None or d < best[1]:
                        best = (n, d)
        return best

    simple_slope = simplest_fraction(slope_lo, slope_hi)
    if simple_slope:
        sn, sd = simple_slope
        print(f"  Simplest rational slope: {sn}/{sd} = {sn / sd:.10f}")

        # Verify with this slope: find valid base range
        slope_exact = sn / sd
        base_lo_exact = max(
            w_dec[i] - slope_exact * offsets[i] for i in range(len(c_dec))
        )
        base_hi_exact = min(
            w_dec[i] + 1 - slope_exact * offsets[i] for i in range(len(c_dec))
        )
        print(
            f"  With slope={sn}/{sd}: base ∈ [{base_lo_exact:.6f}, {base_hi_exact:.6f})"
        )

        # Check integer bases
        for b_int in range(
            int(np.floor(base_lo_exact)), int(np.ceil(base_hi_exact)) + 1
        ):
            if base_lo_exact <= b_int < base_hi_exact:
                pred = np.floor(b_int + slope_exact * offsets).astype(int)
                errors = int(np.sum(pred != w_dec.astype(int)))
                if errors == 0:
                    print(f"  INTEGER BASE: base={b_int}, slope={sn}/{sd} → EXACT!")

        # Also check half-integer bases
        for b_half_2x in range(
            int(np.floor(base_lo_exact * 2)), int(np.ceil(base_hi_exact * 2)) + 1
        ):
            b_half = b_half_2x / 2
            if base_lo_exact <= b_half < base_hi_exact:
                pred = np.floor(b_half + slope_exact * offsets).astype(int)
                errors = int(np.sum(pred != w_dec.astype(int)))
                if errors == 0:
                    # Express as fraction
                    bf = Fraction(b_half_2x, 2)
                    print(
                        f"  HALF-INT BASE: base={bf} = {float(bf):.4f}, slope={sn}/{sd} → EXACT!"
                    )

        results[dec] = {
            "slope_n": sn,
            "slope_d": sd,
            "base_lo": base_lo_exact,
            "base_hi": base_hi_exact,
        }

# Summary
print("\n" + "=" * 70)
print("SUMMARY: Slopes by decade")
print("=" * 70)

for dec in sorted(results):
    r = results[dec]
    s = r["slope_n"] / r["slope_d"]
    print(
        f"  {dec}s: slope = {r['slope_n']}/{r['slope_d']} = {s:.6f}, "
        f"base ∈ [{r['base_lo']:.4f}, {r['base_hi']:.4f})"
    )

# Check if slopes follow a pattern
if len(results) >= 3:
    slopes = [
        (dec, results[dec]["slope_n"] / results[dec]["slope_d"])
        for dec in sorted(results)
    ]
    print(f"\n  Slope values: {[f'{s:.4f}' for _, s in slopes]}")

    # Check for linear relationship between decade and slope
    decs = np.array([d for d, _ in slopes])
    slps = np.array([s for _, s in slopes])

    print("\n  Slope ratios (consecutive):")
    for i in range(1, len(slopes)):
        ratio = slopes[i][1] / slopes[i - 1][1]
        print(f"    {slopes[i][0]}s / {slopes[i - 1][0]}s = {ratio:.4f}")

# Now test: is there a global formula that generates these slopes?
# Common embedded pattern: slope[decade] = A * decade + B, or A * decade^2 + B
print("\n  Linear fit to slopes vs decade:")
decs_arr = np.array([dec for dec in sorted(results)], dtype=float)
slopes_arr = np.array(
    [results[dec]["slope_n"] / results[dec]["slope_d"] for dec in sorted(results)]
)

# Fit slope = a * decade + b
coeffs = np.polyfit(decs_arr, slopes_arr, 1)
pred_slopes = np.polyval(coeffs, decs_arr)
print(f"    slope = {coeffs[0]:.6f} * decade + {coeffs[1]:.4f}")
print(f"    Max residual: {np.max(np.abs(slopes_arr - pred_slopes)):.4f}")

# Fit slope = a * decade^2 + b * decade + c
coeffs2 = np.polyfit(decs_arr, slopes_arr, 2)
pred_slopes2 = np.polyval(coeffs2, decs_arr)
print("\n  Quadratic fit:")
print(f"    slope = {coeffs2[0]:.8f} * d^2 + {coeffs2[1]:.6f} * d + {coeffs2[2]:.4f}")
print(f"    Max residual: {np.max(np.abs(slopes_arr - pred_slopes2)):.4f}")
