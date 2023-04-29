"""Search for exact cadence → watts firmware formula.

Approach:
1. Power law: floor(a * c^b) — sweep b, intersect a-intervals
2. Quadratic integer: floor((A*c^2 + B*c) / D) — LP per D
3. Cubic integer: floor((A*c^3 + B*c^2 + C*c) / D) — LP per D
4. Piecewise power law with breakpoint at cadence ~60-62
"""

import numpy as np
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

c = np.array([d[0] for d in data], dtype=float)
w = np.array([d[1] for d in data], dtype=float)
n = len(c)

print(f"Data: {n} points, cadence {int(c.min())}–{int(c.max())}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# 1. POWER LAW: floor(a * c^b) = w
# ═══════════════════════════════════════════════════════════════════════════
# For fixed b, a ∈ [w_i/c_i^b, (w_i+1)/c_i^b) for all i.

print("=" * 60)
print("1. Power law: floor(a * c^b)")
print("=" * 60)

best_b = None
best_width = 0
feasible_regions = []

for b100 in range(150, 351):  # b from 1.50 to 3.50
    b = b100 / 100.0
    a_lo = -np.inf
    a_hi = np.inf
    for ci, wi in zip(c, w):
        lo = wi / ci**b
        hi = (wi + 1) / ci**b
        a_lo = max(a_lo, lo)
        a_hi = min(a_hi, hi)
    if a_lo < a_hi:
        width = a_hi - a_lo
        feasible_regions.append((b, a_lo, a_hi, width))
        if width > best_width:
            best_width = width
            best_b = b

if feasible_regions:
    print(
        f"\nFeasible b range: {feasible_regions[0][0]:.2f} to {feasible_regions[-1][0]:.2f}"
    )
    print(
        f"Best b = {best_b:.2f}, a ∈ [{feasible_regions[0][1]:.10f}, "
        f"{feasible_regions[0][2]:.10f}), width = {best_width:.2e}"
    )

    # Fine sweep around best b
    fine_best_b = None
    fine_best_width = 0
    fine_feasible = []

    for b10000 in range(int(best_b * 10000) - 5000, int(best_b * 10000) + 5001):
        b = b10000 / 10000.0
        a_lo = -np.inf
        a_hi = np.inf
        for ci, wi in zip(c, w):
            lo = wi / ci**b
            hi = (wi + 1) / ci**b
            a_lo = max(a_lo, lo)
            a_hi = min(a_hi, hi)
        if a_lo < a_hi:
            width = a_hi - a_lo
            fine_feasible.append((b, a_lo, a_hi, width))
            if width > fine_best_width:
                fine_best_width = width
                fine_best_b = b

    if fine_feasible:
        print(f"\nFine sweep: best b = {fine_best_b:.4f}")
        print(
            f"  Feasible b range: [{fine_feasible[0][0]:.4f}, {fine_feasible[-1][0]:.4f}]"
        )

        # Get a-interval at best b
        a_lo = -np.inf
        a_hi = np.inf
        tightest_lo = (-1, 0, 0)
        tightest_hi = (-1, 0, 0)
        for ci, wi in zip(c, w):
            lo = wi / ci**fine_best_b
            hi = (wi + 1) / ci**fine_best_b
            if lo > a_lo:
                a_lo = lo
                tightest_lo = (ci, wi, lo)
            if hi < a_hi:
                a_hi = hi
                tightest_hi = (ci, wi, hi)
        print(f"  a ∈ [{a_lo:.12f}, {a_hi:.12f})")
        print(f"  Width: {a_hi - a_lo:.2e}")
        print(f"  Lower bound set by c={int(tightest_lo[0])}, w={int(tightest_lo[1])}")
        print(f"  Upper bound set by c={int(tightest_hi[0])}, w={int(tightest_hi[1])}")
        a_mid = (a_lo + a_hi) / 2
        print(f"  Midpoint a = {a_mid:.12f}")

        # Verify
        pred = np.floor(a_mid * c**fine_best_b).astype(int)
        errors = np.sum(pred != w.astype(int))
        print(
            f"  Verification: {errors} errors with a={a_mid:.12f}, b={fine_best_b:.4f}"
        )

        # Search for simple rational a = p/q
        print(
            f"\n  Searching for simple rational a = p/q in [{a_lo:.10f}, {a_hi:.10f}):"
        )
        found_rationals = []
        for q in range(1, 10001):
            p_lo = int(np.ceil(a_lo * q))
            p_hi = int(np.floor(a_hi * q - 1e-15))
            if p_lo <= p_hi:
                for p in range(p_lo, min(p_hi + 1, p_lo + 3)):
                    from math import gcd

                    g = gcd(p, q)
                    if g > 1:
                        continue  # only reduced fractions
                    found_rationals.append((p, q, p / q))
                    if len(found_rationals) <= 20:
                        # verify
                        a_rat = p / q
                        pred_rat = np.floor(a_rat * c**fine_best_b).astype(int)
                        err_rat = int(np.sum(pred_rat != w.astype(int)))
                        print(f"    a = {p}/{q} = {a_rat:.12f}, errors={err_rat}")
        if found_rationals:
            print(f"  Total rationals found (q ≤ 10000): {len(found_rationals)}")
            simplest = min(found_rationals, key=lambda x: x[1])
            print(f"  Simplest: {simplest[0]}/{simplest[1]}")
else:
    print("\nNo feasible b found in [1.50, 3.50]")


# ═══════════════════════════════════════════════════════════════════════════
# 2. QUADRATIC INTEGER: floor((A*c^2 + B*c) / D)
# ═══════════════════════════════════════════════════════════════════════════
# For fixed D: w*D ≤ A*c^2 + B*c < (w+1)*D, linear in (A, B)

print("\n" + "=" * 60)
print("2. Quadratic integer: floor((A*c^2 + B*c) / D)")
print("=" * 60)

EPS = 1e-6

quad_results = []
for D in range(1, 5001):
    # Design matrix: [c^2, c]
    X = np.column_stack([c**2, c])

    # Constraints: w*D ≤ A*c^2 + B*c < (w+1)*D
    # → -X @ [A,B] ≤ -w*D
    # →  X @ [A,B] ≤ (w+1)*D - EPS
    A_ub = np.vstack([-X, X])
    b_ub = np.concatenate([-w * D, (w + 1) * D - EPS])

    # Minimize sum of f(c) - w*D (center in feasible region)
    c_obj = X.sum(axis=0)

    result = linprog(
        c_obj, A_ub=A_ub, b_ub=b_ub, method="highs", options={"presolve": True}
    )

    if result.success:
        A_val, B_val = result.x
        pred = np.floor((A_val * c**2 + B_val * c) / D).astype(int)
        errors = int(np.sum(pred != w.astype(int)))
        if errors == 0:
            quad_results.append((D, A_val, B_val))
            if len(quad_results) <= 10:
                print(f"  D={D}: A={A_val:.6f}, B={B_val:.6f}")
                # Check if A, B are close to integers
                A_round = round(A_val)
                B_round = round(B_val)
                pred_int = np.floor((A_round * c**2 + B_round * c) / D).astype(int)
                int_err = int(np.sum(pred_int != w.astype(int)))
                if int_err == 0:
                    print(
                        f"         INTEGER: A={A_round}, B={B_round}, D={D}, errors={int_err}"
                    )

if quad_results:
    print(f"\n  Total feasible D values: {len(quad_results)}")
else:
    print("\n  No feasible quadratic found (D=1..5000)")


# ═══════════════════════════════════════════════════════════════════════════
# 3. CUBIC INTEGER: floor((A*c^3 + B*c^2 + C*c) / D)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("3. Cubic integer: floor((A*c^3 + B*c^2 + C*c) / D)")
print("=" * 60)

cubic_results = []
for D in range(1, 5001):
    X = np.column_stack([c**3, c**2, c])
    A_ub = np.vstack([-X, X])
    b_ub = np.concatenate([-w * D, (w + 1) * D - EPS])
    c_obj = X.sum(axis=0)

    result = linprog(
        c_obj, A_ub=A_ub, b_ub=b_ub, method="highs", options={"presolve": True}
    )

    if result.success:
        A_val, B_val, C_val = result.x
        pred = np.floor((A_val * c**3 + B_val * c**2 + C_val * c) / D).astype(int)
        errors = int(np.sum(pred != w.astype(int)))
        if errors == 0:
            cubic_results.append((D, A_val, B_val, C_val))
            if len(cubic_results) <= 10:
                print(f"  D={D}: A={A_val:.6f}, B={B_val:.6f}, C={C_val:.6f}")
                A_r, B_r, C_r = round(A_val), round(B_val), round(C_val)
                pred_int = np.floor((A_r * c**3 + B_r * c**2 + C_r * c) / D).astype(int)
                int_err = int(np.sum(pred_int != w.astype(int)))
                if int_err == 0:
                    print(f"         INTEGER: A={A_r}, B={B_r}, C={C_r}, D={D}")

if cubic_results:
    print(f"\n  Total feasible D values: {len(cubic_results)}")
else:
    print("\n  No feasible cubic found (D=1..5000)")


# ═══════════════════════════════════════════════════════════════════════════
# 4. PIECEWISE: Different formulas below/above breakpoint
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("4. Piecewise power law: floor(a * c^b), split at breakpoints")
print("=" * 60)

for bp in [55, 58, 60, 61, 62, 63, 65]:
    lo_mask = c <= bp
    hi_mask = c > bp
    if lo_mask.sum() < 3 or hi_mask.sum() < 3:
        continue

    c_lo, w_lo = c[lo_mask], w[lo_mask]
    c_hi, w_hi = c[hi_mask], w[hi_mask]

    print(
        f"\n  Breakpoint at cadence={bp} ({int(lo_mask.sum())} + {int(hi_mask.sum())} points):"
    )

    for label, c_seg, w_seg in [("lower", c_lo, w_lo), ("upper", c_hi, w_hi)]:
        best_b_seg = None
        best_width_seg = 0
        b_range_lo, b_range_hi = None, None

        for b10000 in range(15000, 35001):
            b = b10000 / 10000.0
            a_lo_seg = -np.inf
            a_hi_seg = np.inf
            for ci, wi in zip(c_seg, w_seg):
                lo = wi / ci**b
                hi = (wi + 1) / ci**b
                a_lo_seg = max(a_lo_seg, lo)
                a_hi_seg = min(a_hi_seg, hi)
            if a_lo_seg < a_hi_seg:
                width = a_hi_seg - a_lo_seg
                if b_range_lo is None:
                    b_range_lo = b
                b_range_hi = b
                if width > best_width_seg:
                    best_width_seg = width
                    best_b_seg = b

        if best_b_seg is not None:
            # get interval at best b
            a_lo_seg = max(wi / ci**best_b_seg for ci, wi in zip(c_seg, w_seg))
            a_hi_seg = min((wi + 1) / ci**best_b_seg for ci, wi in zip(c_seg, w_seg))
            a_mid_seg = (a_lo_seg + a_hi_seg) / 2
            pred_seg = np.floor(a_mid_seg * c_seg**best_b_seg).astype(int)
            err_seg = int(np.sum(pred_seg != w_seg.astype(int)))
            print(
                f"    {label}: b={best_b_seg:.4f}, a~{a_mid_seg:.10f}, "
                f"width={best_width_seg:.2e}, errors={err_seg}, "
                f"b_range=[{b_range_lo:.4f}, {b_range_hi:.4f}]"
            )
        else:
            print(f"    {label}: no feasible power law")


# ═══════════════════════════════════════════════════════════════════════════
# 5. SPECIAL FORMS: Common embedded patterns
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("5. Special embedded patterns")
print("=" * 60)

# 5a. floor(c^2 / K) for integer K
print("\n  5a. floor(c^2 / K):")
for K in range(1, 200):
    pred = np.floor(c**2 / K).astype(int)
    if np.all(pred == w.astype(int)):
        print(f"    K={K}: EXACT!")

# 5b. (c * c * K) >> N  ≡  floor(c^2 * K / 2^N)
print("\n  5b. floor(c^2 * K / 2^N):")
for N in range(1, 25):
    for K in range(1, 2 ** min(N, 16)):
        pred = np.floor(c**2 * K / 2**N).astype(int)
        if np.all(pred == w.astype(int)):
            print(f"    K={K}, N={N}: EXACT!  (c^2 * {K} >> {N})")

# 5c. floor((c^2 * K + c * M) / 2^N)
print("\n  5c. floor((c^2*K + c*M) / 2^N) — limited search:")
for N in range(6, 16):
    D = 2**N
    X = np.column_stack([c**2, c])
    A_ub = np.vstack([-X, X])
    b_ub = np.concatenate([-w * D, (w + 1) * D - EPS])
    c_obj = X.sum(axis=0)
    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, method="highs")
    if result.success:
        K_val, M_val = result.x
        K_r, M_r = round(K_val), round(M_val)
        pred = np.floor((K_r * c**2 + M_r * c) / D).astype(int)
        err = int(np.sum(pred != w.astype(int)))
        if err == 0:
            print(f"    N={N} (D={D}): K={K_r}, M={M_r} → EXACT!")
        elif err <= 2:
            print(f"    N={N} (D={D}): K={K_r}, M={M_r} → {err} errors (close)")

# 5d. floor(c^2 * p/q + c * r/s) with rational a, b — brute force small denominators
print("\n  5d. floor(c^2 * p/q + c * r/s) with small q, s:")
best_err = n
best_params = None
for q in range(1, 100):
    # For each q, find feasible p range from the data
    for s in range(1, 100):
        # Too many combinations; only test when q*s <= 500
        if q * s > 500:
            continue
        # LP: floor((p*s*c^2 + r*q*c) / (q*s)) = w
        D = q * s
        X = np.column_stack([c**2, c])  # coefficients are p*s/D and r*q/D
        # We need w*D ≤ p*s*c^2 + r*q*c < (w+1)*D
        # Let A = p*s, B = r*q (integers)
        A_ub_qs = np.vstack([-X, X])
        b_ub_qs = np.concatenate([-w * D, (w + 1) * D - EPS])
        result = linprog(X.sum(axis=0), A_ub=A_ub_qs, b_ub=b_ub_qs, method="highs")
        if result.success:
            Av, Bv = result.x
            Ar, Br = round(Av), round(Bv)
            pred_r = np.floor((Ar * c**2 + Br * c) / D).astype(int)
            err_r = int(np.sum(pred_r != w.astype(int)))
            if err_r == 0:
                # Simplify: A=p*s, B=r*q, D=q*s → a=A/D=p/q, b=B/D=r/s
                from math import gcd

                g1 = gcd(Ar, D)
                g2 = gcd(Br, D)
                print(
                    f"    D={D}: ({Ar}*c^2 + {Br}*c) / {D} "
                    f"= {Ar // g1}/{D // g1}*c^2 + {Br // g2}/{D // g2}*c → EXACT!"
                )
                if best_params is None or D < best_params[2]:
                    best_params = (Ar, Br, D)

if best_params:
    A, B, D = best_params
    print(f"\n  Simplest quadratic integer formula: ({A}*c^2 + {B}*c) / {D}")
    print(f"  watts = floor(({A}*c^2 + {B}*c) / {D})")


# ═══════════════════════════════════════════════════════════════════════════
# 6. FIRST-DIFFERENCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("6. First-difference structure")
print("=" * 60)

# Only consecutive cadences
consec_mask = np.diff(c) == 1
c_consec = c[:-1][consec_mask]
dw = np.diff(w)[consec_mask]

print(f"\n  Consecutive-cadence deltas ({int(consec_mask.sum())} pairs):")
for ci, dwi in zip(c_consec, dw):
    print(f"    c={int(ci)}→{int(ci + 1)}: Δw={int(dwi)}")

# Group by decade
print("\n  By decade:")
for decade_start in range(20, 90, 10):
    mask = (c_consec >= decade_start) & (c_consec < decade_start + 10)
    if mask.sum() > 0:
        deltas = dw[mask]
        unique_d = sorted(set(deltas.astype(int)))
        print(f"    {decade_start}s: deltas={unique_d}, mean={np.mean(deltas):.2f}")
