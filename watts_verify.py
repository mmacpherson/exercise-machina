"""Verify piecewise-linear watts formula with common denominator D=10.

Hypothesis: watts = (B[decade] + N[decade] * (c % 10)) // 10

where decade = c // 10, and B[], N[] are small-integer lookup tables.
"""

import numpy as np

# Ground truth (cadence, watts)
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

c_all = np.array([d[0] for d in data])
w_all = np.array([d[1] for d in data])


def find_exact_BN(decade, data_points):
    """Find all (B, N) pairs where (B + N*offset)//10 == watts for all data."""
    results = []
    for N in range(0, 500):
        # For each N, find valid B range
        b_lo = -(10**9)
        b_hi = 10**9
        for c, w in data_points:
            offset = c - decade * 10
            # Need: w == (B + N*offset) // 10
            # i.e., 10*w <= B + N*offset < 10*(w+1)
            # i.e., 10*w - N*offset <= B < 10*(w+1) - N*offset
            lo = 10 * w - N * offset
            hi = 10 * (w + 1) - N * offset
            b_lo = max(b_lo, lo)
            b_hi = min(b_hi, hi)
        if b_lo < b_hi:
            for B in range(b_lo, b_hi):
                # Verify
                ok = True
                for c, w in data_points:
                    offset = c - decade * 10
                    pred = (B + N * offset) // 10
                    if pred != w:
                        ok = False
                        break
                if ok:
                    results.append((B, N))
    return results


print("=" * 70)
print("Searching for: watts = (B + N * (c % 10)) // 10")
print("=" * 70)

decades = {
    2: [(c, w) for c, w in data if 20 <= c < 30],
    3: [(c, w) for c, w in data if 30 <= c < 40],
    4: [(c, w) for c, w in data if 40 <= c < 50],
    5: [(c, w) for c, w in data if 50 <= c < 60],
    6: [(c, w) for c, w in data if 60 <= c < 70],
    7: [(c, w) for c, w in data if 70 <= c < 80],
    8: [(c, w) for c, w in data if 80 <= c < 90],
}

# Include boundary values from next decade as constraints
# e.g., c=40 (watts=82) constrains the 30s formula at offset=10
decades_with_boundary = {
    2: decades[2] + [(c, w) for c, w in data if c == 30]
    if 30 in [c for c, _ in data]
    else decades[2],
    3: decades[3] + [(40, 82)],  # c=40 is first point of 40s
    4: decades[4] + [(50, 150)],
    5: decades[5] + [(60, 242)],
    6: decades[6] + [(70, 383)],
    7: decades[7] + [(80, 549)],
    8: decades[8],
}

best_BN = {}

for dec in sorted(decades):
    pts = decades[dec]
    if not pts:
        continue

    results = find_exact_BN(dec, pts)
    print(f"\nDecade {dec}0s ({len(pts)} points):")
    if results:
        # Group by N
        by_N = {}
        for B, N in results:
            by_N.setdefault(N, []).append(B)

        print(f"  {len(results)} exact (B,N) pairs, {len(by_N)} distinct N values")
        for N in sorted(by_N)[:5]:
            Bs = sorted(by_N[N])
            print(
                f"    N={N}: B ∈ {{{', '.join(map(str, Bs[:5]))}}}{'...' if len(Bs) > 5 else ''}"
            )

        # Pick the N with smallest valid B range that's "clean"
        # Prefer N where B is a multiple of 5 or 10
        for N in sorted(by_N):
            for B in sorted(by_N[N]):
                if B % 5 == 0:
                    best_BN[dec] = (B, N)
                    break
            if dec in best_BN:
                break
        if dec not in best_BN:
            # Just take the first
            B, N = results[0]
            best_BN[dec] = (B, N)

        B, N = best_BN[dec]
        print(f"  Selected: B={B}, N={N}")
        print(f"    watts = ({B} + {N} * (c % 10)) // 10")
        print("    Verification:")
        for c, w in pts:
            offset = c - dec * 10
            pred = (B + N * offset) // 10
            mark = "✓" if pred == w else "✗"
            print(
                f"      c={c}: ({B} + {N}*{offset}) // 10 = {B + N * offset} // 10 = {pred} "
                f"(actual {w}) {mark}"
            )
    else:
        print("  NO EXACT SOLUTION FOUND")


# ═══════════════════════════════════════════════════════════════════════════
# Full verification against all ground truth
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FULL VERIFICATION")
print("=" * 70)

print("\nFirmware formula: watts = (B[c//10] + N[c//10] * (c%10)) // 10")
print("\nLookup tables:")
print(f"  {'Dec':>3} {'B':>5} {'N':>4}  {'base':>7}  {'slope':>7}")
print("  " + "-" * 35)
for dec in sorted(best_BN):
    B, N = best_BN[dec]
    print(f"  {dec:>3} {B:>5} {N:>4}  {B / 10:>7.1f}  {N / 10:>7.1f}")

print("\nVerification:")
n_errors = 0
for c, w in data:
    dec = c // 10
    if dec not in best_BN:
        print(f"  c={c}: no formula for decade {dec}0s")
        n_errors += 1
        continue
    B, N = best_BN[dec]
    offset = c % 10
    pred = (B + N * offset) // 10
    if pred != w:
        print(f"  c={c}: MISMATCH pred={pred}, actual={w}")
        n_errors += 1

print(
    f"\n  Result: {len(data) - n_errors}/{len(data)} exact matches, {n_errors} errors"
)


# ═══════════════════════════════════════════════════════════════════════════
# Complete watts table (all cadences 0-100)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("COMPLETE PREDICTED WATTS TABLE (cadence 20-89)")
print("=" * 70)

print(
    f"\n  {'Cad':>3} {'Watts':>5}  {'GT':>3}  {'Cad':>3} {'Watts':>5}  {'GT':>3}  "
    f"{'Cad':>3} {'Watts':>5}  {'GT':>3}  {'Cad':>3} {'Watts':>5}  {'GT':>3}"
)
print("  " + "-" * 70)

gt_dict = dict(data)
cols = []
for c in range(20, 90):
    dec = c // 10
    if dec in best_BN:
        B, N = best_BN[dec]
        pred = (B + N * (c % 10)) // 10
        gt_mark = f"{gt_dict[c]}" if c in gt_dict else "?"
        cols.append(f"  {c:>3} {pred:>5}  {gt_mark:>3}")

# Print 4 columns
for i in range(0, len(cols), 4):
    row = cols[i : i + 4]
    print("".join(row))


# ═══════════════════════════════════════════════════════════════════════════
# Pattern in B and N values
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PATTERN ANALYSIS: B and N across decades")
print("=" * 70)

decs = sorted(best_BN)
Bs = [best_BN[d][0] for d in decs]
Ns = [best_BN[d][1] for d in decs]

print(f"\n  B values: {Bs}")
print(f"  N values: {Ns}")
print(f"  B diffs: {[Bs[i + 1] - Bs[i] for i in range(len(Bs) - 1)]}")
print(f"  N diffs: {[Ns[i + 1] - Ns[i] for i in range(len(Ns) - 1)]}")

# Check: B[d+1] = B[d] + 10*N[d] ? (continuity at boundaries)
print("\n  Continuity check: B[d+1] vs B[d] + 10*N[d]")
for i in range(len(decs) - 1):
    d = decs[i]
    B_next_expected = Bs[i] + 10 * Ns[i]
    B_next_actual = Bs[i + 1]
    diff = B_next_actual - B_next_expected
    status = "CONTINUOUS" if diff == 0 else f"JUMP of {diff} ({diff / 10:.1f} watts)"
    print(
        f"    {d}0→{decs[i + 1]}0: {Bs[i]}+10*{Ns[i]}={B_next_expected}, "
        f"actual B={B_next_actual} → {status}"
    )

# N values vs decade: check if N = f(decade)
print("\n  N as function of decade:")
dec_arr = np.array(decs, dtype=float)
N_arr = np.array(Ns, dtype=float)

# Quadratic fit
coeffs = np.polyfit(dec_arr, N_arr, 2)
pred_N = np.polyval(coeffs, dec_arr)
print(f"    Quadratic: N = {coeffs[0]:.4f}*d^2 + {coeffs[1]:.4f}*d + {coeffs[2]:.4f}")
for d, n, pn in zip(decs, Ns, pred_N):
    print(f"      d={d}: N={n}, pred={pn:.1f}, diff={n - pn:.1f}")
