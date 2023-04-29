"""Final watts formula: continuous piecewise-linear with integer arithmetic.

watts = (B[c // 10] + N[c // 10] * (c % 10)) // 10

The continuity constraint B[d+1] = B[d] + 10*N[d] uniquely determines
all parameters given the ground truth data.
"""

import numpy as np

# Ground truth
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
gt = dict(data)

# Continuity-selected parameters (B[d+1] = B[d] + 10*N[d])
# The 20s choice (155, 26) is uniquely determined by continuity with the 30s.
B = {2: 155, 3: 415, 4: 825, 5: 1505, 6: 2425, 7: 3835, 8: 5495}
N = {2: 26, 3: 41, 4: 68, 5: 92, 6: 141, 7: 166, 8: 211}


def watts_formula(c: int) -> int:
    """Compute watts from cadence using the firmware formula."""
    if c <= 0:
        return 0
    d = c // 10
    if d not in B:
        return 0
    return (B[d] + N[d] * (c % 10)) // 10


# ═══════════════════════════════════════════════════════════════════════════
# Continuity verification
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("CONTINUITY VERIFICATION")
print("=" * 60)
print("\n  Constraint: B[d+1] = B[d] + 10*N[d]")
for d in sorted(B)[:-1]:
    expected = B[d] + 10 * N[d]
    actual = B[d + 1]
    status = "✓" if expected == actual else f"✗ (expected {expected})"
    print(
        f"  d={d}→{d + 1}: {B[d]} + 10*{N[d]} = {expected}, B[{d + 1}]={actual} {status}"
    )

# ═══════════════════════════════════════════════════════════════════════════
# Ground truth verification
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("GROUND TRUTH VERIFICATION")
print("=" * 60)

n_match = 0
n_total = len(data)
for c, w in data:
    pred = watts_formula(c)
    if pred == w:
        n_match += 1
    else:
        print(f"  MISMATCH c={c}: predicted={pred}, actual={w}")

print(f"\n  {n_match}/{n_total} exact matches")

# ═══════════════════════════════════════════════════════════════════════════
# Complete table
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("COMPLETE WATTS TABLE (cadence 20-89)")
print("=" * 60)
print("\n  Cad  Watts  GT    Cad  Watts  GT    Cad  Watts  GT")
print("  " + "-" * 54)

rows = []
for c in range(20, 90):
    w = watts_formula(c)
    gt_w = gt.get(c, "")
    mark = "" if gt_w == "" else ("✓" if w == gt_w else "✗")
    rows.append(f"  {c:>3}  {w:>5}  {str(gt_w):>3} {mark}")

for i in range(0, len(rows), 3):
    print("  ".join(rows[i : i + 3]))

# ═══════════════════════════════════════════════════════════════════════════
# Formula summary
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("FIRMWARE FORMULA")
print("=" * 60)
print("""
  watts(c) = (B[c // 10] + N[c // 10] * (c % 10)) // 10

  Decade  B      N    slope/RPM   base(watts)
  ──────  ─────  ───  ─────────   ───────────
  20-29   155    26   2.6          15.5
  30-39   415    41   4.1          41.5
  40-49   825    68   6.8          82.5
  50-59   1505   92   9.2         150.5
  60-69   2425   141  14.1        242.5
  70-79   3835   166  16.6        383.5
  80-89   5495   211  21.1        549.5

  Continuity: B[d+1] = B[d] + 10*N[d] (holds for all decades)

  Implementation: Two 7-entry lookup tables + integer arithmetic.
  Total firmware storage: 14 integers.
""")

# ═══════════════════════════════════════════════════════════════════════════
# N-value analysis: these are the "slopes" per decade (tenths of watts/RPM)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("N-VALUE PATTERN")
print("=" * 60)

N_vals = [N[d] for d in sorted(N)]
print(f"\n  N values: {N_vals}")
print(f"  N diffs:  {[N_vals[i + 1] - N_vals[i] for i in range(len(N_vals) - 1)]}")
print(
    f"  N ratios: {[f'{N_vals[i + 1] / N_vals[i]:.3f}' for i in range(len(N_vals) - 1)]}"
)

# N represents the power derivative at decade midpoints
# If power ∝ RPM^b, then dP/dc ∝ b * RPM^(b-1)
# So N should scale as c^(b-1)
mids = np.array([25, 35, 45, 55, 65, 75, 85], dtype=float)
N_arr = np.array(N_vals, dtype=float)

# Fit log(N) = (b-1)*log(mid) + const
log_N = np.log(N_arr)
log_mid = np.log(mids)
slope, intercept = np.polyfit(log_mid, log_N, 1)
b_implied = slope + 1
print(
    f"\n  Log-linear fit: slope={slope:.4f}, implied power law exponent b={b_implied:.4f}"
)
print(
    f"  (i.e., the underlying calibration curve was roughly watts ∝ c^{b_implied:.2f})"
)

# Show fit quality
pred_N = np.exp(intercept + slope * log_mid)
print(f"\n  {'mid':>5} {'N':>5} {'pred':>7} {'ratio':>7}")
for m, n, pn in zip(mids, N_arr, pred_N):
    print(f"  {int(m):>5} {int(n):>5} {pn:>7.1f} {n / pn:>7.3f}")
