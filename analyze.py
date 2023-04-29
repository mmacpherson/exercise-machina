"""Recover Echo Bike firmware equations from OCR dataset.

Four analysis sections:
1. Cadence → Watts mapping (piecewise-linear integer arithmetic)
2. Distance accumulation (dt sweep, integer arithmetic)
3. Calorie accumulation (K sweep, rounding modes)
4. Display dynamics (transitions, anomalous frames, effective cadence)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.optimize import brentq, curve_fit, linprog

PLOTS = Path("plots")
PLOTS.mkdir(exist_ok=True)

# ── Data loading ────────────────────────────────────────────────────────────

gt = pl.read_csv("top-row-states.csv")

frames = []
with open("frames_1fps.jsonl") as f:
    for line in f:
        frames.append(json.loads(line))
ts = pl.DataFrame(frames)

# Non-zero ground truth for fitting
gt_nz = gt.filter(pl.col("cadence") > 0)
c_gt = gt_nz["cadence"].to_numpy().astype(float)
w_gt = gt_nz["watts"].to_numpy().astype(float)

print(f"Ground truth: {len(gt)} rows ({len(gt_nz)} non-zero)")
print(f"  Cadence range: {int(c_gt.min())}–{int(c_gt.max())} RPM")
print(f"  Watts range: {int(w_gt.min())}–{int(w_gt.max())} W")
print(f"Time series: {len(ts)} frames\n")


def speed_formula(c):
    """Exact: speed = floor(c * 272 / 73) / 10."""
    return (c * 272 // 73) / 10


# Compare as integers (speed_tenths) to avoid float representation issues
gt_nz = gt_nz.with_columns(
    ((pl.col("cadence") * 272) // 73).alias("speed_tenths_pred"),
    (pl.col("speed") * 10).round(0).cast(pl.Int64).alias("speed_tenths"),
)
speed_mismatches = gt_nz.filter(pl.col("speed_tenths") != pl.col("speed_tenths_pred"))
if len(speed_mismatches) == 0:
    print("Speed formula verified: floor(c * 272 / 73) / 10  (all 59 rows)\n")
else:
    print(f"Speed formula: {len(gt_nz) - len(speed_mismatches)}/{len(gt_nz)} match")
    for row in speed_mismatches.iter_rows(named=True):
        print(
            f"  Mismatch cad={int(row['cadence'])}: "
            f"display={row['speed']}, formula={row['speed_tenths_pred'] / 10}"
        )
    print()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: Cadence → Watts mapping
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 72)
print("SECTION 1: Cadence → Watts mapping")
print("=" * 72)

n_gt = len(c_gt)

# ── 1a. Least-squares survey (degrees 1–6, no intercept) ───────────────────
# Provides AIC/BIC comparison; not expected to achieve zero floor errors.

print(
    f"\n{'Deg':>3} {'k':>2} {'MaxRes':>8} {'RSS':>10} {'AIC':>8} {'BIC':>8}"
    f"  {'flr':>4} {'rnd':>4}"
)
print("-" * 58)

poly_ls = {}
for deg in range(1, 7):
    X = np.column_stack([c_gt**p for p in range(1, deg + 1)])
    coeffs_ls, *_ = np.linalg.lstsq(X, w_gt, rcond=None)
    pred = X @ coeffs_ls
    rss = float(np.sum((w_gt - pred) ** 2))
    max_res = float(np.max(np.abs(w_gt - pred)))
    k = deg
    aic = n_gt * np.log(rss / n_gt) + 2 * k
    bic = n_gt * np.log(rss / n_gt) + k * np.log(n_gt)
    flr = int(np.sum(np.floor(pred).astype(int) != w_gt.astype(int)))
    rnd = int(np.sum(np.round(pred).astype(int) != w_gt.astype(int)))
    poly_ls[deg] = dict(
        rss=rss, max_res=max_res, aic=aic, bic=bic, flr_err=flr, rnd_err=rnd
    )
    print(
        f"{deg:>3} {k:>2} {max_res:>8.3f} {rss:>10.1f} {aic:>8.1f} {bic:>8.1f}"
        f"  {flr:>4} {rnd:>4}"
    )

# ── 1b. Power law fit ──────────────────────────────────────────────────────

print("\nPower law: watts = a * cadence^b")
log_c, log_w = np.log(c_gt), np.log(w_gt)
b_ols, lna_ols = np.polyfit(log_c, log_w, 1)


def power_fn(c, a, b):
    return a * c**b


(a_nls, b_nls), _ = curve_fit(power_fn, c_gt, w_gt, p0=[np.exp(lna_ols), b_ols])
pred_pw = power_fn(c_gt, a_nls, b_nls)
flr_pw = int(np.sum(np.floor(pred_pw).astype(int) != w_gt.astype(int)))
rnd_pw = int(np.sum(np.round(pred_pw).astype(int) != w_gt.astype(int)))
print(f"  NLS: a={a_nls:.6f}, b={b_nls:.4f}, floor_err={flr_pw}, round_err={rnd_pw}")

# ── 1c. LP-based floor-exact polynomial search ─────────────────────────────
# Find the minimum-degree polynomial through origin where floor(f(c)) == w
# for all ground truth points. Formulated as a linear feasibility problem:
#   w_j <= a_1*c_j + a_2*c_j^2 + ... + a_d*c_j^d < w_j + 1

print("\nLP search for floor-exact polynomial (degrees 3–12):")

# Normalize cadences for numerical stability
c_scale = 100.0
c_norm = c_gt / c_scale
EPS = 1e-6  # strict inequality margin

lp_result = None
for deg in range(3, 13):
    # Design matrix in normalized coordinates
    X_norm = np.column_stack([c_norm**p for p in range(1, deg + 1)])

    # Constraints: w_j <= X_norm[j] @ a < w_j + 1
    # → -X_norm[j] @ a <= -w_j       (lower bound)
    # →  X_norm[j] @ a <= w_j + 1-eps (upper bound)
    A_ub = np.vstack([-X_norm, X_norm])
    b_ub = np.concatenate([-w_gt, w_gt + 1 - EPS])

    # Objective: minimize sum of residuals = sum(X_norm @ a - w)
    # This centers predictions in [w, w+1) without biasing toward edges
    c_obj = X_norm.sum(axis=0)

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, method="highs")

    if result.success:
        a_norm = result.x
        pred_lp = X_norm @ a_norm
        floor_lp = np.floor(pred_lp).astype(int)
        errors = int(np.sum(floor_lp != w_gt.astype(int)))

        # Convert back to original scale: a_orig_i = a_norm_i / c_scale^i
        a_orig = np.array([a_norm[i] / c_scale ** (i + 1) for i in range(deg)])

        max_resid = float(np.max(pred_lp - w_gt))
        min_resid = float(np.min(pred_lp - w_gt))
        print(
            f"  Deg {deg:>2}: FEASIBLE, floor_err={errors}, "
            f"residuals [{min_resid:.4f}, {max_resid:.4f}]"
        )

        if errors == 0 and lp_result is None:
            lp_result = {
                "deg": deg,
                "a_norm": a_norm,
                "a_orig": a_orig,
                "pred": pred_lp,
                "X_norm": X_norm,
            }
    else:
        print(f"  Deg {deg:>2}: INFEASIBLE")

if lp_result is not None:
    best_deg = lp_result["deg"]
    a_norm_best = lp_result["a_norm"]

    print(f"\n*** EXACT MATCH: degree-{best_deg} polynomial with floor() ***")
    a_orig = lp_result["a_orig"]
    terms = []
    for i, co in enumerate(a_orig):
        p = i + 1
        terms.append(f"{co:.12g}*c" if p == 1 else f"{co:.12g}*c^{p}")
    formula_str = " + ".join(terms)
    print(f"  watts = floor({formula_str})")
else:
    print("\nNo polynomial floor-match exists (degrees 3–12).")
    print("Conclusion: not a global polynomial; testing piecewise structure...")

# ── 1c′. Piecewise-linear integer formula ─────────────────────────────────
# Exact formula discovered via feasibility-interval analysis per decade:
#   watts = (B[c // 10] + N[c // 10] * (c % 10)) // 10
#
# Parameters are uniquely determined by:
#   1. Ground truth matching: all 59 non-zero points exact
#   2. Continuity: B[d+1] = B[d] + 10*N[d] at every decade boundary

WATTS_B = {2: 155, 3: 415, 4: 825, 5: 1505, 6: 2425, 7: 3835, 8: 5495}
WATTS_N = {2: 26, 3: 41, 4: 68, 5: 92, 6: 141, 7: 166, 8: 211}

print("\n  Piecewise-linear formula: watts = (B + N*(c%10)) // 10")
print(f"  {'Dec':>5} {'B':>5} {'N':>4} {'slope':>6}")
for d in sorted(WATTS_B):
    print(
        f"  {d * 10:>2}-{d * 10 + 9:>2} {WATTS_B[d]:>5} {WATTS_N[d]:>4} {WATTS_N[d] / 10:>6.1f}"
    )

# Verify continuity
for d in sorted(WATTS_B)[:-1]:
    expected = WATTS_B[d] + 10 * WATTS_N[d]
    assert expected == WATTS_B[d + 1], f"Continuity broken at {d}0→{d + 1}0"
print("  Continuity B[d+1] = B[d] + 10*N[d]: verified")

# Verify against ground truth
gt_lookup = {int(row["cadence"]): int(row["watts"]) for row in gt.iter_rows(named=True)}
n_err_all = 0
for c_val, w_val in gt_lookup.items():
    d = c_val // 10
    if d in WATTS_B:
        pred = (WATTS_B[d] + WATTS_N[d] * (c_val % 10)) // 10
        if pred != w_val:
            n_err_all += 1
            print(f"    MISMATCH c={c_val}: formula={pred}, actual={w_val}")
print(f"  Ground truth: {len(gt_lookup) - n_err_all}/{len(gt_lookup)} exact")
formula_str = "(B[c//10] + N[c//10]*(c%10)) // 10"


# ── 1d. Segmented regression test ──────────────────────────────────────────

print("\nSegmented regression (2-segment cubic, sweep breakpoints 30–79):")
base_rss = poly_ls[3]["rss"]
best_bp, best_bp_rss = None, base_rss

for bp in range(30, 80):
    lo, hi = c_gt <= bp, c_gt > bp
    if lo.sum() < 4 or hi.sum() < 4:
        continue
    X_lo = np.column_stack([c_gt[lo] ** p for p in range(1, 4)])
    X_hi = np.column_stack([c_gt[hi] ** p for p in range(1, 4)])
    c_lo, *_ = np.linalg.lstsq(X_lo, w_gt[lo], rcond=None)
    c_hi, *_ = np.linalg.lstsq(X_hi, w_gt[hi], rcond=None)
    rss_seg = float(
        np.sum((w_gt[lo] - X_lo @ c_lo) ** 2) + np.sum((w_gt[hi] - X_hi @ c_hi) ** 2)
    )
    if rss_seg < best_bp_rss:
        best_bp_rss = rss_seg
        best_bp = bp

if best_bp is not None:
    improv = (base_rss - best_bp_rss) / base_rss * 100
    print(
        f"  Best breakpoint: cadence={best_bp}, RSS={best_bp_rss:.1f} "
        f"(vs single cubic {base_rss:.1f}, {improv:.1f}% improvement)"
    )
    if improv < 10:
        print("  Negligible — no evidence for piecewise structure")
    else:
        print("  Significant — possible piecewise structure")
else:
    print(f"  No improvement over single-segment fit (RSS={base_rss:.1f})")


# ── 1e. Plots ───────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

c_fine = np.arange(20, 90)
w_fine = np.array(
    [(WATTS_B[c // 10] + WATTS_N[c // 10] * (c % 10)) // 10 for c in c_fine]
)

# Fit quality
ax = axes[0, 0]
ax.scatter(c_gt, w_gt, s=25, zorder=3, label="Ground truth")
ax.plot(c_fine, w_fine, "r-", alpha=0.7, label="Piecewise formula")
for d in sorted(WATTS_B)[1:]:
    ax.axvline(d * 10, color="gray", ls=":", alpha=0.3)
ax.set(xlabel="Cadence (RPM)", ylabel="Watts", title="Cadence → Watts mapping")
ax.legend()
ax.grid(True, alpha=0.3)

# Per-decade slopes
ax = axes[0, 1]
decs = sorted(WATTS_N)
mids = [d * 10 + 5 for d in decs]
slopes_plot = [WATTS_N[d] / 10 for d in decs]
ax.bar(mids, slopes_plot, width=8, alpha=0.6, label="N/10 per decade")
log_fit = np.polyfit(np.log(mids), np.log(slopes_plot), 1)
c_smooth = np.linspace(20, 90, 200)
ax.plot(
    c_smooth,
    np.exp(np.polyval(log_fit, np.log(c_smooth))),
    "r--",
    lw=1.5,
    label=f"~c^{log_fit[0]:.2f}",
)
ax.set(
    xlabel="Cadence (RPM)", ylabel="Slope (watts/RPM)", title="Per-decade slopes (N/10)"
)
ax.legend()
ax.grid(True, alpha=0.3)

# First differences (consecutive cadences only)
ax = axes[1, 0]
dc = np.diff(c_gt)
dw = np.diff(w_gt)
consec = dc == 1
c_mid = c_gt[:-1][consec] + 0.5
ax.bar(c_mid, dw[consec], width=0.6, alpha=0.5, label="Observed Δw (consecutive)")
for d in sorted(WATTS_N):
    slope = WATTS_N[d] / 10
    ax.hlines(slope, d * 10, d * 10 + 10, colors="red", lw=2, alpha=0.7)
ax.set(
    xlabel="Cadence (RPM)",
    ylabel="ΔWatts / ΔRPM",
    title="First differences with per-decade slopes",
)
ax.legend()
ax.grid(True, alpha=0.3)

# Formula verification (error = 0 for all points)
ax = axes[1, 1]
w_pred = np.array(
    [
        (WATTS_B[int(c) // 10] + WATTS_N[int(c) // 10] * (int(c) % 10)) // 10
        for c in c_gt
    ]
)
err = w_gt.astype(int) - w_pred
ax.stem(c_gt, err, linefmt="g-", markerfmt="go", basefmt="k-")
ax.axhline(0, color="k", lw=0.5)
n_err_plot = int(np.sum(err != 0))
ax.set(
    xlabel="Cadence (RPM)",
    ylabel="Error",
    title=f"Formula verification: {n_err_plot} errors / {len(c_gt)} points",
)
ax.grid(True, alpha=0.3)

fig.suptitle("Section 1: Cadence → Watts", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(PLOTS / "section1_watts_mapping.png", dpi=150)
print(f"\nSaved: {PLOTS / 'section1_watts_mapping.png'}")


# ── Helper functions built from Section 1 results ───────────────────────────


def watts_continuous(c):
    """Continuous watts from piecewise formula (non-truncated). Scalar or array."""
    c = np.atleast_1d(np.asarray(c, dtype=float))
    result = np.empty_like(c)
    for i, ci in enumerate(c):
        d = int(ci) // 10
        if d in WATTS_B:
            result[i] = (WATTS_B[d] + WATTS_N[d] * (ci - d * 10)) / 10
        else:
            result[i] = 0.0
    return result if result.size > 1 else float(result[0])


def watts_formula(c):
    """Firmware watts from piecewise integer formula. Scalar or array input."""
    c = np.atleast_1d(np.asarray(c, dtype=int))
    result = np.array(
        [
            (WATTS_B.get(ci // 10, 0) + WATTS_N.get(ci // 10, 0) * (ci % 10)) // 10
            for ci in c
        ]
    )
    return result if result.size > 1 else int(result[0])


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: Distance accumulation
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("SECTION 2: Distance accumulation")
print("=" * 72)

# Workout timer resets to 00:00 at frame 10; first non-zero cadence at frame 16.
WORKOUT_START = 10  # timer = 00:00
PEDAL_START = 16  # first cadence > 0
active = ts.filter(pl.col("frame") >= WORKOUT_START)
n_active = len(active)
print(
    f"\nActive workout: frames {WORKOUT_START}–{ts['frame'].max()} ({n_active} frames)"
)

actual_dist = active["distance"].to_numpy()

# ── 2a. Vectorized dt sweep ─────────────────────────────────────────────────

dt_range = np.linspace(0.97, 1.03, 6001)
speed_displayed = active["speed"].to_numpy()
speed_from_cadence = np.array([speed_formula(c) for c in active["cadence"].to_list()])
speed_continuous = 60 * active["cadence"].to_numpy().astype(float) / 161

print("\ndt sweep (minimizing total |error| across all frames):")

best_dt_results = {}
for label, speeds in [
    ("displayed", speed_displayed),
    ("formula", speed_from_cadence),
    ("continuous", speed_continuous),
]:
    base_cumsum = np.cumsum(speeds / 3600)
    cum_dist = base_cumsum[:, None] * dt_range[None, :]
    disp_dist = np.floor(100 * cum_dist) / 100
    errors = np.sum(np.abs(disp_dist - actual_dist[:, None]), axis=0)

    idx = np.argmin(errors)
    best_dt_results[label] = {
        "dt": dt_range[idx],
        "error": errors[idx],
        "errors_curve": errors,
    }
    print(f"  {label:>12}: dt={dt_range[idx]:.5f}, total_err={errors[idx]:.4f}")

best_speed_label = min(best_dt_results, key=lambda k: best_dt_results[k]["error"])
dt_opt = best_dt_results[best_speed_label]["dt"]
print(f"\n  Best: {best_speed_label} speed, dt = {dt_opt:.5f}")

# ── 2b. Integer arithmetic hypothesis ───────────────────────────────────────

print("\nInteger arithmetic test (accumulate speed_tenths, divide by divisor):")

# Sweep divisor to find best integer model
speed_list = active["speed"].to_list()
dist_list = active["distance"].to_list()
best_div, best_div_err = None, float("inf")
for divisor in range(350, 370):
    acc = 0
    total_err = 0.0
    for spd, dist in zip(speed_list, dist_list):
        speed_tenths = int(round(spd * 10))
        acc += speed_tenths
        sim = (acc // divisor) / 100
        total_err += abs(sim - dist)
    if total_err < best_div_err:
        best_div_err = total_err
        best_div = divisor
print(f"  Best integer divisor={best_div}, total_err={best_div_err:.4f}")
print(
    f"  (360 = 3600/10 would be exact 1Hz; {best_div} implies "
    f"effective dt={360 / best_div:.5f})"
)

# ── 2c. Detailed simulation ────────────────────────────────────────────────

if best_speed_label == "displayed":
    sim_speeds = speed_displayed
elif best_speed_label == "formula":
    sim_speeds = speed_from_cadence
else:
    sim_speeds = speed_continuous

cum_dist = np.cumsum(sim_speeds / 3600) * dt_opt
sim_dist = np.floor(100 * cum_dist) / 100
dist_err = actual_dist - sim_dist
active = active.with_columns(
    pl.Series("sim_dist", sim_dist),
    pl.Series("dist_err", dist_err),
)
max_dist_err = float(np.max(np.abs(dist_err)))

print(f"\nDetailed simulation (dt={dt_opt:.5f}, {best_speed_label} speed):")
print(f"  Max |error|: {max_dist_err:.4f} miles")
print(f"  Final: actual={active['distance'][-1]:.2f}, sim={sim_dist[-1]:.2f}")

# ── 2d. Plot ────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

frame_arr = active["frame"].to_numpy()

ax = axes[0, 0]
ax.plot(frame_arr, actual_dist, "b-", lw=1.5, label="Actual")
ax.plot(frame_arr, sim_dist, "r--", lw=1, label=f"Sim (dt={dt_opt:.4f})")
ax.set(xlabel="Frame", ylabel="Distance (mi)", title="Distance: actual vs simulated")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(frame_arr, dist_err, "g-", lw=1)
ax.axhline(0, color="k", lw=0.5)
ax.set(
    xlabel="Frame",
    ylabel="Error (mi)",
    title=f"Distance residuals (max |err| = {max_dist_err:.4f})",
)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
for label, res in best_dt_results.items():
    ax.plot(dt_range, res["errors_curve"], lw=1, label=label)
ax.axvline(dt_opt, color="k", ls="--", alpha=0.5, label=f"dt={dt_opt:.4f}")
ax.set(xlabel="dt (seconds)", ylabel="Total |error|", title="dt sweep")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(frame_arr, speed_displayed, "b-", lw=1, alpha=0.7, label="Speed (mph)")
ax.set(xlabel="Frame", ylabel="Speed (mph)", title="Speed profile")
ax.grid(True, alpha=0.3)

fig.suptitle("Section 2: Distance accumulation", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(PLOTS / "section2_distance.png", dpi=150)
print(f"\nSaved: {PLOTS / 'section2_distance.png'}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: Calorie accumulation
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("SECTION 3: Calorie accumulation")
print("=" * 72)

actual_cal = active["calories"].to_numpy().astype(float)
ocr_watts = active["watts"].to_numpy().astype(float)
formula_watts_arr = np.array(
    [
        float(watts_formula(int(c))) if c > 0 else 0.0
        for c in active["cadence"].to_list()
    ]
)

# ── 3a. K sweep (vectorized) ───────────────────────────────────────────────

k_range = np.arange(1100, 1301, dtype=float)

print(f"\nK sweep [1100, 1300], dt={dt_opt:.5f}:")
print(f"  {'Source':>14} {'Rounding':>8} {'Best K':>7} {'TotErr':>8}")
print("  " + "-" * 45)

sweep_results = {}
for ws_label, watts_arr in [
    ("ocr_watts", ocr_watts),
    ("formula_watts", formula_watts_arr),
]:
    energy = np.cumsum(watts_arr * dt_opt)

    for rnd_label, rnd_fn in [
        ("floor", np.floor),
        ("round", np.round),
        ("ceil", np.ceil),
    ]:
        sim_cal = rnd_fn(energy[:, None] / k_range[None, :])
        errors = np.sum(np.abs(sim_cal - actual_cal[:, None]), axis=0)

        idx = np.argmin(errors)
        key = f"{ws_label}_{rnd_label}"
        sweep_results[key] = {
            "K": k_range[idx],
            "error": errors[idx],
            "errors_curve": errors,
        }
        print(
            f"  {ws_label:>14} {rnd_label:>8} {k_range[idx]:>7.0f} {errors[idx]:>8.0f}"
        )

best_cal_key = min(sweep_results, key=lambda k: sweep_results[k]["error"])
K_coarse = sweep_results[best_cal_key]["K"]
print(f"\n  Best: {best_cal_key}, K={K_coarse:.0f}")

# ── 3b. Fine sweep around best K ───────────────────────────────────────────

ws_label_best, rnd_label_best = best_cal_key.rsplit("_", 1)
watts_best = ocr_watts if "ocr" in ws_label_best else formula_watts_arr
rnd_fn_best = {"floor": np.floor, "round": np.round, "ceil": np.ceil}[rnd_label_best]

k_fine = np.linspace(K_coarse - 10, K_coarse + 10, 20001)
energy_best = np.cumsum(watts_best * dt_opt)
sim_cal_fine = rnd_fn_best(energy_best[:, None] / k_fine[None, :])
fine_errors = np.sum(np.abs(sim_cal_fine - actual_cal[:, None]), axis=0)

K_opt = k_fine[np.argmin(fine_errors)]
min_fine_err = fine_errors.min()

optimal_mask = fine_errors == min_fine_err
k_optimal = k_fine[optimal_mask]
print(f"\n  Fine sweep: K={K_opt:.4f}, min total_error={min_fine_err:.0f}")
print(f"  Optimal K range: [{k_optimal.min():.4f}, {k_optimal.max():.4f}]")

# ── 3c. Physical interpretation ─────────────────────────────────────────────

print("\n  Physical interpretation:")
print(f"    K = {K_opt:.2f} J/cal")
print(f"    4184 / K = {4184 / K_opt:.4f} (metabolic multiplier)")
print(f"    Implied efficiency = K / 4184 = {K_opt / 4184 * 100:.1f}%")

print("\n    Candidate rational forms:")
for num, den in [
    (4184, 3),
    (4184, 3.45),
    (4184, 3.5),
    (4184, 4),
]:
    ratio = num / den
    if abs(ratio - K_opt) < 5:
        marker = " *** CLOSE" if abs(ratio - K_opt) < 1 else ""
        print(f"      {num}/{den} = {ratio:.4f}{marker}")

# Test exact integer K values near the optimum
print("\n    Testing integer K values:")
for K_int in range(int(K_opt) - 3, int(K_opt) + 4):
    sim = rnd_fn_best(energy_best / K_int)
    err = int(np.sum(np.abs(sim - actual_cal)))
    max_err = int(np.max(np.abs(sim - actual_cal)))
    print(f"      K={K_int}: total_err={err}, max_err={max_err}")

# ── 3d. Joint dt × K optimization ──────────────────────────────────────────

print("\n  Joint dt × K sweep (coarse):")
dt_cal_range = np.linspace(0.98, 1.02, 401)
k_joint_range = np.linspace(K_opt - 5, K_opt + 5, 1001)

best_joint_err = float("inf")
best_joint_dt = dt_opt
best_joint_K = K_opt

for dt_j in dt_cal_range:
    energy_j = np.cumsum(watts_best * dt_j)
    sim_j = rnd_fn_best(energy_j[:, None] / k_joint_range[None, :])
    errs_j = np.sum(np.abs(sim_j - actual_cal[:, None]), axis=0)
    idx_j = np.argmin(errs_j)
    if errs_j[idx_j] < best_joint_err:
        best_joint_err = errs_j[idx_j]
        best_joint_dt = dt_j
        best_joint_K = k_joint_range[idx_j]

print(
    f"    Best joint: dt={best_joint_dt:.5f}, K={best_joint_K:.4f}, "
    f"total_err={best_joint_err:.0f}"
)

# ── 3e. Detailed simulation ────────────────────────────────────────────────

sim_cal_best = rnd_fn_best(energy_best / K_opt).astype(int)
cal_err = actual_cal.astype(int) - sim_cal_best
active = active.with_columns(
    pl.Series("sim_cal", sim_cal_best),
    pl.Series("cal_err", cal_err),
)
max_cal_err = int(np.max(np.abs(cal_err)))
print(f"\n  Simulation with K={K_opt:.2f}:")
print(f"  Max |calorie error|: {max_cal_err}")
print(f"  Final: actual={int(active['calories'][-1])}, sim={sim_cal_best[-1]}")

# ── 3f. Plot ────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.plot(frame_arr, actual_cal, "b-", lw=1.5, label="Actual")
ax.plot(frame_arr, sim_cal_best, "r--", lw=1, label=f"Sim (K={K_opt:.1f})")
ax.set(xlabel="Frame", ylabel="Calories", title="Calories: actual vs simulated")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(frame_arr, cal_err, "g-", lw=1)
ax.axhline(0, color="k", lw=0.5)
ax.set(
    xlabel="Frame",
    ylabel="Error",
    title=f"Calorie residuals (max |err| = {max_cal_err})",
)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(k_fine, fine_errors, "b-", lw=1.5)
ax.axvline(K_opt, color="r", ls="--", label=f"K={K_opt:.1f}")
ax.set(xlabel="K (J/cal)", ylabel="Total |error|", title="K fine sweep")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
for key, res in sweep_results.items():
    if "ocr" in key:
        ax.plot(k_range, res["errors_curve"], lw=1, label=key)
ax.axvline(K_coarse, color="k", ls="--", alpha=0.5)
ax.set(
    xlabel="K (J/cal)",
    ylabel="Total |error|",
    title="K sweep by rounding mode (OCR watts)",
)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle("Section 3: Calorie accumulation", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(PLOTS / "section3_calories.png", dpi=150)
print(f"\nSaved: {PLOTS / 'section3_calories.png'}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: Display dynamics
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("SECTION 4: Display dynamics")
print("=" * 72)

# ── 4a. Build ground truth lookup and find anomalous frames ────────────────

active_nz = active.filter(pl.col("cadence") > 0)

# Compute expected watts from our model
expected_watts = [watts_formula(int(c)) for c in active_nz["cadence"].to_list()]
expected_speed = [speed_formula(c) for c in active_nz["cadence"].to_list()]
active_nz = active_nz.with_columns(
    pl.Series("expected_watts", expected_watts),
    pl.Series("expected_speed", expected_speed),
).with_columns(
    (pl.col("watts") == pl.col("expected_watts")).alias("watts_match"),
    (pl.col("speed") == pl.col("expected_speed")).alias("speed_match"),
)

n_consistent = active_nz["watts_match"].sum()
anomalous = active_nz.filter(~pl.col("watts_match"))

print(f"\n  Non-zero frames: {len(active_nz)}")
print(f"  Watts consistent: {n_consistent}")
print(f"  Watts anomalous: {len(anomalous)}")

speed_anomalous = active_nz.filter(~pl.col("speed_match"))
print(f"  Speed anomalous: {len(speed_anomalous)}")

# ── 4b. Anomalous frame analysis ───────────────────────────────────────────

if len(anomalous) > 0:
    # Only print first 30 and summarize the rest
    n_show = min(30, len(anomalous))
    print(f"\n  First {n_show} anomalous frames:")
    print(
        f"  {'Frame':>5} {'Cad':>4} {'Watts':>5} {'Expct':>5} {'Diff':>5} {'EffCad':>7}"
    )
    print("  " + "-" * 42)

    for idx, row in enumerate(anomalous.iter_rows(named=True)):
        if idx >= n_show:
            break
        expected_w = int(row["expected_watts"])
        diff = int(row["watts"]) - expected_w

        target_w = float(row["watts"])
        try:
            eff_c = brentq(lambda c: watts_continuous(c) - (target_w + 0.5), 1, 120)
        except ValueError:
            eff_c = float("nan")

        print(
            f"  {int(row['frame']):>5} {int(row['cadence']):>4} {int(row['watts']):>5} "
            f"{expected_w:>5} {diff:>+5} {eff_c:>7.2f}"
        )

    if len(anomalous) > n_show:
        print(f"  ... and {len(anomalous) - n_show} more")

    # Summarize: what fraction of anomalous frames are off by exactly 1-2?
    diffs = anomalous["watts"] - anomalous["expected_watts"]
    print("\n  Anomalous watts difference distribution:")
    for d in sorted(diffs.unique().to_list()):
        count = int((diffs == d).sum())
        print(f"    Δ={int(d):+3d}: {count} frames")

# ── 4c. Transition detection ───────────────────────────────────────────────

cad_arr = active_nz["cadence"].to_numpy()
frame_nz_arr = active_nz["frame"].to_numpy()

transitions = []
for i in range(1, len(cad_arr)):
    dc = abs(cad_arr[i] - cad_arr[i - 1])
    if dc >= 3:
        transitions.append(
            {
                "frame": int(frame_nz_arr[i]),
                "from_cad": int(cad_arr[i - 1]),
                "to_cad": int(cad_arr[i]),
                "delta": int(cad_arr[i] - cad_arr[i - 1]),
            }
        )

print(f"\n  Transitions (|Δcadence| >= 3): {len(transitions)}")
for t in transitions[:15]:
    nearby_anom = anomalous.filter(
        (pl.col("frame") >= t["frame"] - 2) & (pl.col("frame") <= t["frame"] + 2)
    )
    anom_flag = " *ANOM*" if len(nearby_anom) > 0 else ""
    print(
        f"    Frame {t['frame']:>3}: {t['from_cad']:>2} → {t['to_cad']:>2} "
        f"(Δ={t['delta']:+d}){anom_flag}"
    )
if len(transitions) > 15:
    print(f"    ... and {len(transitions) - 15} more")

# ── 4d. Steady-state consistency ───────────────────────────────────────────

steady_runs = []
run_start = 0
for i in range(1, len(cad_arr)):
    if cad_arr[i] != cad_arr[run_start]:
        if i - run_start >= 3:
            steady_runs.append(
                (
                    int(frame_nz_arr[run_start]),
                    int(frame_nz_arr[i - 1]),
                    int(cad_arr[run_start]),
                    i - run_start,
                )
            )
        run_start = i
if len(cad_arr) - run_start >= 3:
    steady_runs.append(
        (
            int(frame_nz_arr[run_start]),
            int(frame_nz_arr[-1]),
            int(cad_arr[run_start]),
            len(cad_arr) - run_start,
        )
    )

print(f"\n  Steady-state periods (>= 3 frames same cadence): {len(steady_runs)}")

steady_errors = 0
for start_f, end_f, cadence, length in steady_runs:
    subset = active_nz.filter((pl.col("frame") >= start_f) & (pl.col("frame") <= end_f))
    expected = watts_formula(cadence)
    errs = int((subset["watts"] != expected).sum())
    if errs > 0:
        steady_errors += errs
        print(
            f"    Frames {start_f}–{end_f} (cad={cadence}, {length}f): "
            f"{errs} watts mismatches (display={subset['watts'][0]}, "
            f"model={expected})"
        )

if steady_errors == 0:
    print("  All steady-state periods: watts perfectly consistent with model")
else:
    print(f"\n  Total steady-state mismatches: {steady_errors}")
    print(
        "  (Systematic offset suggests model needs refinement or firmware uses "
        "a lookup table)"
    )

# ── 4e. Ramp analysis ──────────────────────────────────────────────────────

print("\n  Ramp analysis (startup and large transitions):")

ramp_region = active_nz.filter((pl.col("frame") >= 360) & (pl.col("frame") <= 380))
if len(ramp_region) > 0:
    print("\n    Late ramp (frames 360–380):")
    for row in ramp_region.iter_rows(named=True):
        match = "ok" if row["watts_match"] else "ANOM"
        print(
            f"      f={int(row['frame']):>3} cad={int(row['cadence']):>2} "
            f"w={int(row['watts']):>3} exp={int(row['expected_watts']):>3} "
            f"spd={row['speed']:.1f} [{match}]"
        )

startup = active_nz.filter(pl.col("frame") <= 25)
if len(startup) > 0:
    print("\n    Startup ramp (frames <= 25):")
    for row in startup.iter_rows(named=True):
        match = "ok" if row["watts_match"] else "ANOM"
        print(
            f"      f={int(row['frame']):>3} cad={int(row['cadence']):>2} "
            f"w={int(row['watts']):>3} exp={int(row['expected_watts']):>3} "
            f"spd={row['speed']:.1f} [{match}]"
        )

# ── 4f. Plot ────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

trans_frames = set()
for t in transitions:
    for f in range(t["frame"] - 2, t["frame"] + 3):
        trans_frames.add(f)

all_frame_arr = active["frame"].to_numpy()
all_cad_arr = active["cadence"].to_numpy()
all_watts_arr = active["watts"].to_numpy()
all_speed_arr = active["speed"].to_numpy()

# Cadence
ax = axes[0]
ax.plot(all_frame_arr, all_cad_arr, "b-", lw=1, label="Cadence (RPM)")
for f in trans_frames:
    ax.axvspan(f - 0.5, f + 0.5, alpha=0.08, color="red")
if len(anomalous) > 0:
    ax.scatter(
        anomalous["frame"].to_numpy(),
        anomalous["cadence"].to_numpy(),
        color="red",
        s=15,
        zorder=3,
        label=f"Anomalous ({len(anomalous)})",
    )
ax.set(
    ylabel="Cadence (RPM)", title="Display dynamics — transition regions highlighted"
)
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# Watts
ax = axes[1]
ax.plot(all_frame_arr, all_watts_arr, "b-", lw=1, label="Watts (displayed)")
active_exp_watts = np.array(
    [watts_formula(int(c)) if c > 0 else 0 for c in active["cadence"].to_list()]
)
ax.plot(
    all_frame_arr,
    active_exp_watts,
    "r--",
    lw=0.8,
    alpha=0.5,
    label="Model from cadence",
)
for f in trans_frames:
    ax.axvspan(f - 0.5, f + 0.5, alpha=0.08, color="red")
if len(anomalous) > 0:
    ax.scatter(
        anomalous["frame"].to_numpy(),
        anomalous["watts"].to_numpy(),
        color="red",
        s=15,
        zorder=3,
    )
ax.set(ylabel="Watts")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# Speed
ax = axes[2]
ax.plot(all_frame_arr, all_speed_arr, "b-", lw=1, label="Speed (displayed)")
active_exp_speed = np.array([speed_formula(c) for c in active["cadence"].to_list()])
ax.plot(
    all_frame_arr,
    active_exp_speed,
    "r--",
    lw=0.8,
    alpha=0.5,
    label="Formula from cadence",
)
for f in trans_frames:
    ax.axvspan(f - 0.5, f + 0.5, alpha=0.08, color="red")
ax.set(xlabel="Frame", ylabel="Speed (mph)")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

fig.suptitle("Section 4: Display dynamics", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(PLOTS / "section4_display_dynamics.png", dpi=150)
print(f"\nSaved: {PLOTS / 'section4_display_dynamics.png'}")


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
print("\n  Speed:    floor(c * 272 / 73) / 10")
print("  Watts:    (B[c//10] + N[c//10] * (c%10)) // 10")
print(f"            B = {dict(sorted(WATTS_B.items()))}")
print(f"            N = {dict(sorted(WATTS_N.items()))}")
print(f"  Distance: cumsum(speed/3600 * {dt_opt:.5f}), display = floor(100*d)/100")
print(f"  Calories: {rnd_label_best}(cumsum(watts*{dt_opt:.5f}) / {K_opt:.2f})")

print("\n  Verification:")
print(
    f"    Speed:    {len(gt_nz) - len(speed_mismatches)}/{len(gt_nz)} "
    f"ground truth match"
)
print(f"    Watts:    {n_err_all} errors on {len(gt_lookup)} ground truth rows")
print(f"    Distance: max |error| = {max_dist_err:.4f} miles")
print(f"    Calories: max |error| = {max_cal_err}")
print(
    f"    Display:  {n_consistent}/{len(active_nz)} frames consistent, "
    f"{len(anomalous)} anomalous"
)
