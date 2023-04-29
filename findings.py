import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reverse-Engineering the Rogue Echo Bike Firmware

    We recovered every equation the Rogue Echo Bike uses to compute its display
    fields — speed, watts, distance, and calories — by filming a workout,
    OCR'ing each frame, and fitting integer-arithmetic models to the data. The
    firmware turns out to be remarkably simple: one sensor (cadence) feeds two
    lookup formulas and two accumulators, all using integer-only arithmetic with
    no filtering or smoothing.

    ## Dataset

    A single 6:37 workout was filmed at 29.97 fps (iPhone, HEVC). Frames were
    extracted at two sample rates with `ffmpeg`, then OCR'd (Claude Sonnet
    vision) to read the seven-segment LCD:

    | Dataset | Rate | Frames | Purpose |
    |---------|------|--------|---------|
    | 1 fps   | 1/s  | 408    | Equation discovery |
    | 4 fps   | 4/s  | 1,631  | Temporal analysis, clock measurement |

    A QC pipeline corrects OCR errors: top-row removal, formula consistency
    checks, monotonic enforcement, and manual patches for LCD digit confusion
    (e.g., `4↔9` on seven-segment displays).
    """)
    return


@app.cell
def _(gt_nz, mo, ts_1fps, ts_4fps):
    _cad = gt_nz["cadence"]
    mo.md(
        f"**{len(gt_nz)} unique cadence–watts pairs** observed across "
        f"cadence {int(_cad.min())}–{int(_cad.max())} RPM, "
        f"from {len(ts_1fps)} frames (1fps) and {len(ts_4fps)} frames (4fps) "
        f"over a 6:37 / 407.85s workout."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Speed Formula

    Speed is a pure function of cadence with no state or hysteresis:

    $$
    \text{speed} = \left\lfloor c \cdot \frac{272}{73} \right\rfloor \Big/ 10
    $$

    The ratio $272/73 \approx 3.726$ converts RPM to mph in tenths.
    The integer floor means the firmware does `c * 272 / 73` in integer
    division, yielding speed in tenths of mph. This matches **all 59
    non-zero ground-truth rows** with zero error.
    """)
    return


@app.cell
def _(
    aes,
    c_gt,
    geom_abline,
    geom_point,
    ggplot,
    labs,
    np,
    s_gt,
    speed_formula,
    theme_minimal,
):
    _s_pred = np.array([speed_formula(int(c)) for c in c_gt])
    (
        ggplot(
            {"observed": s_gt.tolist(), "predicted": _s_pred.tolist()},
            aes(x="predicted", y="observed"),
        )
        + geom_point(size=3, alpha=0.7, color="#2196F3")
        + geom_abline(slope=1, intercept=0, color="red", linetype="dashed")
        + labs(
            x="Predicted speed (mph)",
            y="Observed speed (mph)",
            title="Speed: observed vs predicted (all 59 points on y = x)",
        )
        + theme_minimal()
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Watts Formula

    Watts use a **piecewise-linear integer formula** with a different slope
    per decade of cadence (20s, 30s, …, 80s):

    $$
    \text{watts} = \left\lfloor \frac{B[d] + N[d] \cdot (c \bmod 10)}{10} \right\rfloor,
    \quad d = \lfloor c / 10 \rfloor
    $$

    The parameters are linked by a **continuity constraint** at every decade
    boundary: $B[d{+}1] = B[d] + 10 \, N[d]$. So only $B[2]$ and the 7
    slopes $N[2..8]$ are free — **8 integers** define the entire power curve.
    """)
    return


@app.cell
def _(WATTS_B, WATTS_N, mo):
    _rows = []
    for _d in sorted(WATTS_B):
        _lo, _hi = _d * 10, _d * 10 + 9
        _w_lo = (WATTS_B[_d] + WATTS_N[_d] * 0) // 10
        _w_hi = (WATTS_B[_d] + WATTS_N[_d] * 9) // 10
        _rows.append(
            f"| {_lo}–{_hi} | {WATTS_B[_d]:>5} | {WATTS_N[_d]:>3} "
            f"| {WATTS_N[_d] / 10:.1f} | {_w_lo}–{_w_hi} |"
        )
    mo.md(
        "| Cadence | B | N | Slope (W/RPM) | Watts range |\n"
        "|---------|----:|----:|--------------:|------------:|\n" + "\n".join(_rows)
    )
    return


@app.cell
def _(
    WATTS_B,
    aes,
    c_gt,
    geom_line,
    geom_point,
    geom_vline,
    ggplot,
    labs,
    np,
    theme_minimal,
    w_gt,
    watts_formula,
):
    _c_fine = np.arange(20, 90)
    _w_fine = np.array([watts_formula(c) for c in _c_fine]).astype(float)
    _bounds = [float(d * 10) for d in sorted(WATTS_B)][1:]
    (
        ggplot()
        + geom_line(
            aes(x="cadence", y="watts"),
            data={"cadence": _c_fine.astype(float).tolist(), "watts": _w_fine.tolist()},
            color="red",
            alpha=0.7,
        )
        + geom_point(
            aes(x="cadence", y="watts"),
            data={"cadence": c_gt.tolist(), "watts": w_gt.tolist()},
            size=3,
            color="#2196F3",
        )
        + geom_vline(xintercept=_bounds, color="gray", linetype="dotted", alpha=0.5)
        + labs(
            x="Cadence (RPM)",
            y="Watts",
            title="Cadence → Watts: piecewise formula (0 errors / 59 points)",
        )
        + theme_minimal()
    )
    return


@app.cell
def _(WATTS_N, aes, geom_bar, ggplot, labs, np, theme_minimal):
    _decs = sorted(WATTS_N)
    _mids = [float(d * 10 + 5) for d in _decs]
    _slopes = [WATTS_N[d] / 10 for d in _decs]
    _alpha = np.polyfit(np.log(_mids), np.log(_slopes), 1)[0]
    (
        ggplot(
            {"cadence_mid": _mids, "slope": _slopes}, aes(x="cadence_mid", y="slope")
        )
        + geom_bar(stat="identity", fill="#FF9800", alpha=0.7, width=8)
        + labs(
            x="Cadence (RPM, decade midpoint)",
            y="Slope N/10 (watts per RPM)",
            title=f"Per-decade slopes — underlying power law ~ c^{_alpha:.2f}",
        )
        + theme_minimal()
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    The slopes grow roughly as $c^{1.75}$, implying $\text{watts} \sim c^{2.75}$
    — consistent with air resistance on a fan flywheel ($P \propto \omega^3$,
    reduced slightly by bearing friction). The piecewise-linear approximation
    with integer coefficients perfectly reproduces the firmware's displayed values.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Distance Accumulation

    Distance uses an **integer accumulator** incremented every firmware tick:

    ```
    acc += speed_tenths          # = floor(c * 272 / 73)
    display = acc // 360 / 100   # X.XX miles
    ```

    The divisor $360 = 3600/10$ converts tenths-of-mph · seconds to miles.
    The firmware tick period $\Delta t \approx 1.0006\text{s}$ (measured below).
    """)
    return


@app.cell
def _(
    aes,
    geom_hline,
    geom_line,
    ggplot,
    labs,
    mo,
    np,
    pl,
    speed_formula,
    theme_minimal,
    ts_1fps,
):
    _active = ts_1fps.filter(pl.col("frame") >= 10)
    _speeds = np.array([speed_formula(int(c)) for c in _active["cadence"].to_list()])
    _actual = _active["distance"].to_numpy()
    _frames = _active["frame"].to_numpy().astype(float)

    # Sweep dt to find best-fit tick period
    _dt_range = np.linspace(0.97, 1.03, 6001)
    _cumsum = np.cumsum(_speeds / 3600)
    _errors = np.sum(
        np.abs(
            np.floor(100 * _cumsum[:, None] * _dt_range[None, :]) / 100
            - _actual[:, None]
        ),
        axis=0,
    )
    _dt_opt = _dt_range[np.argmin(_errors)]
    _sim = np.floor(100 * _cumsum * _dt_opt) / 100
    _res = _actual - _sim
    _max_err = float(np.max(np.abs(_res)))

    _p1 = (
        ggplot(
            {
                "frame": np.concatenate([_frames, _frames]).tolist(),
                "distance": np.concatenate([_actual, _sim]).tolist(),
                "series": ["Observed"] * len(_frames) + ["Simulated"] * len(_frames),
            },
            aes(x="frame", y="distance", color="series"),
        )
        + geom_line()
        + labs(
            x="Frame",
            y="Distance (mi)",
            title=f"Distance: observed vs simulated (dt = {_dt_opt:.5f} s)",
        )
        + theme_minimal()
    )
    _p2 = (
        ggplot(
            {"frame": _frames.tolist(), "residual": _res.tolist()},
            aes(x="frame", y="residual"),
        )
        + geom_line(color="#4CAF50")
        + geom_hline(yintercept=0, linetype="dashed", alpha=0.5)
        + labs(
            x="Frame",
            y="Error (mi)",
            title=f"Distance residuals (max |err| = {_max_err:.4f} mi)",
        )
        + theme_minimal()
    )
    mo.vstack([_p1, _p2])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Calorie Accumulation

    Same accumulator pattern:

    ```
    acc += watts
    display = acc // K           # K ≈ 1212 J per displayed calorie
    ```

    The firmware uses **floor** rounding (confirmed by 4fps data — `round`
    produces systematically worse fits). The divisor $K \approx 1212$ implies
    a metabolic multiplier $4184 / K \approx 3.45$ — between Concept2's
    standard $3.5\times$ and a conservative $3.0\times$.
    """)
    return


@app.cell
def _(
    aes,
    geom_line,
    geom_vline,
    ggplot,
    labs,
    np,
    pl,
    speed_formula,
    theme_minimal,
    ts_1fps,
    watts_formula,
):
    # K sweep: compare floor vs round rounding
    _active = ts_1fps.filter(pl.col("frame") >= 10)

    # Recover dt from distance fit
    _speeds = np.array([speed_formula(int(c)) for c in _active["cadence"].to_list()])
    _dt_range = np.linspace(0.97, 1.03, 6001)
    _cumsum = np.cumsum(_speeds / 3600)
    _d_err = np.sum(
        np.abs(
            np.floor(100 * _cumsum[:, None] * _dt_range[None, :]) / 100
            - _active["distance"].to_numpy()[:, None]
        ),
        axis=0,
    )
    _dt = _dt_range[np.argmin(_d_err)]

    _watts = np.array(
        [
            float(watts_formula(int(c))) if c > 0 else 0.0
            for c in _active["cadence"].to_list()
        ]
    )
    _energy = np.cumsum(_watts * _dt)
    _cal = _active["calories"].to_numpy().astype(float)
    _k_range = np.linspace(1100, 1300, 2001)

    _res = {}
    for _label, _fn in [("floor", np.floor), ("round", np.round)]:
        _err = np.sum(
            np.abs(_fn(_energy[:, None] / _k_range[None, :]) - _cal[:, None]), axis=0
        )
        _res[_label] = _err
    _best_mode = min(_res, key=lambda k: _res[k].min())
    _K_best = _k_range[np.argmin(_res[_best_mode])]

    (
        ggplot(
            {
                "K": np.concatenate([_k_range, _k_range]).tolist(),
                "error": np.concatenate([_res["floor"], _res["round"]]).tolist(),
                "mode": ["floor"] * len(_k_range) + ["round"] * len(_k_range),
            },
            aes(x="K", y="error", color="mode"),
        )
        + geom_line()
        + geom_vline(xintercept=_K_best, linetype="dashed", alpha=0.5)
        + labs(
            x="K (joules per displayed calorie)",
            y="Total |error|",
            title=f"Calorie K sweep — best: K = {_K_best:.0f}, mode = {_best_mode}",
        )
        + theme_minimal()
    )
    return


@app.cell
def _(
    aes,
    frames_4fps,
    geom_histogram,
    geom_vline,
    ggplot,
    labs,
    np,
    theme_minimal,
):
    # Energy per calorie tick from 4fps data
    _clean = [f for f in frames_4fps if f["frame"] >= 62]
    _bad = set()
    for _i in range(1, len(_clean) - 1):
        if (
            _clean[_i]["cadence"] == 0
            and _clean[_i - 1]["cadence"] > 0
            and _clean[_i + 1]["cadence"] > 0
        ):
            _bad.add(_clean[_i]["frame"])
    _clean = [f for f in _clean if f["frame"] not in _bad]

    _max_cal = _clean[0]["calories"]
    _ticks = []
    for _i in range(1, len(_clean)):
        if _clean[_i]["calories"] > _max_cal:
            _ticks.append((_i, _max_cal, _clean[_i]["calories"]))
            _max_cal = _clean[_i]["calories"]

    _energies = []
    for _j in range(1, len(_ticks)):
        _s, _e = _ticks[_j - 1][0], _ticks[_j][0]
        _step = _ticks[_j][2] - _ticks[_j - 1][2]
        _energies.append(sum(_clean[k]["watts"] * 0.25 for k in range(_s, _e)) / _step)

    _arr = np.array(_energies)
    _med = float(np.median(_arr))
    (
        ggplot({"energy": _arr.tolist()}, aes(x="energy"))
        + geom_histogram(bins=30, fill="#4CAF50", alpha=0.7, color="black")
        + geom_vline(xintercept=_med, color="red", linetype="dashed")
        + geom_vline(xintercept=4184 / 3.5, color="blue", linetype="dotted")
        + labs(
            x="Energy per calorie tick (J)",
            y="Count",
            title=f"Energy per calorie tick — median = {_med:.0f} J  (blue = 4184/3.5 = {4184 / 3.5:.0f} J)",
        )
        + theme_minimal()
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Display Timing

    The 4fps dataset (250 ms resolution) reveals the firmware's internal clock:

    - **Sensor fields** (cadence, watts, speed) are **100% coupled** — they
      always update together. No partial updates or split-field artifacts.
    - **No display latency**: all 66 cadence transitions (≥ 2 RPM) show watts
      and speed already updated in the same frame. Response time < 250 ms.
    - **No transient values**: during large jumps (Δ ≥ 5 RPM), the display
      steps directly from old to new with no damping or smoothing.
    - **~1 Hz internal clock**: dominant inter-update interval is 4 frames
      (1000 ms), confirming a 1 Hz firmware tick.
    """)
    return


@app.cell
def _(aes, frames_4fps, geom_bar, ggplot, labs, np, theme_minimal):
    _ALL = ["cadence", "watts", "speed", "distance", "calories", "time"]
    _clean = [f for f in frames_4fps if f["frame"] >= 62]
    _bad = set()
    for _i in range(1, len(_clean) - 1):
        _c = _clean[_i]
        if (
            _c["cadence"] == 0
            and _clean[_i - 1]["cadence"] > 0
            and _clean[_i + 1]["cadence"] > 0
        ):
            _bad.add(_c["frame"])
        if (
            _c["distance"] == 0
            and _clean[_i - 1]["distance"] > 0
            and _clean[_i + 1]["distance"] > 0
        ):
            _bad.add(_c["frame"])
    _clean = [f for f in _clean if f["frame"] not in _bad]

    _intervals = []
    _prev_i = None
    for _i in range(1, len(_clean)):
        if any(_clean[_i][f] != _clean[_i - 1][f] for f in _ALL):
            if _prev_i is not None:
                _intervals.append(_i - _prev_i)
            _prev_i = _i

    _arr = np.array(_intervals)
    _vals, _counts = np.unique(_arr, return_counts=True)
    (
        ggplot(
            {
                "interval": _vals.astype(float).tolist(),
                "count": _counts.astype(float).tolist(),
            },
            aes(x="interval", y="count"),
        )
        + geom_bar(stat="identity", fill="#2196F3", alpha=0.7, width=0.7)
        + labs(
            x="Inter-update interval (frames at 4fps; 1 frame = 250 ms)",
            y="Count",
            title="Display update interval — dominant mode at 4 frames (1000 ms)",
        )
        + theme_minimal()
    )
    return


@app.cell
def _(
    aes,
    element_blank,
    frames_4fps,
    geom_point,
    ggplot,
    labs,
    scale_fill_manual,
    theme,
    theme_minimal,
):
    _ALL = ["cadence", "watts", "speed", "distance", "calories", "time"]
    _clean = [f for f in frames_4fps if f["frame"] >= 62]
    _bad = set()
    for _i in range(1, len(_clean) - 1):
        if (
            _clean[_i]["cadence"] == 0
            and _clean[_i - 1]["cadence"] > 0
            and _clean[_i + 1]["cadence"] > 0
        ):
            _bad.add(_clean[_i]["frame"])
    _clean = [f for f in _clean if f["frame"] not in _bad]

    _events = []
    for _i in range(1, len(_clean)):
        _changed = {f for f in _ALL if _clean[_i][f] != _clean[_i - 1][f]}
        if _changed:
            _events.append((_i, _changed))

    # Build heatmap-like scatter (subsampled for readability)
    _x, _y, _s = [], [], []
    _step = max(1, len(_events) // 300)
    for _j in range(0, len(_events), _step):
        for _f in _ALL:
            _x.append(float(_j))
            _y.append(_f)
            _s.append("changed" if _f in _events[_j][1] else "unchanged")

    (
        ggplot(
            {"event": _x, "field": _y, "status": _s},
            aes(x="event", y="field", fill="status"),
        )
        + geom_point(shape=15, size=1.5)
        + scale_fill_manual(values={"changed": "#2196F3", "unchanged": "#EEEEEE"})
        + labs(
            x="Change event index",
            y="Field",
            title="Field update simultaneity — sensor fields always change together",
        )
        + theme_minimal()
        + theme(legend_position="none", panel_grid=element_blank())
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Firmware Architecture

    The entire computation each tick:

    ```
    every ~1000.6 ms:
        c = read_cadence_sensor()

        speed_tenths = c * 272 / 73            # integer division
        d = c / 10
        watts = (B[d] + N[d] * (c % 10)) / 10  # integer division

        dist_acc += speed_tenths
        cal_acc  += watts

        display speed_tenths/10, watts, dist_acc/360/100, cal_acc/K
    ```

    - **Integer-only arithmetic** — no floating point anywhere
    - **No filtering or smoothing** — pure function of current cadence + two accumulators
    - **No state beyond `dist_acc` and `cal_acc`** — everything else recomputed each tick
    - **14 numbers define the power curve** ($B[2..8]$, $N[2..8]$); with continuity, only 8 are independent
    - **~1 Hz clock, +0.6 ms/tick drift** — accumulates ~0.24 s over a 400 s workout
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Appendix: Exact Timestamp Model

    The source video is 29.97 fps ($30000/1001$ exactly), and `ffmpeg fps=4`
    selects the nearest source frame for each output. This produces
    **non-uniform spacing** — gaps of 7 or 8 source frames (~233.6 or ~266.9 ms):

    $$
    M = \operatorname{round}\!\left(\frac{N \cdot 30000}{4 \cdot 1001}\right),
    \qquad t_N = M \cdot \frac{1001}{30000}
    $$

    827 gaps of 7 frames + 803 gaps of 8 frames → weighted mean of exactly
    250.0 ms, confirming 4 fps on average despite ±16.7 ms per-frame jitter.
    """)
    return


@app.cell
def _(
    SRC_FPS_DEN,
    SRC_FPS_NUM,
    aes,
    geom_bar,
    ggplot,
    labs,
    np,
    theme_minimal,
):
    _src = [round(n * SRC_FPS_NUM / (4 * SRC_FPS_DEN)) for n in range(1631)]
    _gaps = np.diff(_src)
    _vals, _counts = np.unique(_gaps, return_counts=True)
    (
        ggplot(
            {
                "gap": _vals.astype(float).tolist(),
                "count": _counts.astype(float).tolist(),
            },
            aes(x="gap", y="count"),
        )
        + geom_bar(stat="identity", fill="#9C27B0", alpha=0.7, width=0.6)
        + labs(
            x="Gap (source frames at 29.97 fps)",
            y="Count",
            title=f"Inter-frame gaps: {_counts[0]}×{_vals[0]:.0f} + {_counts[1]}×{_vals[1]:.0f} source frames",
        )
        + theme_minimal()
    )
    return


@app.cell
def _(
    aes,
    exact_timestamp,
    geom_histogram,
    geom_vline,
    ggplot,
    labs,
    np,
    theme_minimal,
):
    _dt_ms = np.diff([exact_timestamp(n + 1) for n in range(1631)]) * 1000
    (
        ggplot({"dt_ms": _dt_ms.tolist()}, aes(x="dt_ms"))
        + geom_histogram(bins=20, fill="#FF5722", alpha=0.7, color="black")
        + geom_vline(xintercept=250.0, linetype="dashed", color="blue", alpha=0.7)
        + labs(
            x="Inter-frame interval (ms)",
            y="Count",
            title=f"Exact intervals — mean = {float(np.mean(_dt_ms)):.1f} ms, bimodal from 29.97 fps quantization",
        )
        + theme_minimal()
    )
    return


@app.cell
def _(json, pl):
    """Data loading and firmware constants."""
    frames_1fps = [json.loads(l) for l in open("frames_1fps.jsonl")]
    frames_4fps = [json.loads(l) for l in open("frames_4fps.jsonl")]
    gt = pl.read_csv("top-row-states.csv")
    ts_1fps = pl.DataFrame(frames_1fps)
    ts_4fps = pl.DataFrame(frames_4fps)
    gt_nz = gt.filter(pl.col("cadence") > 0)

    WATTS_B = {2: 155, 3: 415, 4: 825, 5: 1505, 6: 2425, 7: 3835, 8: 5495}
    WATTS_N = {2: 26, 3: 41, 4: 68, 5: 92, 6: 141, 7: 166, 8: 211}

    def speed_formula(c):
        return (c * 272 // 73) / 10

    def watts_formula(c):
        d = c // 10
        return (WATTS_B.get(d, 0) + WATTS_N.get(d, 0) * (c % 10)) // 10

    c_gt = gt_nz["cadence"].to_numpy().astype(float)
    w_gt = gt_nz["watts"].to_numpy().astype(float)
    s_gt = gt_nz["speed"].to_numpy().astype(float)

    SRC_FPS_NUM = 30000
    SRC_FPS_DEN = 1001

    def exact_timestamp(frame_1indexed, fps=4):
        n = frame_1indexed - 1
        src_frame = round(n * SRC_FPS_NUM / (fps * SRC_FPS_DEN))
        return src_frame * SRC_FPS_DEN / SRC_FPS_NUM

    return (
        SRC_FPS_DEN,
        SRC_FPS_NUM,
        WATTS_B,
        WATTS_N,
        c_gt,
        exact_timestamp,
        frames_4fps,
        gt_nz,
        s_gt,
        speed_formula,
        ts_1fps,
        ts_4fps,
        w_gt,
        watts_formula,
    )


@app.cell
def _():
    import json

    import numpy as np
    import polars as pl
    from lets_plot import (
        LetsPlot,
        aes,
        element_blank,
        geom_abline,
        geom_bar,
        geom_histogram,
        geom_hline,
        geom_line,
        geom_point,
        geom_vline,
        ggplot,
        labs,
        scale_fill_manual,
        theme,
        theme_minimal,
    )

    LetsPlot.setup_html()
    return (
        aes,
        element_blank,
        geom_abline,
        geom_bar,
        geom_histogram,
        geom_hline,
        geom_line,
        geom_point,
        geom_vline,
        ggplot,
        json,
        labs,
        np,
        pl,
        scale_fill_manual,
        theme,
        theme_minimal,
    )


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
