import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Echo Bike Firmware Simulator

    A "glass box" into the Rogue Echo Bike's firmware. Pick a cadence and see
    every intermediate value, integer truncation, and accumulator state — or
    build a workout schedule and watch the firmware tick.

    _Equations reverse-engineered from OCR'd video frames — see `findings.py`._
    """)
    return


@app.cell
def _(mo):
    mode = mo.ui.radio(options=["Instant", "Timeline"], value="Instant", label="Mode")
    mode
    return (mode,)


# ── Instant Mode ─────────────────────────────────────────────────────────


@app.cell
def _(mo, mode):
    mo.stop(mode.value != "Instant")
    preset = mo.ui.dropdown(
        options={
            "Custom (use slider)": 0,
            "Easy (35 RPM)": 35,
            "Moderate (50 RPM)": 50,
            "Hard (65 RPM)": 65,
            "Sprint (80 RPM)": 80,
        },
        value="Custom (use slider)",
        label="Preset",
    )
    cadence_slider = mo.ui.slider(
        start=0, stop=89, step=1, value=55, label="Cadence (RPM)"
    )
    mo.hstack([preset, cadence_slider], justify="start", gap=2)
    return (cadence_slider, preset)


@app.cell
def _(cadence_slider, mode, mo, preset):
    mo.stop(mode.value != "Instant")
    cadence = preset.value if preset.value > 0 else cadence_slider.value
    mo.md(f"**Active cadence: {cadence} RPM**")
    return (cadence,)


@app.cell(hide_code=True)
def _(cadence, mode, mo, speed_compute, watts_compute):
    mo.stop(mode.value != "Instant")
    _s = speed_compute(cadence)
    _w = watts_compute(cadence)

    _speed_md = f"""### Speed: {cadence} RPM → {_s["speed_mph"]:.1f} mph

| Step | Operation | Result |
|------|-----------|-------:|
| 1 | `c` | {cadence} |
| 2 | `c × 272` | {_s["c_times_272"]} |
| 3 | `÷ 73` (exact) | {_s["exact"]:.4f} |
| 4 | `⌊floor⌋` → speed_tenths | **{_s["speed_tenths"]}** |
| 5 | `÷ 10` → display | **{_s["speed_mph"]:.1f} mph** |

Truncation loss: **{_s["trunc_loss"]:.4f}** tenths"""

    if cadence >= 20:
        _watts_md = f"""### Watts: {cadence} RPM → {_w["watts"]}W

| Step | Operation | Result |
|------|-----------|-------:|
| 1 | `d = c ÷ 10` | {_w["decade"]} |
| 2 | `r = c mod 10` | {_w["remainder"]} |
| 3 | `B[{_w["decade"]}]` | {_w["B"]} |
| 4 | `N[{_w["decade"]}]` | {_w["N"]} |
| 5 | `N × r = {_w["N"]} × {_w["remainder"]}` | {_w["N_times_r"]} |
| 6 | `B + N×r = {_w["B"]} + {_w["N_times_r"]}` | {_w["numerator"]} |
| 7 | `÷ 10` (exact) | {_w["exact"]:.1f} |
| 8 | `⌊floor⌋` → watts | **{_w["watts"]}W** |

Truncation loss: **{_w["trunc_loss"]:.1f}W**"""
    else:
        _watts_md = (
            f"### Watts: {cadence} RPM → 0W\n\n"
            "Cadence below 20 RPM — firmware outputs 0W."
        )

    mo.vstack([mo.md(_speed_md), mo.md(_watts_md)])
    return


@app.cell(hide_code=True)
def _(WATTS_B, WATTS_N, cadence, mode, mo):
    mo.stop(mode.value != "Instant")
    _active = cadence // 10
    _rows = []
    for _d in sorted(WATTS_B):
        _lo = _d * 10
        _w_lo = WATTS_B[_d] // 10
        _w_hi = (WATTS_B[_d] + WATTS_N[_d] * 9) // 10
        _mark = " **◀**" if _d == _active else ""
        _rows.append(
            f"| {_lo}–{_lo + 9} | {WATTS_B[_d]} | {WATTS_N[_d]} "
            f"| {WATTS_N[_d] / 10:.1f} | {_w_lo}–{_w_hi} |{_mark}"
        )
    mo.md(
        "### Piecewise lookup table\n\n"
        "| Cadence | B | N | W/RPM | Watts | |\n"
        "|---------|----:|----:|------:|------:|---|\n" + "\n".join(_rows)
    )
    return


@app.cell
def _(mo, mode):
    mo.stop(mode.value != "Instant")
    hold_duration = mo.ui.slider(
        start=10, stop=300, step=10, value=60, label="Hold duration (seconds)"
    )
    hold_duration
    return (hold_duration,)


@app.cell(hide_code=True)
def _(
    CAL_K,
    DIST_DIVISOR,
    TICK_PERIOD,
    aes,
    cadence,
    geom_line,
    ggplot,
    hold_duration,
    labs,
    mode,
    mo,
    np,
    speed_compute,
    theme_minimal,
    watts_compute,
):
    mo.stop(mode.value != "Instant")
    _s = speed_compute(cadence)
    _w = watts_compute(cadence)
    _n = int(hold_duration.value / TICK_PERIOD)
    _ticks = np.arange(1, _n + 1)
    _t = (_ticks * TICK_PERIOD).tolist()

    # Distance accumulator
    _d_raw = _ticks * _s["speed_tenths"]
    _d_exact = (_d_raw / DIST_DIVISOR / 100).tolist()
    _d_display = ((_d_raw // DIST_DIVISOR) / 100).tolist()

    # Calorie accumulator
    _c_raw = _ticks * _w["watts"]
    _c_exact = (_c_raw / CAL_K).tolist()
    _c_display = (_c_raw // CAL_K).astype(float).tolist()

    _p1 = (
        ggplot()
        + geom_line(
            aes(x="t", y="v"),
            data={"t": _t, "v": _d_exact},
            color="#BBBBBB",
            linetype="dashed",
        )
        + geom_line(
            aes(x="t", y="v"),
            data={"t": _t, "v": _d_display},
            color="#2196F3",
            size=1,
        )
        + labs(
            x="Time (s)",
            y="Distance (mi)",
            title=f"Distance at {cadence} RPM — display (blue) vs continuous (gray)",
        )
        + theme_minimal()
    )

    _p2 = (
        ggplot()
        + geom_line(
            aes(x="t", y="v"),
            data={"t": _t, "v": _c_exact},
            color="#BBBBBB",
            linetype="dashed",
        )
        + geom_line(
            aes(x="t", y="v"),
            data={"t": _t, "v": _c_display},
            color="#FF5722",
            size=1,
        )
        + labs(
            x="Time (s)",
            y="Calories",
            title=f"Calories at {cadence} RPM — display (orange) vs continuous (gray)",
        )
        + theme_minimal()
    )

    mo.vstack([_p1, _p2])
    return


@app.cell(hide_code=True)
def _(
    WATTS_B,
    aes,
    geom_bar,
    geom_vline,
    ggplot,
    labs,
    mode,
    mo,
    np,
    speed_compute,
    theme_minimal,
    watts_compute,
):
    mo.stop(mode.value != "Instant")
    _cadences = np.arange(20, 90)
    _s_loss = np.array([speed_compute(int(c))["trunc_loss"] for c in _cadences])
    _w_loss = np.array([watts_compute(int(c))["trunc_loss"] for c in _cadences])
    _bounds = [float(d * 10) for d in sorted(WATTS_B)][1:]

    _p1 = (
        ggplot(
            {"c": _cadences.astype(float).tolist(), "loss": _s_loss.tolist()},
            aes(x="c", y="loss"),
        )
        + geom_bar(stat="identity", fill="#2196F3", alpha=0.7)
        + geom_vline(xintercept=_bounds, color="gray", linetype="dotted", alpha=0.5)
        + labs(
            x="Cadence (RPM)",
            y="Loss (tenths of mph)",
            title="Speed truncation loss by cadence",
        )
        + theme_minimal()
    )

    _p2 = (
        ggplot(
            {"c": _cadences.astype(float).tolist(), "loss": _w_loss.tolist()},
            aes(x="c", y="loss"),
        )
        + geom_bar(stat="identity", fill="#FF5722", alpha=0.7)
        + geom_vline(xintercept=_bounds, color="gray", linetype="dotted", alpha=0.5)
        + labs(
            x="Cadence (RPM)",
            y="Loss (watts)",
            title="Watts truncation loss by cadence",
        )
        + theme_minimal()
    )

    mo.vstack([mo.md("### Truncation loss across all cadences"), _p1, _p2])
    return


# ── Timeline Mode ────────────────────────────────────────────────────────


@app.cell
def _(mo, mode):
    mo.stop(mode.value != "Timeline")
    workout_preset = mo.ui.dropdown(
        options={
            "Steady State (50 RPM × 5 min)": "steady",
            "Intervals (65/35 RPM × 30s, 8 rounds)": "intervals",
            "Ramp Test (30→80 RPM, 1 min/step)": "ramp",
            "Tabata (75/0 RPM, 20s/10s × 8)": "tabata",
            "Custom": "custom",
        },
        value="Steady State (50 RPM × 5 min)",
        label="Workout",
    )
    custom_text = mo.ui.text_area(
        value="50:60\n70:30\n50:60\n70:30",
        label="Custom segments (cadence:seconds, one per line)",
        full_width=True,
    )
    mo.vstack(
        [
            workout_preset,
            mo.accordion({"Custom segments": custom_text}),
        ]
    )
    return (custom_text, workout_preset)


@app.cell
def _(custom_text, mode, mo, simulate_workout, workout_preset):
    mo.stop(mode.value != "Timeline")
    _PRESETS = {
        "steady": [(50, 300)],
        "intervals": [(65, 30), (35, 30)] * 8,
        "ramp": [(c, 60) for c in range(30, 85, 10)],
        "tabata": [(75, 20), (0, 10)] * 8,
    }
    if workout_preset.value == "custom":
        _schedule = []
        for _line in custom_text.value.strip().split("\n"):
            _parts = _line.strip().split(":")
            if len(_parts) == 2:
                try:
                    _schedule.append((int(_parts[0]), int(_parts[1])))
                except ValueError:
                    pass
    else:
        _schedule = _PRESETS[workout_preset.value]

    tick_log = simulate_workout(_schedule)
    schedule = _schedule
    _total = sum(d for _, d in _schedule)
    mo.md(
        f"**Simulated {len(tick_log)} ticks** across {_total}s "
        f"({len(_schedule)} segments)"
    )
    return (schedule, tick_log)


@app.cell(hide_code=True)
def _(
    aes,
    geom_line,
    geom_vline,
    ggplot,
    labs,
    mode,
    mo,
    np,
    schedule,
    theme_minimal,
    tick_log,
):
    mo.stop(mode.value != "Timeline")
    mo.stop(len(tick_log) == 0, mo.md("_No ticks to display._"))

    _t = np.array([s["time"] for s in tick_log])
    _cad = np.array([s["cadence"] for s in tick_log], dtype=float)
    _watts = np.array([s["watts"] for s in tick_log], dtype=float)
    _dist_d = np.array([s["dist_display"] for s in tick_log])
    _dist_e = np.array([s["dist_exact"] for s in tick_log])
    _cal_d = np.array([s["cal_display"] for s in tick_log], dtype=float)
    _cal_e = np.array([s["cal_exact"] for s in tick_log])

    # Segment boundaries
    _bounds = []
    _acc = 0.0
    for _, _dur in schedule:
        _acc += _dur
        _bounds.append(_acc)
    _bounds = _bounds[:-1]

    _tl = _t.tolist()

    _p1 = (
        ggplot()
        + geom_line(
            aes(x="t", y="v"),
            data={"t": _tl, "v": _cad.tolist()},
            color="#2196F3",
        )
        + geom_vline(xintercept=_bounds, color="gray", linetype="dotted", alpha=0.4)
        + labs(x="Time (s)", y="RPM", title="Cadence")
        + theme_minimal()
    )

    _p2 = (
        ggplot()
        + geom_line(
            aes(x="t", y="v"),
            data={"t": _tl, "v": _watts.tolist()},
            color="#4CAF50",
        )
        + geom_vline(xintercept=_bounds, color="gray", linetype="dotted", alpha=0.4)
        + labs(x="Time (s)", y="Watts", title="Power output")
        + theme_minimal()
    )

    _p3 = (
        ggplot()
        + geom_line(
            aes(x="t", y="v"),
            data={"t": _tl, "v": _dist_e.tolist()},
            color="#BBBBBB",
            linetype="dashed",
        )
        + geom_line(
            aes(x="t", y="v"),
            data={"t": _tl, "v": _dist_d.tolist()},
            color="#2196F3",
            size=1,
        )
        + geom_vline(xintercept=_bounds, color="gray", linetype="dotted", alpha=0.4)
        + labs(
            x="Time (s)",
            y="Distance (mi)",
            title="Distance — display (blue) vs continuous (gray)",
        )
        + theme_minimal()
    )

    _p4 = (
        ggplot()
        + geom_line(
            aes(x="t", y="v"),
            data={"t": _tl, "v": _cal_e.tolist()},
            color="#BBBBBB",
            linetype="dashed",
        )
        + geom_line(
            aes(x="t", y="v"),
            data={"t": _tl, "v": _cal_d.tolist()},
            color="#FF5722",
            size=1,
        )
        + geom_vline(xintercept=_bounds, color="gray", linetype="dotted", alpha=0.4)
        + labs(
            x="Time (s)",
            y="Calories",
            title="Calories — display (orange) vs continuous (gray)",
        )
        + theme_minimal()
    )

    mo.vstack([_p1, _p2, _p3, _p4])
    return


@app.cell(hide_code=True)
def _(mode, mo, tick_log):
    mo.stop(mode.value != "Timeline")
    mo.stop(len(tick_log) == 0)
    _step = max(1, len(tick_log) // 200)
    _rows = tick_log[::_step]
    _table = [
        {
            "time": f"{r['time']:.1f}",
            "cadence": r["cadence"],
            "speed": f"{r['speed_mph']:.1f}",
            "watts": r["watts"],
            "dist_acc": r["dist_acc_raw"],
            "dist": f"{r['dist_display']:.2f}",
            "cal_acc": r["cal_acc_raw"],
            "cal": int(r["cal_display"]),
        }
        for r in _rows
    ]
    mo.accordion({f"Register state (every ~{_step} ticks)": mo.ui.table(_table)})
    return


@app.cell(hide_code=True)
def _(
    CAL_K,
    aes,
    geom_hline,
    geom_line,
    ggplot,
    labs,
    mode,
    mo,
    np,
    theme_minimal,
    tick_log,
):
    mo.stop(mode.value != "Timeline")
    mo.stop(len(tick_log) == 0)
    _t = np.array([s["time"] for s in tick_log])
    _energy = np.array([s["cal_acc_raw"] for s in tick_log], dtype=float)
    _max_cal = int(tick_log[-1]["cal_display"])
    _thresholds = [float(CAL_K * i) for i in range(1, _max_cal + 2)]

    (
        ggplot()
        + geom_line(
            aes(x="t", y="e"),
            data={"t": _t.tolist(), "e": _energy.tolist()},
            color="#FF5722",
            size=1,
        )
        + geom_hline(
            yintercept=_thresholds, color="#999999", linetype="dotted", alpha=0.5
        )
        + labs(
            x="Time (s)",
            y="Accumulated energy (firmware units)",
            title=f"Energy budget — each line = 1 displayed calorie ({CAL_K} units)",
        )
        + theme_minimal()
    )
    return


# ── Firmware Engine ──────────────────────────────────────────────────────


@app.cell
def _():
    WATTS_B = {2: 155, 3: 415, 4: 825, 5: 1505, 6: 2425, 7: 3835, 8: 5495}
    WATTS_N = {2: 26, 3: 41, 4: 68, 5: 92, 6: 141, 7: 166, 8: 211}
    TICK_PERIOD = 1.0006  # seconds per firmware tick
    CAL_K = 1195  # firmware units per displayed calorie
    DIST_DIVISOR = 360  # speed_tenths accumulator → 0.01 mile increments

    def speed_compute(c):
        """Full computation trace for speed at cadence c."""
        product = c * 272
        exact = product / 73
        tenths = product // 73
        return {
            "cadence": c,
            "c_times_272": product,
            "exact": exact,
            "speed_tenths": tenths,
            "trunc_loss": exact - tenths,
            "speed_mph": tenths / 10,
        }

    def watts_compute(c):
        """Full computation trace for watts at cadence c."""
        d = c // 10
        r = c % 10
        B = WATTS_B.get(d, 0)
        N = WATTS_N.get(d, 0)
        nr = N * r
        num = B + nr
        exact = num / 10
        w = num // 10
        return {
            "cadence": c,
            "decade": d,
            "remainder": r,
            "B": B,
            "N": N,
            "N_times_r": nr,
            "numerator": num,
            "exact": exact,
            "watts": w,
            "trunc_loss": exact - w,
        }

    def simulate_workout(schedule):
        """Run firmware tick-by-tick. schedule: list of (cadence, seconds)."""
        log = []
        dist_acc = 0
        cal_acc = 0
        t = 0.0
        for cad, dur in schedule:
            s = speed_compute(cad)
            w = watts_compute(cad)
            n_ticks = int(dur / TICK_PERIOD)
            for _ in range(n_ticks):
                t += TICK_PERIOD
                dist_acc += s["speed_tenths"]
                cal_acc += w["watts"]
                log.append(
                    {
                        "time": t,
                        "cadence": cad,
                        "speed_tenths": s["speed_tenths"],
                        "speed_mph": s["speed_mph"],
                        "watts": w["watts"],
                        "dist_acc_raw": dist_acc,
                        "dist_display": (dist_acc // DIST_DIVISOR) / 100,
                        "dist_exact": dist_acc / DIST_DIVISOR / 100,
                        "cal_acc_raw": cal_acc,
                        "cal_display": cal_acc // CAL_K,
                        "cal_exact": cal_acc / CAL_K,
                    }
                )
        return log

    return (
        CAL_K,
        DIST_DIVISOR,
        TICK_PERIOD,
        WATTS_B,
        WATTS_N,
        simulate_workout,
        speed_compute,
        watts_compute,
    )


@app.cell
def _():
    import numpy as np
    from lets_plot import (
        LetsPlot,
        aes,
        geom_bar,
        geom_hline,
        geom_line,
        geom_vline,
        ggplot,
        labs,
        theme_minimal,
    )

    LetsPlot.setup_html()
    return (
        aes,
        geom_bar,
        geom_hline,
        geom_line,
        geom_vline,
        ggplot,
        labs,
        np,
        theme_minimal,
    )


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
