"""Temporal analysis of Echo Bike display at 4fps sampling.

Sections 5-9 (continuing from analyze.py sections 1-4).
Exploits 4x temporal resolution to characterize LCD refresh
timing, field update simultaneity, and transition dynamics.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PLOTS = Path("plots")
PLOTS.mkdir(exist_ok=True)

# ── Formulas from analyze.py (Section 1) ──────────────────────────────────

WATTS_B = {2: 155, 3: 415, 4: 825, 5: 1505, 6: 2425, 7: 3835, 8: 5495}
WATTS_N = {2: 26, 3: 41, 4: 68, 5: 92, 6: 141, 7: 166, 8: 211}


def speed_formula(c):
    """Exact: speed = floor(c * 272 / 73) / 10."""
    return (c * 272 // 73) / 10


def watts_formula(c):
    """Firmware watts from piecewise integer formula."""
    d = c // 10
    return (WATTS_B.get(d, 0) + WATTS_N.get(d, 0) * (c % 10)) // 10


# ── Data loading ──────────────────────────────────────────────────────────
# frames_4fps.jsonl: full 4fps extraction from IMG_1392.MOV (407.85s, 29.97fps HEVC).
# ffmpeg fps=4 selects nearest source frame at 0.25s intervals (no interpolation).
# 1631 frames covering entire video. Timer: 01:30 countdown → 00:00 → 06:37.
# See frames_1fps.jsonl for the original 408-frame 1fps dataset (analyzed by analyze.py).

frames = []
with open("frames_4fps.jsonl") as f:
    for line in f:
        frames.append(json.loads(line))

FPS = 4
FRAME_MS = 1000 // FPS  # 250ms nominal between output frames
VIDEO_DURATION = 407.853333  # from ffprobe (IMG_1392.MOV)

# Source video timing: 29.97fps = 30000/1001 fps exactly.
# ffmpeg fps=4 selects nearest source frame for each output frame.
# Output frame N (1-indexed) → source frame M → exact timestamp.
SRC_FPS_NUM = 30000  # source framerate numerator
SRC_FPS_DEN = 1001  # source framerate denominator
SRC_FRAME_DUR = SRC_FPS_DEN / SRC_FPS_NUM  # ~33.367ms per source frame


def exact_timestamp(frame_1indexed):
    """Exact video timestamp for a 4fps output frame.

    ffmpeg fps=4 output frame N (0-indexed) has nominal time N/4.
    It selects the nearest source frame M = round(N * src_fps / out_fps).
    Exact timestamp = M * src_frame_duration.
    """
    n = frame_1indexed - 1  # 0-indexed
    src_frame = round(n * SRC_FPS_NUM / (FPS * SRC_FPS_DEN))
    return src_frame * SRC_FRAME_DUR


# Build exact timestamp array for all frames
frame_timestamps = {f["frame"]: exact_timestamp(f["frame"]) for f in frames}

# Key frame indices (in the 1-1631 numbering)
WORKOUT_START = 39  # timer resets to 00:00
PEDAL_START = 62  # first non-zero cadence

# Fields we track for change detection
SENSOR_FIELDS = ["cadence", "watts", "speed"]
ACCUM_FIELDS = ["distance", "calories"]
TIME_FIELD = ["time"]
ALL_FIELDS = SENSOR_FIELDS + ACCUM_FIELDS + TIME_FIELD

# Show actual frame spacing statistics
dts = [
    frame_timestamps[frames[i + 1]["frame"]] - frame_timestamps[frames[i]["frame"]]
    for i in range(len(frames) - 1)
]
print(f"Loaded {len(frames)} frames at {FPS}fps")
print(
    f"  Exact inter-frame intervals: min={min(dts) * 1000:.1f}ms, "
    f"max={max(dts) * 1000:.1f}ms, mean={np.mean(dts) * 1000:.1f}ms"
)
print(f"Timer: {frames[0]['time']} → {frames[-1]['time']}")
print(f"Workout start: frame {WORKOUT_START}, Pedal start: frame {PEDAL_START}")

# Active = pedaling frames only (skip countdown + idle)
active = [f for f in frames if f["frame"] >= PEDAL_START]
print(f"Active frames: {len(active)} (from frame {PEDAL_START})")


def parse_time(t):
    """Parse MM:SS or M:SS to integer seconds."""
    parts = t.split(":")
    return int(parts[0]) * 60 + int(parts[1])


# ── OCR error detection ───────────────────────────────────────────────────
# Detect frames with implausible field values: zero cadence flanked by
# non-zero, backward distance/calorie jumps, time field backward jumps.
# Apply monotonicity constraints where possible.

ocr_errors = set()
for i in range(1, len(active) - 1):
    cur = active[i]
    prev = active[i - 1]
    nxt = active[i + 1]
    # Zero cadence flanked by non-zero cadence is an OCR glitch
    if cur["cadence"] == 0 and prev["cadence"] > 0 and nxt["cadence"] > 0:
        ocr_errors.add(cur["frame"])
    # Sudden distance drop to 0 then recovery
    if cur["distance"] == 0 and prev["distance"] > 0 and nxt["distance"] > 0:
        ocr_errors.add(cur["frame"])

# Time monotonicity check: flag frames where parsed time decreases
for i in range(1, len(active)):
    t_prev = parse_time(active[i - 1]["time"])
    t_cur = parse_time(active[i]["time"])
    if t_cur < t_prev - 1:  # allow 1s jitter, flag big backwards jumps
        ocr_errors.add(active[i]["frame"])

clean = [f for f in active if f["frame"] not in ocr_errors]
print(f"OCR errors detected: {len(ocr_errors)} frames")
print(f"Clean frames: {len(clean)}\n")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: Display Update Rate
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 72)
print("SECTION 5: Display Update Rate")
print("=" * 72)

# Detect change events — frames where ANY tracked field differs from previous
change_frames = []  # (index_in_clean, frame_number, set_of_changed_fields)
for i in range(1, len(clean)):
    changed = set()
    for field in ALL_FIELDS:
        if clean[i][field] != clean[i - 1][field]:
            changed.add(field)
    if changed:
        change_frames.append((i, clean[i]["frame"], changed))

print(f"\nChange events: {len(change_frames)} out of {len(clean) - 1} frame pairs")
print(
    f"Static frames: {len(clean) - 1 - len(change_frames)} "
    f"({(len(clean) - 1 - len(change_frames)) / (len(clean) - 1) * 100:.1f}%)"
)

# Compute inter-change intervals
if len(change_frames) > 1:
    intervals_frames = []
    for j in range(1, len(change_frames)):
        gap = change_frames[j][0] - change_frames[j - 1][0]
        intervals_frames.append(gap)

    intervals_ms = [g * FRAME_MS for g in intervals_frames]
    intervals_arr = np.array(intervals_frames)
    intervals_ms_arr = np.array(intervals_ms)

    print(f"\nInter-update intervals (in frames at {FPS}fps):")
    print(
        f"  Mean:   {intervals_arr.mean():.2f} frames ({intervals_ms_arr.mean():.0f}ms)"
    )
    print(
        f"  Median: {np.median(intervals_arr):.1f} frames ({np.median(intervals_ms_arr):.0f}ms)"
    )
    print(f"  Min:    {intervals_arr.min()} frames ({intervals_ms_arr.min()}ms)")
    print(f"  Max:    {intervals_arr.max()} frames ({intervals_ms_arr.max()}ms)")
    print(
        f"  Std:    {intervals_arr.std():.2f} frames ({intervals_ms_arr.std():.0f}ms)"
    )

    # Distribution
    vals, counts = np.unique(intervals_arr, return_counts=True)
    print("\n  Interval distribution (frames → count):")
    for v, c in zip(vals, counts):
        bar = "#" * min(c, 60)
        print(f"    {v:>2} frames ({v * FRAME_MS:>4}ms): {c:>4}  {bar}")

    # Inferred refresh rate
    dominant_interval = vals[np.argmax(counts)]
    refresh_hz = FPS / dominant_interval
    print(
        f"\n  Dominant interval: {dominant_interval} frames = {dominant_interval * FRAME_MS}ms"
    )
    print(f"  Implied refresh rate: {refresh_hz:.2f} Hz")

# ── 5 Plot ────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Histogram of inter-update intervals
ax = axes[0]
ax.hist(
    intervals_ms_arr,
    bins=range(0, int(intervals_ms_arr.max()) + FRAME_MS + 1, FRAME_MS),
    edgecolor="black",
    alpha=0.7,
)
ax.set(
    xlabel="Inter-update interval (ms)",
    ylabel="Count",
    title="Distribution of display update intervals",
)
ax.axvline(1000, color="red", ls="--", lw=1.5, alpha=0.7, label="1000ms (1Hz)")
ax.legend()
ax.grid(True, alpha=0.3)

# Timeline of change events
ax = axes[1]
change_frame_nums = [cf[1] for cf in change_frames]
ax.eventplot([change_frame_nums], lineoffsets=0, linelengths=0.8, colors="blue")
ax.set(
    xlabel="Frame number",
    ylabel="",
    title="Timeline of display update events",
    yticks=[],
)
ax.grid(True, alpha=0.3, axis="x")

fig.suptitle("Section 5: Display Update Rate", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(PLOTS / "section5_update_rate.png", dpi=150)
print(f"\nSaved: {PLOTS / 'section5_update_rate.png'}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: Field Update Simultaneity
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("SECTION 6: Field Update Simultaneity")
print("=" * 72)

# Per-field update frequency
field_update_counts = {field: 0 for field in ALL_FIELDS}
for _, _, changed in change_frames:
    for field in changed:
        field_update_counts[field] += 1

print(f"\nPer-field update counts (out of {len(change_frames)} change events):")
for field in ALL_FIELDS:
    pct = field_update_counts[field] / len(change_frames) * 100
    print(f"  {field:>10}: {field_update_counts[field]:>4} ({pct:.1f}%)")

# Identify "split updates" — frames where sensor fields partially updated
sensor_change_events = []
for idx, frame_num, changed in change_frames:
    sensor_changed = changed & set(SENSOR_FIELDS)
    if sensor_changed and sensor_changed != set(SENSOR_FIELDS):
        sensor_change_events.append((frame_num, sensor_changed, changed))

print(
    f"\nSplit sensor updates (partial cad/watts/speed change): {len(sensor_change_events)}"
)
if sensor_change_events:
    n_show = min(20, len(sensor_change_events))
    for frame_num, sensor_changed, all_changed in sensor_change_events[:n_show]:
        print(
            f"  Frame {frame_num:>4}: sensor={sorted(sensor_changed)}, "
            f"all={sorted(all_changed)}"
        )
    if len(sensor_change_events) > n_show:
        print(f"  ... and {len(sensor_change_events) - n_show} more")

# Classify update patterns
pattern_counts = {}
for _, _, changed in change_frames:
    key = tuple(sorted(changed))
    pattern_counts[key] = pattern_counts.get(key, 0) + 1

print("\nUpdate patterns (which fields change together):")
sorted_patterns = sorted(pattern_counts.items(), key=lambda x: -x[1])
for pattern, count in sorted_patterns[:15]:
    pct = count / len(change_frames) * 100
    fields_str = ", ".join(pattern)
    print(f"  {count:>4} ({pct:>5.1f}%): {fields_str}")
if len(sorted_patterns) > 15:
    print(f"  ... and {len(sorted_patterns) - 15} more patterns")

# Check: do sensor fields (cad/watts/speed) always change together?
sensor_all_together = 0
sensor_any = 0
for _, _, changed in change_frames:
    sensor_changed = changed & set(SENSOR_FIELDS)
    if sensor_changed:
        sensor_any += 1
        if sensor_changed == set(SENSOR_FIELDS):
            sensor_all_together += 1

print("\nSensor field coupling:")
print(f"  Events with any sensor change: {sensor_any}")
print(f"  Events with ALL sensors changing: {sensor_all_together}")
print(
    f"  Coupling rate: {sensor_all_together / sensor_any * 100:.1f}%"
    if sensor_any
    else "  N/A"
)

# Check: do accum fields update independently of sensors?
accum_only = 0
sensor_only = 0
both = 0
for _, _, changed in change_frames:
    has_sensor = bool(changed & set(SENSOR_FIELDS))
    has_accum = bool(changed & set(ACCUM_FIELDS))
    if has_sensor and has_accum:
        both += 1
    elif has_sensor:
        sensor_only += 1
    elif has_accum:
        accum_only += 1

print("\nSensor vs accumulator independence:")
print(f"  Sensor only:  {sensor_only}")
print(f"  Accum only:   {accum_only}")
print(f"  Both:         {both}")

# ── 6 Plot ────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Heatmap of field changes per event (ordered by frame)
ax = axes[0]
field_order = ALL_FIELDS
n_events = len(change_frames)
heatmap = np.zeros((len(field_order), n_events), dtype=int)
for j, (_, _, changed) in enumerate(change_frames):
    for fi, field in enumerate(field_order):
        if field in changed:
            heatmap[fi, j] = 1

# Subsample for readability if too many events
if n_events > 400:
    step = max(1, n_events // 400)
    heatmap_show = heatmap[:, ::step]
    event_indices = np.arange(0, n_events, step)
else:
    heatmap_show = heatmap
    event_indices = np.arange(n_events)

ax.imshow(heatmap_show, aspect="auto", cmap="Blues", interpolation="nearest")
ax.set_yticks(range(len(field_order)))
ax.set_yticklabels(field_order)
ax.set(xlabel="Change event index", title="Field changes per update event")

# Per-field update frequency bar chart
ax = axes[1]
fields = list(field_update_counts.keys())
counts_bar = [field_update_counts[f] for f in fields]
colors = [
    "#2196F3" if f in SENSOR_FIELDS else "#FF9800" if f in ACCUM_FIELDS else "#4CAF50"
    for f in fields
]
ax.barh(fields, counts_bar, color=colors, edgecolor="black", alpha=0.7)
ax.set(xlabel="Number of updates", title="Per-field update frequency")
ax.grid(True, alpha=0.3, axis="x")

fig.suptitle("Section 6: Field Update Simultaneity", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(PLOTS / "section6_simultaneity.png", dpi=150)
print(f"\nSaved: {PLOTS / 'section6_simultaneity.png'}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: Distance Accumulation at Sub-Second Resolution
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("SECTION 7: Distance Accumulation at Sub-Second Resolution")
print("=" * 72)

# Identify distance increment frames
dist_increments = []  # (frame_index_in_clean, frame_number, prev_dist, new_dist)
for i in range(1, len(clean)):
    if clean[i]["distance"] > clean[i - 1]["distance"]:
        dist_increments.append(
            (
                i,
                clean[i]["frame"],
                clean[i - 1]["distance"],
                clean[i]["distance"],
            )
        )

print(f"\nDistance increments: {len(dist_increments)}")
print(f"Total distance: {clean[0]['distance']:.2f} → {clean[-1]['distance']:.2f}")

if dist_increments:
    # Inter-increment intervals
    inc_intervals = []
    inc_speeds = []
    for j in range(1, len(dist_increments)):
        gap_frames = dist_increments[j][0] - dist_increments[j - 1][0]
        gap_ms = gap_frames * FRAME_MS
        inc_intervals.append(gap_ms)
        # Speed at the increment point
        frame_idx = dist_increments[j][0]
        inc_speeds.append(clean[frame_idx]["speed"])

    inc_intervals_arr = np.array(inc_intervals)
    inc_speeds_arr = np.array(inc_speeds)

    print("\nInter-increment intervals:")
    print(f"  Mean:   {inc_intervals_arr.mean():.0f}ms")
    print(f"  Median: {np.median(inc_intervals_arr):.0f}ms")
    print(f"  Min:    {inc_intervals_arr.min()}ms")
    print(f"  Max:    {inc_intervals_arr.max()}ms")

    # Distribution
    vals, counts = np.unique(inc_intervals_arr, return_counts=True)
    print("\n  Interval distribution:")
    for v, c in sorted(zip(vals, counts), key=lambda x: -x[1])[:10]:
        print(f"    {v:>5}ms: {c:>3}")

    # Cross-reference with speed: at higher speed, increments should be more frequent
    # Group by speed range
    speed_bins = np.arange(0, 35, 5)
    print("\n  Mean increment interval by speed range:")
    for lo, hi in zip(speed_bins[:-1], speed_bins[1:]):
        mask = (inc_speeds_arr >= lo) & (inc_speeds_arr < hi)
        if mask.sum() > 0:
            mean_ms = inc_intervals_arr[mask].mean()
            print(f"    {lo:>2}–{hi:>2} mph: {mean_ms:>6.0f}ms (n={mask.sum()})")

    # ── dt refinement with 4fps data ──────────────────────────────────────
    # The firmware accumulates distance at ~1Hz internally. At 4fps, each frame
    # is ~0.25s apart, so dt ≈ 0.25. But the firmware may use a slightly different
    # clock, and we have 4x the data points to constrain it.
    print("\n  dt sweep with 4fps data:")
    dt_range = np.linspace(0.23, 0.27, 4001)

    speed_arr = np.array([f["speed"] for f in clean], dtype=float)
    actual_dist_arr = np.array([f["distance"] for f in clean], dtype=float)

    base_cumsum = np.cumsum(speed_arr / 3600)
    cum_dist_sweep = base_cumsum[:, None] * dt_range[None, :]
    disp_dist_sweep = np.floor(100 * cum_dist_sweep) / 100
    errors = np.sum(np.abs(disp_dist_sweep - actual_dist_arr[:, None]), axis=0)

    best_idx = np.argmin(errors)
    dt_4fps = dt_range[best_idx]
    min_err = errors[best_idx]
    print(f"    Best dt = {dt_4fps:.6f}s, total_err = {min_err:.4f}")
    print(f"    Implied effective fps = {1 / dt_4fps:.2f}")
    print("    (analyze.py 1fps dt ≈ 1.00; expected 4fps equivalent = 0.25000)")

    # Compare with formula-based speed
    speed_formula_arr = np.array(
        [speed_formula(f["cadence"]) for f in clean], dtype=float
    )
    base_cumsum_f = np.cumsum(speed_formula_arr / 3600)
    cum_dist_f = base_cumsum_f[:, None] * dt_range[None, :]
    disp_dist_f = np.floor(100 * cum_dist_f) / 100
    errors_f = np.sum(np.abs(disp_dist_f - actual_dist_arr[:, None]), axis=0)
    best_idx_f = np.argmin(errors_f)
    dt_4fps_f = dt_range[best_idx_f]
    print(
        f"    Formula speed: dt = {dt_4fps_f:.6f}s, total_err = {errors_f[best_idx_f]:.4f}"
    )

    # ── Firmware tick model ───────────────────────────────────────────────
    # The firmware accumulates distance at ~1Hz, not at 4fps. Subsample
    # to one frame per displayed-time second for the accumulation model.
    # Use monotonic time envelope to handle OCR jitter in the time field.
    print("\n  Firmware tick model (subsample to 1Hz):")

    # Parse and monotonize time
    parsed_times = [parse_time(f["time"]) for f in clean]
    mono_times = []
    cur_max = parsed_times[0]
    for t in parsed_times:
        cur_max = max(cur_max, t)
        mono_times.append(cur_max)

    n_time_corrected = sum(1 for r, m in zip(parsed_times, mono_times) if r != m)
    print(
        f"    Time field corrections (monotonicity): {n_time_corrected}/{len(parsed_times)}"
    )

    # Take first frame for each monotonic time value
    seen_times = set()
    firmware_frames = []
    for i, fr in enumerate(clean):
        t = mono_times[i]
        if t not in seen_times:
            seen_times.add(t)
            firmware_frames.append(fr)

    print(f"    Unique time values → {len(firmware_frames)} firmware ticks")

    fw_speed = np.array([f["speed"] for f in firmware_frames], dtype=float)
    fw_dist = np.array([f["distance"] for f in firmware_frames], dtype=float)

    # At 1Hz, dt should be ~1.0
    dt_1hz_range = np.linspace(0.97, 1.03, 6001)
    fw_cumsum = np.cumsum(fw_speed / 3600)
    fw_cum_dist = fw_cumsum[:, None] * dt_1hz_range[None, :]
    fw_disp_dist = np.floor(100 * fw_cum_dist) / 100
    fw_errors = np.sum(np.abs(fw_disp_dist - fw_dist[:, None]), axis=0)

    fw_best_idx = np.argmin(fw_errors)
    dt_firmware = dt_1hz_range[fw_best_idx]
    print(
        f"    Best dt (1Hz) = {dt_firmware:.6f}s, total_err = {fw_errors[fw_best_idx]:.4f}"
    )

    # Integer divisor sweep at 1Hz
    best_div_1hz, best_div_1hz_err = None, float("inf")
    for divisor in range(350, 370):
        acc = 0
        total_err = 0.0
        for fr in firmware_frames:
            speed_tenths = int(round(fr["speed"] * 10))
            acc += speed_tenths
            sim = (acc // divisor) / 100
            total_err += abs(sim - fr["distance"])
        if total_err < best_div_1hz_err:
            best_div_1hz_err = total_err
            best_div_1hz = divisor
    print(
        f"    Best integer divisor (1Hz) = {best_div_1hz}, "
        f"total_err = {best_div_1hz_err:.4f}"
    )
    print("    (360 = 3600/10 means exact 1Hz with speed_tenths)")

    # ── Firmware clock period ─────────────────────────────────────────────
    # Video metadata (ffprobe): IMG_1392.MOV, 407.853s, 29.97fps (30000/1001),
    # 12225 source frames.
    #
    # ffmpeg fps=4 outputs at nominal 0.25s intervals, but each output frame
    # is a copy of the NEAREST source frame. Source frames are spaced at
    # 1001/30000 ≈ 33.37ms. So each 4fps frame has ±16.7ms jitter relative
    # to its nominal timestamp. Over many frames this averages out, but
    # individual frame-to-frame intervals are not exactly 250ms.
    #
    # Frame N → nominal video time (N-1) * 0.25s (±16.7ms jitter).
    #
    # Strategy: find frames where the monotonic time increments by exactly +1s.
    # These are clean timer ticks. Measure real-time intervals between them.
    # Using nominal times over many ticks, the jitter averages out.

    print("\n  Firmware clock period (video-anchored):")
    print(f"    Video: {VIDEO_DURATION:.3f}s, frame interval = exactly {FRAME_MS}ms")

    # Exact real time of each clean frame (accounts for 29.97fps source)
    real_times = [exact_timestamp(f["frame"]) for f in clean]

    # Find clean +1s timer ticks using monotonic time
    tick_positions = []  # (clean_index, real_time_s, timer_value)
    for i in range(1, len(mono_times)):
        if mono_times[i] == mono_times[i - 1] + 1:
            tick_positions.append((i, real_times[i], mono_times[i]))

    print(f"    Clean +1s ticks: {len(tick_positions)}")

    if len(tick_positions) > 10:
        # Per-tick real-time intervals
        tick_intervals = np.array(
            [
                tick_positions[j][1] - tick_positions[j - 1][1]
                for j in range(1, len(tick_positions))
            ]
        )

        print("\n    Per-tick real-time intervals:")
        print(
            f"      Mean:   {tick_intervals.mean():.4f}s ({tick_intervals.mean() * 1000:.1f}ms)"
        )
        print(f"      Median: {np.median(tick_intervals):.4f}s")
        print(f"      Std:    {tick_intervals.std():.4f}s")

        # Distribution
        vals, counts = np.unique(np.round(tick_intervals, 3), return_counts=True)
        print("\n      Distribution:")
        for v, c in sorted(zip(vals, counts), key=lambda x: -x[1])[:8]:
            print(f"        {v:.3f}s: {c}")

        # Global measurement: first clean tick to last
        t_timer_elapsed = tick_positions[-1][2] - tick_positions[0][2]
        t_real_elapsed = tick_positions[-1][1] - tick_positions[0][1]
        clock_period = t_real_elapsed / t_timer_elapsed
        print("\n    Global clock measurement:")
        print(f"      Timer elapsed: {t_timer_elapsed} seconds")
        print(f"      Real elapsed:  {t_real_elapsed:.3f} seconds")
        print(f"      Clock period:  {clock_period:.6f}s ({clock_period * 1000:.3f}ms)")
        print(
            f"      Deviation:     {(clock_period - 1.0) * 1000:+.3f}ms/tick "
            f"({(clock_period - 1.0) * 100:+.4f}%)"
        )
        print(
            f"      Cumulative drift: {(clock_period - 1.0) * t_timer_elapsed:.3f}s "
            f"over {t_timer_elapsed}s"
        )

        # Phase drift: does the sub-second phase of ticks drift linearly?
        phases = np.array([tp[1] % 1.0 for tp in tick_positions])
        tick_idx = np.arange(len(phases))
        slope, intercept = np.polyfit(tick_idx, phases, 1)
        print("\n    Phase drift analysis:")
        print(f"      Phase slope: {slope:.6f} s/tick")
        print(f"      Implied clock period: {1.0 + slope:.6f}s")

        avg_period_ms = clock_period * 1000

# ── 7 Plot ────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distance increment timing vs speed
ax = axes[0, 0]
if len(inc_speeds_arr) > 0:
    ax.scatter(inc_speeds_arr, inc_intervals_arr, s=8, alpha=0.5)
    ax.set(
        xlabel="Speed (mph)",
        ylabel="Inter-increment interval (ms)",
        title="Distance increment timing vs speed",
    )
    ax.grid(True, alpha=0.3)

# Histogram of inter-increment intervals
ax = axes[0, 1]
ax.hist(inc_intervals_arr, bins=30, edgecolor="black", alpha=0.7)
ax.set(
    xlabel="Inter-increment interval (ms)",
    ylabel="Count",
    title="Distribution of distance increment intervals",
)
ax.grid(True, alpha=0.3)

# dt sweep curve
ax = axes[1, 0]
ax.plot(dt_range * 1000, errors, "b-", lw=1.5, label="Displayed speed")
ax.plot(dt_range * 1000, errors_f, "r--", lw=1, label="Formula speed")
ax.axvline(dt_4fps * 1000, color="k", ls="--", alpha=0.5)
ax.set(
    xlabel="dt (ms)",
    ylabel="Total |error|",
    title=f"dt sweep (best = {dt_4fps * 1000:.2f}ms)",
)
ax.legend()
ax.grid(True, alpha=0.3)

# Distance timeline with increment markers
ax = axes[1, 1]
frame_nums = np.array([f["frame"] for f in clean])
dist_arr = np.array([f["distance"] for f in clean])
ax.plot(frame_nums, dist_arr, "b-", lw=1, label="Distance")
inc_frame_nums = [dist_increments[j][1] for j in range(len(dist_increments))]
inc_dist_vals = [dist_increments[j][3] for j in range(len(dist_increments))]
ax.scatter(
    inc_frame_nums,
    inc_dist_vals,
    s=5,
    color="red",
    alpha=0.5,
    label=f"Increment events ({len(dist_increments)})",
    zorder=3,
)
ax.set(
    xlabel="Frame",
    ylabel="Distance (mi)",
    title="Distance with increment events marked",
)
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle(
    "Section 7: Distance Accumulation at Sub-Second Resolution",
    fontsize=14,
    fontweight="bold",
)
fig.tight_layout()
fig.savefig(PLOTS / "section7_distance_timing.png", dpi=150)
print(f"\nSaved: {PLOTS / 'section7_distance_timing.png'}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: Calorie Accumulation Timing
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("SECTION 8: Calorie Accumulation Timing")
print("=" * 72)

# Identify calorie increment frames using monotonic envelope.
# At 4fps, OCR jitter can cause brief calorie drops (e.g., 6→5→6),
# which would double-count the 5→6 transition. Track the running max
# and only count transitions to a new maximum.
running_max_cal = clean[0]["calories"]
cal_increments = []  # (frame_index_in_clean, frame_number, prev_cal, new_cal)
for i in range(1, len(clean)):
    cal_i = clean[i]["calories"]
    if cal_i > running_max_cal:
        cal_increments.append((i, clean[i]["frame"], running_max_cal, cal_i))
        running_max_cal = cal_i

n_cal_raw = sum(
    1 for i in range(1, len(clean)) if clean[i]["calories"] > clean[i - 1]["calories"]
)
print(f"\nCalorie increments: {len(cal_increments)} (monotonic, vs {n_cal_raw} raw)")
print(f"Total calories: {clean[0]['calories']} → {clean[-1]['calories']}")

if cal_increments:
    DT_FRAME = 1.0 / FPS

    # ── Energy between ticks ──────────────────────────────────────────────
    print(f"\nEnergy analysis between calorie ticks (dt = {DT_FRAME}s per frame):")

    energies_between_ticks = []
    intervals_between_ticks = []
    cal_steps = []  # how many calories each tick covers (usually 1)

    for j in range(1, len(cal_increments)):
        start_idx = cal_increments[j - 1][0]
        end_idx = cal_increments[j][0]
        interval_frames = end_idx - start_idx
        interval_ms = interval_frames * FRAME_MS
        step = cal_increments[j][3] - cal_increments[j - 1][3]

        energy_j = sum(clean[k]["watts"] * DT_FRAME for k in range(start_idx, end_idx))
        energies_between_ticks.append(energy_j / step)  # normalize per calorie
        intervals_between_ticks.append(interval_ms)
        cal_steps.append(step)

    energies_arr = np.array(energies_between_ticks)
    intervals_cal_arr = np.array(intervals_between_ticks)

    print("\n  Energy per calorie (should ≈ K ≈ 1195 J):")
    print(f"    Mean:   {energies_arr.mean():.2f} J")
    print(f"    Median: {np.median(energies_arr):.2f} J")
    print(f"    Std:    {energies_arr.std():.2f} J")
    print(f"    Min:    {energies_arr.min():.2f} J")
    print(f"    Max:    {energies_arr.max():.2f} J")
    print(f"    Multi-cal steps: {sum(1 for s in cal_steps if s > 1)}")

    K_est = np.median(energies_arr)
    print(f"\n  Estimated K = {K_est:.2f} J/cal")
    print(f"  4184 / K = {4184 / K_est:.4f}")
    print(f"  Compare: 4184/3.5 = {4184 / 3.5:.2f}")

    # ── K sweep at 4fps resolution ────────────────────────────────────────
    # Use firmware-tick subsampled data for clean accumulation
    print("\n  K sweep (firmware tick model):")

    # Accumulate energy using firmware frames (1Hz subsample)
    fw_watts = np.array([f["watts"] for f in firmware_frames], dtype=float)
    fw_cal = np.array([f["calories"] for f in firmware_frames], dtype=float)
    fw_energy = np.cumsum(fw_watts * dt_firmware)

    k_range = np.linspace(1100, 1300, 2001)
    best_k_result = {}
    for rnd_label, rnd_fn in [("floor", np.floor), ("round", np.round)]:
        sim_cal_sweep = rnd_fn(fw_energy[:, None] / k_range[None, :])
        cal_errors = np.sum(np.abs(sim_cal_sweep - fw_cal[:, None]), axis=0)
        best_k_idx = np.argmin(cal_errors)
        best_k_result[rnd_label] = {
            "K": k_range[best_k_idx],
            "err": cal_errors[best_k_idx],
            "errors": cal_errors,
        }
        print(
            f"    {rnd_label:>5}: K = {k_range[best_k_idx]:.2f}, "
            f"total_err = {cal_errors[best_k_idx]:.0f}"
        )

    # Fine sweep around best
    best_rnd = min(best_k_result, key=lambda k: best_k_result[k]["err"])
    K_coarse = best_k_result[best_rnd]["K"]
    rnd_fn_best = np.floor if best_rnd == "floor" else np.round

    k_fine = np.linspace(K_coarse - 20, K_coarse + 20, 40001)
    sim_cal_fine = rnd_fn_best(fw_energy[:, None] / k_fine[None, :])
    cal_errors_fine = np.sum(np.abs(sim_cal_fine - fw_cal[:, None]), axis=0)
    best_k_fine_idx = np.argmin(cal_errors_fine)
    min_err_fine = cal_errors_fine[best_k_fine_idx]
    K_opt = k_fine[best_k_fine_idx]

    optimal_mask = cal_errors_fine == min_err_fine
    k_optimal = k_fine[optimal_mask]
    print(
        f"    {best_rnd:>5} fine: K = {K_opt:.4f}, "
        f"err = {min_err_fine:.0f}, "
        f"range = [{k_optimal.min():.4f}, {k_optimal.max():.4f}]"
    )

    # ── Rounding mode test ────────────────────────────────────────────────
    print("\n  Rounding mode analysis (1Hz firmware model):")
    K_test = 4184 / 3.5  # ≈ 1195.4

    for rnd_label, rnd_fn in [("floor", np.floor), ("round", np.round)]:
        sim = rnd_fn(fw_energy / K_test).astype(int)
        err = int(np.sum(np.abs(sim - fw_cal.astype(int))))
        max_err = int(np.max(np.abs(sim - fw_cal.astype(int))))
        print(f"    K=4184/3.5, {rnd_label:>5}: total_err={err}, max_err={max_err}")

    # Test integer K values near the optimum
    print("\n    Integer K values near optimum:")
    for K_int in range(int(K_opt) - 3, int(K_opt) + 4):
        sim = rnd_fn_best(fw_energy / K_int)
        err = int(np.sum(np.abs(sim - fw_cal)))
        max_err = int(np.max(np.abs(sim - fw_cal)))
        print(f"      K={K_int}: total_err={err}, max_err={max_err}")

    # ── Calorie tick timing precision ─────────────────────────────────────
    # Recompute energy_cumul for all clean frames for tick analysis
    watts_arr_clean = np.array([f["watts"] for f in clean], dtype=float)
    actual_cal_arr = np.array([f["calories"] for f in clean], dtype=float)
    energy_cumul = np.cumsum(watts_arr_clean * DT_FRAME)

    print("\n  Calorie tick timing precision:")
    print("    First 15 calorie ticks:")
    print(
        f"    {'Tick':>4} {'Frame':>5} {'Cal':>4} {'Energy':>8} {'E/K':>7} {'Phase':>6}"
    )
    for j, (idx, fnum, prev_c, new_c) in enumerate(cal_increments[:15]):
        energy_at_tick = energy_cumul[idx]
        e_over_k = energy_at_tick / K_test
        phase = e_over_k - int(e_over_k)
        print(
            f"    {j + 1:>4} {fnum:>5} {new_c:>4} {energy_at_tick:>8.1f} "
            f"{e_over_k:>7.3f} {phase:>6.3f}"
        )

# ── 8 Plot ────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Energy per calorie tick
ax = axes[0, 0]
ax.plot(energies_arr, "b.-", lw=1, markersize=3)
ax.axhline(K_est, color="red", ls="--", lw=1.5, label=f"Median = {K_est:.1f} J")
ax.axhline(
    4184 / 3.5, color="green", ls=":", lw=1.5, label=f"4184/3.5 = {4184 / 3.5:.1f} J"
)
ax.set(
    xlabel="Calorie tick index",
    ylabel="Energy per calorie (J)",
    title="Energy between consecutive calorie ticks",
)
ax.legend()
ax.grid(True, alpha=0.3)

# Calorie increment interval distribution
ax = axes[0, 1]
ax.hist(intervals_cal_arr / 1000, bins=30, edgecolor="black", alpha=0.7)
ax.set(
    xlabel="Interval between calorie ticks (s)",
    ylabel="Count",
    title="Distribution of calorie tick intervals",
)
ax.grid(True, alpha=0.3)

# Calories: actual vs simulated (1Hz firmware model)
ax = axes[1, 0]
K_plot = K_opt
fw_frame_nums = np.array([f["frame"] for f in firmware_frames])
sim_cal_plot = rnd_fn_best(fw_energy / K_plot)
ax.plot(fw_frame_nums, fw_cal, "b-", lw=1.5, label="Actual")
ax.plot(fw_frame_nums, sim_cal_plot, "r--", lw=1, label=f"Sim (K={K_plot:.1f})")
tick_frames = [ci[1] for ci in cal_increments]
tick_cals = [ci[3] for ci in cal_increments]
ax.scatter(tick_frames, tick_cals, s=15, color="green", zorder=3, label="Tick events")
ax.set(
    xlabel="Frame", ylabel="Calories", title="Calories: actual vs simulated (1Hz model)"
)
ax.legend()
ax.grid(True, alpha=0.3)

# K sweep curve
ax = axes[1, 1]
for rnd_label, res in best_k_result.items():
    ax.plot(k_range, res["errors"], lw=1.5, label=f"{rnd_label}")
ax.axvline(K_opt, color="k", ls="--", alpha=0.5, label=f"K={K_opt:.1f}")
ax.axvline(4184 / 3.5, color="red", ls=":", alpha=0.5, label="4184/3.5")
ax.set(xlabel="K (J/cal)", ylabel="Total |error|", title="K sweep (1Hz firmware model)")
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle("Section 8: Calorie Accumulation Timing", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(PLOTS / "section8_calorie_timing.png", dpi=150)
print(f"\nSaved: {PLOTS / 'section8_calorie_timing.png'}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9: Transition Dynamics and Display Latency
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("SECTION 9: Transition Dynamics and Display Latency")
print("=" * 72)

# Identify cadence transitions in clean data (|Δcadence| >= 2 between change events)
# Work with change events that include sensor updates
sensor_changes = [
    (idx, fnum, changed)
    for idx, fnum, changed in change_frames
    if changed & set(SENSOR_FIELDS)
]

transitions = []
for j in range(1, len(sensor_changes)):
    idx_cur = sensor_changes[j][0]
    idx_prev = sensor_changes[j - 1][0]
    cad_cur = clean[idx_cur]["cadence"]
    cad_prev = clean[idx_prev]["cadence"]
    delta_cad = cad_cur - cad_prev

    if abs(delta_cad) >= 2:
        transitions.append(
            {
                "idx": idx_cur,
                "frame": clean[idx_cur]["frame"],
                "from_cad": cad_prev,
                "to_cad": cad_cur,
                "delta": delta_cad,
                "gap_frames": idx_cur - idx_prev,
            }
        )

print(f"\nCadence transitions (|Δ| >= 2): {len(transitions)}")

ramp_up = [t for t in transitions if t["delta"] > 0]
ramp_down = [t for t in transitions if t["delta"] < 0]
print(f"  Ramp up:   {len(ramp_up)}")
print(f"  Ramp down: {len(ramp_down)}")

# For each transition, check per-field update timing in the surrounding window
print("\nTransition analysis (per-field response):")
print(
    f"  {'Frame':>5} {'ΔCad':>5} {'Dir':>4} {'SameFrame':>10} {'WattsLag':>9} {'SpeedLag':>9}"
)

watts_lag_up = []
watts_lag_down = []
same_frame_count = 0

for t in transitions:
    idx = t["idx"]
    frame_num = t["frame"]
    from_c, to_c = t["from_cad"], t["to_cad"]

    # Expected watts/speed at new cadence
    expected_w_new = watts_formula(to_c) if to_c > 0 else 0
    expected_s_new = speed_formula(to_c) if to_c > 0 else 0.0

    # Check if watts and speed updated in the same frame as cadence
    actual_w = clean[idx]["watts"]
    actual_s = clean[idx]["speed"]

    watts_same = actual_w == expected_w_new
    speed_same = abs(actual_s - expected_s_new) < 0.05

    if watts_same and speed_same:
        same_frame_count += 1
        lag = 0
    else:
        # Look ahead for when watts catches up
        lag = None
        for k in range(idx + 1, min(idx + 20, len(clean))):
            if clean[k]["watts"] == expected_w_new:
                lag = k - idx
                break

    direction = "UP" if t["delta"] > 0 else "DOWN"
    lag_str = f"{lag}" if lag is not None else "N/A"

    if t["delta"] > 0 and lag is not None:
        watts_lag_up.append(lag)
    elif t["delta"] < 0 and lag is not None:
        watts_lag_down.append(lag)

print(
    f"\n  Same-frame updates: {same_frame_count}/{len(transitions)} "
    f"({same_frame_count / len(transitions) * 100:.1f}%)"
)

if watts_lag_up:
    print("\n  Watts lag (ramp UP):")
    print(
        f"    Mean: {np.mean(watts_lag_up):.2f} frames ({np.mean(watts_lag_up) * FRAME_MS:.0f}ms)"
    )
    vals, counts = np.unique(watts_lag_up, return_counts=True)
    for v, c in zip(vals, counts):
        print(f"    {v} frames ({v * FRAME_MS}ms): {c}")

if watts_lag_down:
    print("\n  Watts lag (ramp DOWN):")
    print(
        f"    Mean: {np.mean(watts_lag_down):.2f} frames ({np.mean(watts_lag_down) * FRAME_MS:.0f}ms)"
    )
    vals, counts = np.unique(watts_lag_down, return_counts=True)
    for v, c in zip(vals, counts):
        print(f"    {v} frames ({v * FRAME_MS}ms): {c}")

# Asymmetry test
if watts_lag_up and watts_lag_down:
    mean_up = np.mean(watts_lag_up)
    mean_down = np.mean(watts_lag_down)
    print(
        f"\n  Latency asymmetry: up={mean_up:.2f}f vs down={mean_down:.2f}f "
        f"(diff={abs(mean_up - mean_down):.2f}f = {abs(mean_up - mean_down) * FRAME_MS:.0f}ms)"
    )

# ── Transient detection ───────────────────────────────────────────────────
# Check for intermediate values during large jumps (|Δ| >= 5)
print("\nTransient detection (large jumps |Δcad| >= 5):")

large_jumps = [t for t in transitions if abs(t["delta"]) >= 5]
print(f"  Large jumps: {len(large_jumps)}")

transients_found = 0
for t in large_jumps:
    idx = t["idx"]
    from_c, to_c = t["from_cad"], t["to_cad"]
    from_w = watts_formula(from_c) if from_c > 0 else 0
    to_w = watts_formula(to_c) if to_c > 0 else 0

    # Look at the exact frame of transition and the next few
    window = []
    for k in range(max(0, idx - 2), min(len(clean), idx + 5)):
        w = clean[k]["watts"]
        c = clean[k]["cadence"]
        expected_w = watts_formula(c) if c > 0 else 0
        is_transient = w != from_w and w != to_w and w != expected_w
        window.append((clean[k]["frame"], c, w, expected_w, is_transient))
        if is_transient:
            transients_found += 1

print(f"  Transient intermediate values: {transients_found}")

# Show a few example transition windows
n_show_trans = min(8, len(large_jumps))
print(f"\n  First {n_show_trans} large transition windows:")
for t in large_jumps[:n_show_trans]:
    idx = t["idx"]
    print(
        f"    Frame {t['frame']}: cad {t['from_cad']}→{t['to_cad']} (Δ={t['delta']:+d})"
    )
    for k in range(max(0, idx - 2), min(len(clean), idx + 4)):
        c = clean[k]["cadence"]
        w = clean[k]["watts"]
        s = clean[k]["speed"]
        exp_w = watts_formula(c) if c > 0 else 0
        exp_s = speed_formula(c) if c > 0 else 0.0
        w_match = "ok" if w == exp_w else f"exp={exp_w}"
        s_match = "ok" if abs(s - exp_s) < 0.05 else f"exp={exp_s}"
        marker = " <<<" if clean[k]["frame"] == t["frame"] else ""
        print(
            f"      f={clean[k]['frame']:>4} c={c:>2} w={w:>3}({w_match:>6}) "
            f"s={s:>5.1f}({s_match:>8}){marker}"
        )

# ── 9 Plot ────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Transition timeline
ax = axes[0, 0]
trans_frames_arr = np.array([t["frame"] for t in transitions])
trans_deltas = np.array([t["delta"] for t in transitions])
colors = ["green" if d > 0 else "red" for d in trans_deltas]
ax.bar(trans_frames_arr, trans_deltas, width=2, color=colors, alpha=0.7)
ax.axhline(0, color="k", lw=0.5)
ax.set(xlabel="Frame", ylabel="Δ Cadence", title="Cadence transitions over time")
ax.grid(True, alpha=0.3)

# Lag distribution
ax = axes[0, 1]
all_lags = watts_lag_up + watts_lag_down
if all_lags:
    all_lags_ms = np.array(all_lags) * FRAME_MS
    ax.hist(
        all_lags_ms,
        bins=range(0, int(max(all_lags_ms)) + FRAME_MS + 1, FRAME_MS),
        edgecolor="black",
        alpha=0.7,
    )
    ax.set(
        xlabel="Watts response lag (ms)",
        ylabel="Count",
        title="Display lag after cadence transition",
    )
ax.grid(True, alpha=0.3)

# Zoomed transition example (pick the largest jump)
if large_jumps:
    biggest = max(large_jumps, key=lambda t: abs(t["delta"]))
    idx = biggest["idx"]
    window_start = max(0, idx - 8)
    window_end = min(len(clean), idx + 12)

    ax = axes[1, 0]
    w_frames = [clean[k]["frame"] for k in range(window_start, window_end)]
    w_cad = [clean[k]["cadence"] for k in range(window_start, window_end)]
    w_watts = [clean[k]["watts"] for k in range(window_start, window_end)]
    w_exp_watts = [watts_formula(c) if c > 0 else 0 for c in w_cad]

    ax.step(w_frames, w_cad, "b-", lw=2, where="post", label="Cadence")
    ax.axvline(biggest["frame"], color="red", ls="--", alpha=0.5)
    ax.set(
        xlabel="Frame",
        ylabel="Cadence (RPM)",
        title=f"Largest transition: cad {biggest['from_cad']}→{biggest['to_cad']}",
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.step(w_frames, w_watts, "r-", lw=2, where="post", label="Watts (actual)")
    ax2.step(w_frames, w_exp_watts, "r:", lw=1, where="post", label="Watts (expected)")
    ax2.set_ylabel("Watts")
    ax2.legend(loc="upper right")

# Lag asymmetry: up vs down
ax = axes[1, 1]
if watts_lag_up and watts_lag_down:
    up_ms = np.array(watts_lag_up) * FRAME_MS
    down_ms = np.array(watts_lag_down) * FRAME_MS
    positions = [1, 2]
    bp = ax.boxplot(
        [up_ms, down_ms], positions=positions, widths=0.6, patch_artist=True
    )
    bp["boxes"][0].set_facecolor("green")
    bp["boxes"][1].set_facecolor("red")
    ax.set_xticks(positions)
    ax.set_xticklabels(["Ramp Up", "Ramp Down"])
    ax.set(ylabel="Watts lag (ms)", title="Response latency: ramp up vs ramp down")
else:
    ax.text(
        0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center", va="center"
    )
ax.grid(True, alpha=0.3)

fig.suptitle(
    "Section 9: Transition Dynamics and Display Latency", fontsize=14, fontweight="bold"
)
fig.tight_layout()
fig.savefig(PLOTS / "section9_transitions.png", dpi=150)
print(f"\nSaved: {PLOTS / 'section9_transitions.png'}")


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("SUMMARY: Temporal Analysis at 4fps")
print("=" * 72)

print(f"\n  Dataset: {len(frames)} frames at {FPS}fps ({FRAME_MS}ms resolution)")
print(f"  OCR errors removed: {len(ocr_errors)}")

if len(change_frames) > 1:
    print("\n  Display refresh:")
    print(f"    Update events: {len(change_frames)}")
    print(
        f"    Dominant interval: {dominant_interval} frames = {dominant_interval * FRAME_MS}ms"
    )
    print(f"    Implied rate: {refresh_hz:.2f} Hz")

print("\n  Field coupling:")
print(
    f"    Sensor (cad/watts/speed) coupled: {sensor_all_together}/{sensor_any} "
    f"({sensor_all_together / sensor_any * 100:.1f}%)"
    if sensor_any
    else "    N/A"
)
print(f"    Update patterns: {len(pattern_counts)} distinct")

if dist_increments:
    print("\n  Distance timing:")
    print(f"    Increments detected: {len(dist_increments)}")
    print(f"    Firmware dt (1Hz model): {dt_firmware:.6f}s")
    print(f"    Best integer divisor (1Hz): {best_div_1hz}")
    print(f"    Firmware clock period: {avg_period_ms:.2f}ms")

if cal_increments:
    print("\n  Calorie timing:")
    print(f"    Ticks detected (monotonic): {len(cal_increments)}")
    print(f"    Best K: {K_opt:.2f} J/cal")
    print(f"    Median energy/tick: {K_est:.2f} J")

print("\n  Transition dynamics:")
print(f"    Transitions (|Δ| >= 2): {len(transitions)}")
print(f"    Same-frame response: {same_frame_count}/{len(transitions)}")
if watts_lag_up and watts_lag_down:
    print(f"    Mean lag (up): {np.mean(watts_lag_up) * FRAME_MS:.0f}ms")
    print(f"    Mean lag (down): {np.mean(watts_lag_down) * FRAME_MS:.0f}ms")
