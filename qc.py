"""QC/QA pipeline for Echo Bike OCR datasets.

Reads raw OCR JSONL files, applies automated corrections and manual patches,
writes corrected files for downstream analysis.

Raw inputs:   frames_1fps_raw.jsonl, frames_4fps_raw.jsonl
Corrected:    frames_1fps.jsonl,     frames_4fps.jsonl

Corrections applied:
  1. Top-row removal:  all-zero frames flanked by non-zero (display in wrong mode)
  2. Formula fixes:    derive watts/speed from cadence where cadence looks correct
  3. Monotonic fields: enforce non-decreasing time, distance, calories via forward-fill
  4. Manual patches:   specific frame fixes from visual inspection / dead reckoning
"""

import json
from copy import deepcopy

# ── Firmware formulas (from analyze.py Section 1) ─────────────────────────

WATTS_B = {2: 155, 3: 415, 4: 825, 5: 1505, 6: 2425, 7: 3835, 8: 5495}
WATTS_N = {2: 26, 3: 41, 4: 68, 5: 92, 6: 141, 7: 166, 8: 211}


def speed_formula(c):
    return (c * 272 // 73) / 10


def watts_formula(c):
    d = c // 10
    return (WATTS_B.get(d, 0) + WATTS_N.get(d, 0) * (c % 10)) // 10


def parse_time(t):
    parts = t.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def format_time(secs):
    return f"{secs // 60:02d}:{secs % 60:02d}"


# ── Correction engine ─────────────────────────────────────────────────────


def apply_corrections(frames, label, manual_patches=None):
    """Apply all corrections to a list of frame dicts. Returns corrected copy."""
    out = deepcopy(frames)
    log = []

    # ── 1. Top-row removal ────────────────────────────────────────────────
    # All-zero frames flanked by non-zero: interpolate from neighbors.
    for i in range(1, len(out) - 1):
        cur, prev, nxt = out[i], out[i - 1], out[i + 1]
        if (
            cur["cadence"] == 0
            and cur["distance"] == 0.0
            and prev["cadence"] > 0
            and nxt["cadence"] > 0
        ):
            # Interpolate: take previous frame's values (hold last known state)
            for field in ["cadence", "watts", "speed", "distance", "calories"]:
                if cur[field] != prev[field]:
                    log.append(
                        (cur["frame"], field, cur[field], prev[field], "top-row")
                    )
                    cur[field] = prev[field]
            # Time: take next frame's time minus 1, or prev plus 1
            t_prev = parse_time(prev["time"])
            cur["time"] = format_time(t_prev)
            log.append((cur["frame"], "time", "zeroed", cur["time"], "top-row"))

    # ── 2. Manual patches ─────────────────────────────────────────────────
    if manual_patches:
        frame_lookup = {f["frame"]: f for f in out}
        for frame_num, field, value, reason in manual_patches:
            if frame_num in frame_lookup:
                old = frame_lookup[frame_num][field]
                if old != value:
                    frame_lookup[frame_num][field] = value
                    log.append((frame_num, field, old, value, f"manual: {reason}"))

    # ── 3. Formula consistency ────────────────────────────────────────────
    # Where cadence is in known range and watts/speed disagree with formula,
    # correct watts and speed. Only fix if cadence itself looks plausible
    # (non-zero, in range 20-89).
    for f in out:
        c = f["cadence"]
        if 20 <= c <= 89:
            exp_w = watts_formula(c)
            exp_s = speed_formula(c)
            if f["watts"] != exp_w:
                log.append((f["frame"], "watts", f["watts"], exp_w, "formula"))
                f["watts"] = exp_w
            if abs(f["speed"] - exp_s) > 0.05:
                log.append((f["frame"], "speed", f["speed"], exp_s, "formula"))
                f["speed"] = exp_s

    # ── 4. Monotonic enforcement ──────────────────────────────────────────
    # Time (parsed to seconds), distance, and calories must be non-decreasing.
    # Forward-fill: replace violations with the running maximum.

    # Time — find workout start (minimum time value, usually 00:00) and enforce
    # monotonicity only from there forward. Pre-workout countdown legitimately
    # decreases. If no clear minimum found, start from frame index 0.
    parsed = [parse_time(f["time"]) for f in out]
    workout_start_idx = parsed.index(min(parsed))
    max_t = parsed[workout_start_idx]
    for i in range(workout_start_idx + 1, len(out)):
        t = parsed[i]
        if t < max_t:
            corrected = format_time(max_t)
            log.append(
                (out[i]["frame"], "time", out[i]["time"], corrected, "monotonic")
            )
            out[i]["time"] = corrected
        else:
            max_t = t

    # Distance
    max_d = out[0]["distance"]
    for f in out[1:]:
        if f["distance"] < max_d:
            log.append((f["frame"], "distance", f["distance"], max_d, "monotonic"))
            f["distance"] = max_d
        else:
            max_d = f["distance"]

    # Calories
    max_c = out[0]["calories"]
    for f in out[1:]:
        if f["calories"] < max_c:
            log.append((f["frame"], "calories", f["calories"], max_c, "monotonic"))
            f["calories"] = max_c
        else:
            max_c = f["calories"]

    # ── Report ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"QC: {label}")
    print(f"{'=' * 60}")
    print(f"  Frames: {len(out)}")
    print(f"  Corrections: {len(log)}")

    by_reason = {}
    for _, _, _, _, reason in log:
        by_reason[reason] = by_reason.get(reason, 0) + 1
    for reason, count in sorted(by_reason.items()):
        print(f"    {reason}: {count}")

    by_field = {}
    for _, field, _, _, _ in log:
        by_field[field] = by_field.get(field, 0) + 1
    if by_field:
        print(f"  By field: {dict(sorted(by_field.items()))}")

    # Show individual corrections (first 30)
    if log:
        n_show = min(30, len(log))
        print(f"\n  First {n_show} corrections:")
        for frame, field, old, new, reason in log[:n_show]:
            print(f"    f={frame:>4} {field:>9}: {old!s:>8} → {new!s:<8}  ({reason})")
        if len(log) > n_show:
            print(f"    ... and {len(log) - n_show} more")

    # Post-QC validation (time monotonicity only checked from workout start)
    print("\n  Post-QC validation:")
    issues = 0
    for i in range(workout_start_idx + 1, len(out)):
        t_cur = parse_time(out[i]["time"])
        t_prev = parse_time(out[i - 1]["time"])
        if t_cur < t_prev:
            print(f"    FAIL: time not monotonic at frame {out[i]['frame']}")
            issues += 1
    for i in range(1, len(out)):
        if out[i]["distance"] < out[i - 1]["distance"]:
            print(f"    FAIL: distance not monotonic at frame {out[i]['frame']}")
            issues += 1
        if out[i]["calories"] < out[i - 1]["calories"]:
            print(f"    FAIL: calories not monotonic at frame {out[i]['frame']}")
            issues += 1
    if issues == 0:
        print("    All monotonicity checks passed")

    # Formula consistency check
    n_watts_ok, n_speed_ok, n_checked = 0, 0, 0
    for f in out:
        c = f["cadence"]
        if 20 <= c <= 89:
            n_checked += 1
            if f["watts"] == watts_formula(c):
                n_watts_ok += 1
            if abs(f["speed"] - speed_formula(c)) < 0.05:
                n_speed_ok += 1
    print(
        f"    Formula match: watts {n_watts_ok}/{n_checked}, "
        f"speed {n_speed_ok}/{n_checked}"
    )

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Manual patches — specific fixes from visual inspection and dead reckoning
# Format: (frame_number, field, corrected_value, reason)
# ═══════════════════════════════════════════════════════════════════════════

PATCHES_1FPS = [
    # Frame 25: all-zero mid-workout (top-row handles this, but time needs help)
    # Frame 50: same
    # Frame 249: distance briefly dips 1.48 → 1.44 → 1.48.
    # Frame 248 shows 1.48, frame 250 shows 1.48. OCR misread "8" as "4".
    (249, "distance", 1.48, "OCR: 8→4 in hundredths"),
]

PATCHES_4FPS = [
    # NOTE: frame numbers are in the full 1-1631 numbering.
    # Frames 409-1631 were the original 4fps OCR (old numbering 1-1223, +408 offset).
    # Frames 1-408 are the newly OCR'd prefix.
    # Frames 609-611 (old 201-203): distance reads 0.16 instead of 0.76.
    # OCR misread tenths digit "7" → "1". Context: 0.76 at f=608, 0.77 at f=618.
    (609, "distance", 0.76, "OCR: tenths 7→1"),
    (610, "distance", 0.76, "OCR: tenths 7→1"),
    (611, "distance", 0.77, "OCR: tenths 7→1"),
    # Frames 977, 989, 991, 1001, 1006, 1013 (old 569, 581, 583, 593, 598, 605):
    # distance reads 1.9x instead of 1.4x.
    # Systematic OCR confusion between "4" and "9" on 7-segment LCD in this region.
    # True distance is monotonically ~1.40-1.48; the "1.9x" values are misreads.
    (977, "distance", 1.40, "OCR: tenths 4→9"),
    (989, "distance", 1.43, "OCR: tenths 4→9"),
    (991, "distance", 1.43, "OCR: tenths 4→9"),
    (1001, "distance", 1.45, "OCR: tenths 4→9"),
    (1006, "distance", 1.46, "OCR: tenths 4→9"),
    (1013, "distance", 1.48, "OCR: tenths 4→9"),
]


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def load_jsonl(path):
    frames = []
    with open(path) as f:
        for line in f:
            frames.append(json.loads(line))
    return frames


def save_jsonl(frames, path):
    with open(path, "w") as f:
        for fr in frames:
            f.write(json.dumps(fr) + "\n")


if __name__ == "__main__":
    # ── 1fps dataset ──────────────────────────────────────────────────────
    raw_1fps = load_jsonl("frames_1fps_raw.jsonl")
    clean_1fps = apply_corrections(raw_1fps, "1fps", PATCHES_1FPS)
    save_jsonl(clean_1fps, "frames_1fps.jsonl")
    print(f"\n  Wrote: frames_1fps.jsonl ({len(clean_1fps)} frames)")

    # ── 4fps dataset (full 1-1631) ──────────────────────────────────────
    raw_4fps = load_jsonl("frames_4fps_raw.jsonl")
    clean_4fps = apply_corrections(raw_4fps, "4fps", PATCHES_4FPS)
    save_jsonl(clean_4fps, "frames_4fps.jsonl")
    print(f"\n  Wrote: frames_4fps.jsonl ({len(clean_4fps)} frames)")
