"""Assemble the full 4fps dataset from prefix + OCR batches + existing data.

Combines:
  1. prefix_4fps.jsonl          (frames 1-61, generated from visual inspection)
  2. batch_prefix_01..07.jsonl  (frames 62-408, OCR'd by subagents)
  3. frames_4fps_raw.jsonl      (frames 1-1223, renumbered → 409-1631)

Output: frames_4fps_raw_full.jsonl (frames 1-1631)
"""

import json
from pathlib import Path

SONNET_OCR = Path("data/sonnet-ocr")


def load_jsonl(path):
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def save_jsonl(frames, path):
    with open(path, "w") as f:
        for fr in frames:
            f.write(json.dumps(fr) + "\n")


# ── 1. Pre-pedaling prefix (frames 1-61) ────────────────────────────────
prefix = load_jsonl("prefix_4fps.jsonl")
print(f"Prefix: {len(prefix)} frames (1-{prefix[-1]['frame']})")

# ── 2. OCR'd active frames (frames 62-408) ──────────────────────────────
ocr_frames = []
for i in range(1, 8):
    batch_path = SONNET_OCR / f"batch_prefix_{i:02d}.jsonl"
    if not batch_path.exists():
        print(f"  WARNING: {batch_path} not found — skipping")
        continue
    batch = load_jsonl(batch_path)
    print(
        f"  Batch {i:02d}: {len(batch)} frames ({batch[0]['frame']}-{batch[-1]['frame']})"
    )
    ocr_frames.extend(batch)

print(f"OCR total: {len(ocr_frames)} frames")

# ── 3. Existing 4fps data (renumber 1-1223 → 409-1631) ──────────────────
existing = load_jsonl("frames_4fps_raw.jsonl")
print(
    f"Existing: {len(existing)} frames (renumbering 1-{existing[-1]['frame']} → 409-1631)"
)
for f in existing:
    f["frame"] += 408

# ── Combine ──────────────────────────────────────────────────────────────
all_frames = prefix + ocr_frames + existing

# Validate frame numbering
expected = list(range(1, 1632))
actual = [f["frame"] for f in all_frames]

if actual == expected:
    print(f"\nFrame numbering: OK (1-1631, {len(all_frames)} frames)")
else:
    # Find gaps and duplicates
    actual_set = set(actual)
    expected_set = set(expected)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing:
        print(f"\nMissing frames: {missing[:20]}{'...' if len(missing) > 20 else ''}")
    if extra:
        print(f"Extra frames: {extra[:20]}{'...' if len(extra) > 20 else ''}")
    # Check for duplicates
    from collections import Counter

    dups = {k: v for k, v in Counter(actual).items() if v > 1}
    if dups:
        print(f"Duplicate frames: {dict(list(dups.items())[:10])}")
    print(f"Total: {len(all_frames)} frames (expected 1631)")

# Save
save_jsonl(all_frames, "frames_4fps_raw_full.jsonl")
print(f"\nWrote: frames_4fps_raw_full.jsonl ({len(all_frames)} frames)")
