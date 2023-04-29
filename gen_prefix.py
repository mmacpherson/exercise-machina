"""Generate pre-pedaling 4fps frames (1-61) from visual inspection.

These frames have zero cadence/watts/speed/distance/calories.
Only the time field changes. Time pattern verified from frame images.
"""

import json

# Time map: (start_frame, end_frame_inclusive, time_str)
# Verified by reading individual frame images from data/frames_4fps/
TIME_MAP = [
    (1, 1, "01:30"),  # 1 frame (video starts mid-second)
    (2, 5, "01:31"),  # 4 frames
    (6, 9, "01:32"),  # 4 frames (verified f8, f9)
    (10, 13, "01:33"),  # 4 frames (verified f13)
    (14, 17, "01:34"),  # 4 frames (verified f15)
    (18, 21, "01:35"),  # 4 frames (verified f20)
    (22, 25, "01:36"),  # 4 frames (verified f25)
    (26, 38, "01:37"),  # 13 frames — countdown pauses here (verified f30, f37, f38)
    (39, 41, "00:00"),  # 3 frames — workout start (verified f39, f40; f42=00:01)
    (42, 45, "00:01"),  # 4 frames (verified f42, f43, f45)
    (46, 49, "00:02"),  # 4 frames (verified f47)
    (50, 53, "00:03"),  # 4 frames (verified f51)
    (54, 57, "00:04"),  # 4 frames (verified f55)
    (58, 61, "00:05"),  # 4 frames (verified f59, f60, f61)
]

frames = []
for start, end, time_str in TIME_MAP:
    for f in range(start, end + 1):
        frames.append(
            {
                "frame": f,
                "speed": 0.0,
                "watts": 0,
                "cadence": 0,
                "distance": 0.0,
                "time": time_str,
                "calories": 0,
            }
        )

assert len(frames) == 61, f"Expected 61 frames, got {len(frames)}"
assert frames[0]["frame"] == 1
assert frames[-1]["frame"] == 61

with open("prefix_4fps.jsonl", "w") as f:
    for fr in frames:
        f.write(json.dumps(fr) + "\n")

print(f"Wrote prefix_4fps.jsonl ({len(frames)} frames)")
print(f"  Timer range: {frames[0]['time']} → {frames[-1]['time']}")
