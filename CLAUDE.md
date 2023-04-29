# CLAUDE.md

## Project overview

Exercise Machina reverse-engineers the Rogue Echo Bike's firmware equations from OCR'd video frames of the LCD display. The project has recovered the exact integer arithmetic used by the microcontroller for speed, watts, distance, and calories.

## Key firmware constants

```
WATTS_B = {2: 155, 3: 415, 4: 825, 5: 1505, 6: 2425, 7: 3835, 8: 5495}
WATTS_N = {2: 26, 3: 41, 4: 68, 5: 92, 6: 141, 7: 166, 8: 211}
TICK_PERIOD = 1.0006 seconds
CAL_K = 1195 firmware units per displayed calorie
DIST_DIVISOR = 360 speed_tenths per 0.01 mile

speed_tenths(c) = floor(c * 272 / 73)
watts(c) = floor((B[c//10] + N[c//10] * (c%10)) / 10)   for c >= 20
```

## Build & run

```bash
just env              # Install Python deps with uv
just analyze          # Run 1fps analysis
just analyze-4fps     # Run 4fps temporal analysis
open simulator/index.html    # Open firmware simulator
open converter/index.html    # Open conversion table
```

## Webapps

Two standalone HTML/JS webapps (no build step, no frameworks):
- `simulator/index.html` — glass-box firmware simulator showing every computation step
- `converter/index.html` — calorie/distance/time conversion table at constant effort

All share the same dark theme CSS variables and DSEG7 LCD font.

## Code standards

- Python: polars for dataframes, numpy for numerics, ruff for formatting
- HTML/JS: Single-file webapps, no build tools, no frameworks
- Integer arithmetic throughout to match firmware behavior
- Continuity constraint: B[d+1] = B[d] + 10*N[d] across all decades
