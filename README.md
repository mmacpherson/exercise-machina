# Exercise Machina

[![CI](https://github.com/mmacpherson/exercise-machina/actions/workflows/ci.yml/badge.svg)](https://github.com/mmacpherson/exercise-machina/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Reverse-engineering the Rogue Echo Bike's firmware equations from OCR'd video of its LCD display.

## Live webapps

- **[Firmware Simulator](https://mmacpherson.github.io/exercise-machina/simulator/)** — glass-box view of every integer arithmetic step the firmware computes
- **[Conversion Table](https://mmacpherson.github.io/exercise-machina/converter/)** — calories &harr; distance &harr; time equivalence at constant effort

## The firmware equations

The Echo Bike's microcontroller uses integer arithmetic with two 7-entry lookup tables — 14 integers total.

### Speed

```
speed_tenths = floor(cadence * 272 / 73)
speed_mph    = speed_tenths / 10
```

### Watts

```
watts = floor((B[cadence // 10] + N[cadence // 10] * (cadence % 10)) / 10)
```

| Decade | B     | N   | Watts range |
|--------|-------|-----|-------------|
| 20–29  | 155   | 26  | 15–38       |
| 30–39  | 415   | 41  | 41–78       |
| 40–49  | 825   | 68  | 82–143      |
| 50–59  | 1505  | 92  | 150–233     |
| 60–69  | 2425  | 141 | 242–369     |
| 70–79  | 3835  | 166 | 383–532     |
| 80–89  | 5495  | 211 | 549–739     |

The continuity constraint `B[d+1] = B[d] + 10*N[d]` holds across all decades, meaning the piecewise-linear curve has no jumps.

### Accumulators (per tick, ~1.0006s)

```
dist_acc += speed_tenths    → display = floor(dist_acc / 360) / 100  miles
cal_acc  += watts           → display = floor(cal_acc / 1195)        calories
```

## How it was done

1. **Film the LCD** — Recorded the Echo Bike display at known cadence intervals
2. **OCR the frames** — Extracted speed/watts/cadence/distance/time/calories from each frame using Claude's vision
3. **Recover the formulas** — Found the exact integer arithmetic by fitting the OCR data, exploiting the quantization artifacts that reveal the firmware's truncation behavior
4. **Verify** — 58/58 exact matches against ground truth, plus temporal accumulator verification across hundreds of frames

## Quickstart

```bash
# Install dependencies
just env

# Run the analysis pipeline
just analyze
just analyze-4fps

# Open a webapp
open simulator/index.html
open converter/index.html
```

## Project structure

```
├── simulator/index.html      Firmware simulator webapp
├── converter/index.html      Conversion table webapp
├── watts_final.py            Definitive watts formula + verification
├── analyze.py                1fps OCR data analysis
├── analyze_4fps.py           4fps temporal analysis
├── qc.py                     OCR quality control pipeline
├── findings.py               Marimo findings notebook
├── simulator.py              Marimo simulator notebook
├── frames_1fps.jsonl         Cleaned 1fps OCR dataset
├── frames_4fps.jsonl         Cleaned 4fps OCR dataset
├── top-row-states.csv        Ground truth calibration data
├── justfile                  Build automation
└── pyproject.toml            Python project config
```

## License

[MIT](LICENSE)
