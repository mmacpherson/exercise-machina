# Contributing

Thanks for your interest in Exercise Machina! This project reverse-engineers the Rogue Echo Bike's firmware equations.

## Getting started

```bash
# Install dependencies
just env

# Run the analysis pipeline
just analyze
just analyze-4fps

# Open a webapp locally
open simulator/index.html
open converter/index.html
open bioenergetics/index.html
```

## Project structure

- **Webapps** (`simulator/`, `converter/`, `bioenergetics/`): Standalone HTML/JS, no build step
- **Analysis** (`analyze.py`, `analyze_4fps.py`, `qc.py`): Python scripts using polars/numpy
- **Notebooks** (`findings.py`, `simulator.py`): Marimo notebooks for interactive exploration
- **Data** (`*.jsonl`, `*.csv`): OCR datasets and ground truth

## Reporting issues

If you find a discrepancy between the firmware equations and your Echo Bike's display, please open an issue with:
- Your bike's firmware version (if known)
- The cadence/speed/watts values you observed
- The values the simulator predicts

## Code style

- Python: ruff for linting and formatting
- HTML/JS: No build tools, keep webapps as single standalone files
- Use integer arithmetic to match firmware behavior where possible
