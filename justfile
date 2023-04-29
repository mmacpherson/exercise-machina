# Exercise Machina — Rogue Echo Bike firmware reverse-engineering

default:
    @just --list

# Install/sync dependencies
env:
    uv sync

# Run 1fps analysis (sections 1-4)
analyze:
    uv run python analyze.py

# Run 4fps temporal analysis (sections 5-9)
analyze-4fps:
    uv run python analyze_4fps.py

# Run QC pipeline on raw OCR data
qc:
    uv run python qc.py

# Run firmware simulator (webapp)
simulator:
    open simulator/index.html

# Open conversion table (webapp)
converter:
    open converter/index.html

# Run firmware simulator (marimo notebook)
simulator-nb:
    uv run marimo run simulator.py

# Open findings notebook in browser
findings:
    uv run marimo edit findings.py

# Export findings notebook to static HTML
export:
    uv run marimo export html findings.py -o findings.html

# Full pipeline: QC → analysis → export
all: qc analyze analyze-4fps export

# Install pre-commit hooks
install-hooks:
    uv run pre-commit install

# Update pre-commit hook versions
update-hooks:
    uv run pre-commit autoupdate

# Run all pre-commit hooks
lint:
    uv run pre-commit run --all-files

# Remove generated artifacts
clean:
    rm -f plots/*.png findings.html
