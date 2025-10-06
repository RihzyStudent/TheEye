# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

The Eye is an ML-based exoplanet detection system that analyzes light curve data from various space missions (Kepler, K2, TESS) to identify new exoplanets. The project uses Transit Least Squares (TLS) and other astronomical analysis tools.

## Development Environment

### Virtual Environment
This project uses a Python virtual environment (`.venv`). Always activate it before development:
```bash
source .venv/bin/activate  # On Linux/macOS
```

### Key Dependencies
- **Core Analysis**: lightkurve, transitleastsquares, astropy, astroquery
- **ML/Data Science**: numpy, pandas, scikit-learn, xgboost, lightgbm
- **Visualization**: matplotlib, seaborn, plotly
- **Web Framework**: fastapi, gradio
- **Exoplanet Modeling**: batman-package, wotan, celerite2, eleanor

## Common Development Tasks

### Running Light Curve Analysis
The main analysis code is in `LightKurve_source/LK_src.py`. To run light curve analysis:
```bash
cd LightKurve_source
python LK_src.py
```

### Starting the Web Interface
For the FastAPI server:
```bash
uvicorn app:app --reload  # Replace 'app' with actual module name when created
```

For Gradio interface:
```bash
python gradio_app.py  # Replace with actual filename when created
```

### Working with Jupyter Notebooks
```bash
jupyter lab
```

### Code Quality Tools
```bash
# Format code
black .

# Lint code
ruff check .

# Type checking
mypy .
```

### Running Tests
```bash
pytest
```

## Architecture and Code Structure

### Current Structure
- **LightKurve_source/**: Contains light curve analysis code and output images
  - `LK_src.py`: Main analysis script for downloading and processing exoplanet light curves
  - Various `.png` files: Generated visualizations from light curve analysis

### Key Concepts
1. **Light Curves**: Time series data showing brightness variations of stars
2. **Transit Detection**: Uses TLS (Transit Least Squares) to find periodic dips in brightness
3. **Data Sources**: NASA's Kepler, K2, and TESS missions via lightkurve API
4. **Exoplanet Catalog**: Integration with NASA Exoplanet Archive API for known planet data

### Analysis Workflow
1. Search and download light curve data from missions
2. Clean and flatten the data (remove outliers, normalize)
3. Apply Transit Least Squares algorithm to find periodic signals
4. Fold data at detected period to visualize transit
5. Calculate planet parameters (radius, orbital period, etc.)
6. Query NASA Exoplanet Archive for additional stellar/planetary data

### API Integration
The code integrates with NASA's Exoplanet Archive TAP service to fetch:
- Stellar parameters (mass, radius, temperature)
- Known exoplanet parameters
- Target object identifiers (TIC, KIC, EPIC)

## Important Technical Details

- The project handles different mission identifiers (TIC for TESS, KIC for Kepler, EPIC for K2)
- Light curve correction includes crowding/dilution corrections using CROWDSAP values
- Custom detrending functions implement spline-based systematic correction
- Period search windows are calculated based on observational constraints
- The TLS implementation uses box-shaped transit models with limb darkening

## Future Development Areas

Based on the challenge objectives:
- Implement ML model training pipeline for automatic exoplanet classification
- Create web interface for user interaction and data upload
- Add model accuracy statistics and hyperparameter tuning
- Enable real-time model updates with new user-provided data
- Implement visualization dashboards for research and educational use
