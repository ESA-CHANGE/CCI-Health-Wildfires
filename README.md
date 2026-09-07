# Wildfire Impacts on Air Quality and Human Health

This repository contains code developed for the wildfire-related health use case of the ESA CHANGE project.

## Overview

The current version of the repository includes the machine-learning workflow used to generate global future burned area projections under SSP scenarios.

The model uses environmental and climate predictors to estimate future burned area and forms part of the workflow used to assess future wildfire impacts on air quality and human health.

## Code

### `scripts/Ensemble_ssp.py`

Machine-learning workflow used to generate future global burned area projections under SSP scenarios.

The script is associated with the training data archived on Zenodo.

## Data

The training data and archived version of the script are available from Zenodo:

**Zenodo record:** 16927100

The Zenodo archive contains:

- `Ensemble_ssp.py`
- `training data.zip`

The training dataset is not duplicated in this GitHub repository because of its size.

## Requirements

The workflow requires Python 3 and commonly used scientific Python packages, including:

- NumPy
- pandas
- scikit-learn
- XGBoost
- Matplotlib

Additional package requirements may depend on the local computing environment.

## Usage

Download the training data from Zenodo and place the required input files in the directory expected by `Ensemble_ssp.py`.

Run the workflow with:

```bash
python scripts/Ensemble_ssp.py
