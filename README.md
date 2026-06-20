# Optimizing Urban Heat Mitigation via AI/ML 🌍🌡️

**Target Region:** Delhi-NCR (National Capital Region, India)
**Resolution:** 30m pixel grid across a massive 1,500 sq km bounding box.

## Project Overview
This repository contains the end-to-end Machine Learning pipeline developed during the 4-day "Optimizing Urban Heat Mitigation" hackathon. The project transitions from raw geospatial extraction to advanced Physics-Informed Neural Networks (PINN), concluding with evolutionary multi-objective optimization (NSGA-II) to prescribe targeted, cost-bound Local Climate Zone (LCZ) cooling interventions.

## The 4-Pillar Architecture

1. **Geospatial Data Architecture:** Direct extraction of 30m resolution Landsat 8 (LST) and Sentinel-2 (NDVI, Albedo) rasters via Google Earth Engine.
2. **Feature Engineering & ML Baseline:** Empirical derivation of Relative Heat Island Intensity (RHII) using a high-NDVI/low-Albedo cropland baseline. LightGBM serves as the baseline regressor, interpreted via SHAP values.
3. **Urban Metabolism & PINN:** Mathematical proxy derivation of Anthropogenic Heat Emissions (Building & Transportation heat). A custom PyTorch Physics-Informed Neural Network bounds predictions within a simplified Stefan-Boltzmann Surface Energy Balance.
4. **Evolutionary Optimization:** Pymoo's NSGA-II Genetic Algorithm hunts the Pareto front to optimize Cool Roof and Urban Greening scenarios for the top 100 extreme UHI hotspots, constrained by a strict municipal intervention budget.

## Execution Instructions

To replicate the pipeline, execute the scripts sequentially from the root directory:

### 1. Data Extraction & Feature Engineering
```bash
# Day 1: Download LST, NDVI, and Albedo .tif files directly from GEE
python src/data/gee_extraction.py

# Day 2: Flatten raster grids to tabular features and calculate basic physics transformations
python src/data/build_features.py

# Day 3: Generate Urban Metabolism (AHE) proxies mathematically to avoid OSM API timeouts
python src/data/build_ahe_proxy.py
```

### 2. Model Training & Evaluation
```bash
# Train the LightGBM Baseline and generate SHAP Global/Local interpretations
python src/models/train_lightgbm.py

# Train the Physics-Informed Neural Network (PINN) and compare against baseline
python src/models/train_pinn.py
```

### 3. Scenario Optimization
```bash
# Run the NSGA-II Genetic Algorithm to optimize mitigation strategies for extreme hotspots
python src/models/optimize_scenarios.py
```

## Key Dependencies
* `earthengine-api`: Geospatial raw data extraction.
* `rasterio`: TIF manipulation and raster-to-grid flattening.
* `pandas` & `numpy`: Tabular feature engineering and normalization.
* `lightgbm` & `shap`: Tree-based regression and interpretability.
* `torch` (PyTorch): Deep learning and custom physics-bounded loss functions.
* `pymoo`: NSGA-II evolutionary optimization algorithm.
