# California Wildfire Severity Prediction

A point-based geospatial machine learning project for predicting California wildfire severity using pre-fire environmental conditions, terrain, drought context, and vegetation/fuel features.

This project began as a county-level wildfire analysis, but was rebuilt into a more defensible point-based pipeline. Instead of assigning broad county-level values to each fire, the final workflow samples weather, terrain, drought, and fuel/vegetation features at each wildfire's ignition coordinates.

## Project Overview

The goal of this project is to predict wildfire severity from conditions available before or near the start of a fire.

The final dataset combines:

- CAL FIRE incident records
- gridMET daily weather and fire-danger variables
- USGS 3DEP terrain rasters
- U.S. Drought Monitor drought intensity
- LANDFIRE vegetation and fuel layers

The project uses a time-based validation setup:

```text
Train: 2016-2021
Test:  2022-2024
```

This split is intentionally stricter than a random split because it evaluates whether the model can generalize to future fire seasons.

## Motivation

Wildfire size is extremely difficult to predict from environmental data alone because final acreage depends on many factors that are not fully captured in public pre-fire datasets, including suppression response, ignition cause, fuel continuity, wind shifts, road access, and human development patterns.

Because of this, the project does not frame the task as exact acreage prediction. Instead, it reframes wildfire severity as an imbalanced classification problem using NWCG-derived fire-size tiers.

## Target Definition

The original continuous target, `AcresBurned`, was highly right-skewed:

```text
Median acres burned:        80
Mean acres burned:       4,333
95th percentile:         8,806
99th percentile:        83,844
Maximum:             1,032,648
```

Because raw acreage is unstable and heavily affected by outliers, the project uses severity tiers as the primary modeling target:

| Severity Tier |          Acreage Range |
| ------------- | ---------------------: |
| Small         |            < 100 acres |
| Medium        |   100 to < 1,000 acres |
| Large         | 1,000 to < 5,000 acres |
| Extreme       |         >= 5,000 acres |

Final target distribution:

| Class   | Count |
| ------- | ----: |
| Small   | 1,311 |
| Medium  |   720 |
| Large   |   200 |
| Extreme |   166 |

This class imbalance is why the project emphasizes balanced accuracy, macro F1, per-class recall, and Large/Extreme recall instead of raw accuracy.

## Data Sources

### CAL FIRE Incidents

CAL FIRE incident records were used as the base wildfire dataset. Each incident includes fields such as:

- Fire name
- Start date
- Latitude and longitude
- Acres burned
- Administrative metadata

The project uses each fire's latitude and longitude as the central point for feature sampling.

### gridMET Weather and Fire-Danger Data

gridMET daily climate and fire-danger variables were sampled at each fire's ignition coordinates. For each incident, pre-fire windows were computed over 7-day, 14-day, and 30-day periods.

Key variables include:

- Maximum temperature
- Minimum temperature
- Vapor pressure deficit
- Wind speed
- Minimum relative humidity
- Precipitation
- Energy Release Component
- Burning Index
- 100-hour fuel moisture
- 1000-hour fuel moisture

### USGS 3DEP Terrain Features

Terrain features were sampled from USGS 3DEP-derived rasters at each fire point.

Features include:

- Elevation
- Slope
- Aspect
- Northness
- Eastness

Aspect was transformed into northness and eastness to avoid treating circular direction as a linear variable.

### U.S. Drought Monitor Features

Weekly U.S. Drought Monitor GeoTIFFs were downloaded and sampled at each wildfire point. For each incident, the pipeline selected the latest drought map available before the fire start date to avoid temporal leakage.

Features include:

- USDM raw raster value
- USDM category
- USDM intensity
- D0+ abnormally dry indicator
- D1+ drought indicator
- D2+ severe drought indicator
- D3+ extreme drought indicator

### LANDFIRE Vegetation and Fuel Features

LANDFIRE layers were sampled at each fire point using the LANDFIRE Product Service.

Features include:

- FBFM40 fire behavior fuel model
- Broad FBFM40 fuel group
- Existing vegetation type
- Existing vegetation cover
- Existing vegetation height
- Canopy cover
- Fuel vegetation type
- Fuel vegetation cover

Raw high-cardinality vegetation codes are retained for EDA, while broader fuel groups are used for more interpretable modeling.

## Pipeline

The final pipeline follows a structured `raw/interim/processed` workflow:

```text
CAL FIRE incidents
        |
        v
Clean fire records and assign gridmet_id
        |
        v
Sample gridMET weather/fire-danger variables at ignition points
        |
        v
Aggregate 7-day, 14-day, and 30-day pre-fire windows
        |
        v
Sample USGS 3DEP terrain rasters
        |
        v
Sample weekly USDM drought rasters
        |
        v
Sample LANDFIRE fuel and vegetation rasters
        |
        v
Final enriched modeling table
        |
        v
EDA and modeling
```

Final modeling table:

```text
data/processed/calfire_with_gridmet_terrain_drought_veg.csv
```

## Repository Structure

```text
wildfire-severity-v2/
├── data/
│   ├── raw/
│   │   ├── gridmet_output/
│   │   ├── terrain/
│   │   ├── drought/
│   │   └── landfire/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 00_calfire_extraction.ipynb
│   ├── 01_gridmet_extraction_and_aggregation.ipynb
│   ├── 03_terrain_download_and_sampling.ipynb
│   ├── 06_drought_enrichment.ipynb
│   ├── 07_landfire_vegetation_fuels_enrichment.ipynb
│   ├── 08_final_eda.ipynb
│   └── 09_final_modeling.ipynb
├── outputs/
│   ├── final_eda/
│   └── final_modeling/
├── scripts/
├── requirements.txt
└── README.md
```

## Exploratory Data Analysis

The final EDA found several useful patterns:

- Acres burned is highly right-skewed, supporting severity-tier classification instead of raw acreage regression.
- Larger fires generally occur under drier and more fire-dangerous conditions.
- Higher severity tiers tend to show higher VPD, ERC, and Burning Index.
- Higher severity tiers tend to show lower relative humidity and lower fuel moisture.
- Large and Extreme fires tend to occur at higher median elevation and steeper slopes.
- Extreme fires have higher average USDM drought intensity than Small fires.
- LANDFIRE fuel groups differ across severity tiers, with larger fires showing higher representation in shrub and timber-related fuel groups.
- Several fire-danger variables are highly correlated, especially ERC, Burning Index, fuel moisture, temperature, and VPD.
- The 2016-2021 train and 2022-2024 test split shows distribution shift, especially because the test period contains a higher proportion of Small fires.

## Modeling

Two modeling tasks were evaluated.

### Task 1: Multiclass Severity Prediction

Classes:

```text
Small, Medium, Large, Extreme
```

Best multiclass model:

```text
Model: Random Forest
Features: gridMET + terrain + drought + LANDFIRE
Accuracy: 0.4965
Balanced accuracy: 0.3543
Macro F1: 0.3353
Large/Extreme recall: 0.4045
```

The majority baseline achieved higher raw accuracy because it predicted only the most common class, but it had 0% Large/Extreme recall and much lower macro F1. This confirms that raw accuracy is misleading for this imbalanced task.

### Task 2: Binary High-Severity Triage

Classes:

```text
Low:  Small or Medium
High: Large or Extreme
```

Best binary model:

```text
Model: Random Forest
Features: gridMET + terrain + drought
Accuracy: 0.8640
Balanced accuracy: 0.6210
Macro F1: 0.6240
High recall: 0.3146
High precision: 0.3333
ROC-AUC: 0.6942
Average precision: 0.2732
```

The binary task provides a more practical risk-triage framing than exact four-class prediction.

### Threshold Tuning

At the default 0.50 threshold, the binary model is conservative. Lowering the high-severity threshold improves recall:

| Threshold | High Recall | High Precision | Macro F1 |
| --------: | ----------: | -------------: | -------: |
|      0.35 |      0.5843 |         0.2088 |   0.5692 |
|      0.40 |      0.4944 |         0.2431 |   0.6002 |
|      0.45 |      0.3933 |         0.2536 |   0.6016 |
|      0.50 |      0.3146 |         0.3333 |   0.6240 |

A 0.40 threshold may be more useful for triage because it catches nearly half of high-severity fires while maintaining better precision than more aggressive thresholds.

## Interpretation

The final models show that environmental and geospatial features contain meaningful wildfire severity signal, but exact severity prediction remains difficult.

The strongest interpretation is not that the model is operationally ready. Instead, the project demonstrates a defensible geospatial ML pipeline that:

- Samples point-based environmental features instead of relying on county-level averages
- Avoids post-fire leakage
- Uses realistic time-based validation
- Handles target skew and class imbalance explicitly
- Evaluates both multiclass severity prediction and binary high-severity triage
- Provides an extensible foundation for future wildfire risk modeling

## Limitations

This project does not include every factor that determines final fire size. Important missing predictors include:

- Ignition cause
- Suppression response
- Initial attack timing
- Fire perimeter growth
- Fuel continuity beyond the ignition point
- Road access
- WUI and population exposure
- Daily wind direction and extreme wind events
- Lightning and human ignition context
- Fire weather warnings or red flag events

The model should not be used for operational wildfire decision-making. It is a data science project focused on environmental signal discovery and severity-risk modeling.

## Future Improvements

Potential next steps:

- Add ignition cause when available.
- Add WUI and population exposure features.
- Add distance to roads, cities, and fire stations.
- Add fuel continuity metrics from buffers around ignition points instead of only point samples.
- Use fire perimeter data to sample fuels across burned area.
- Add drought lag features across multiple prior weeks or months.
- Use year-held-out cross-validation.
- Tune Random Forest and XGBoost hyperparameters.
- Explore probability calibration for high-severity triage.
- Build an interactive dashboard or map-based model explainer.

## How to Reproduce

1. Clone the repository.

```bash
git clone https://github.com/mynameisdevinchau/wildfire-severity-v2.git
cd wildfire-severity-v2
```

2. Create and activate a virtual environment.

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Run notebooks in order.

```text
00_calfire_extraction.ipynb
01_gridmet_extraction_and_aggregation.ipynb
03_terrain_download_and_sampling.ipynb
06_drought_enrichment.ipynb
07_landfire_vegetation_fuels_enrichment.ipynb
08_final_eda.ipynb
09_final_modeling.ipynb
```

5. Review final outputs.

```text
outputs/final_eda/
outputs/final_modeling/
```

## Important Git Notes

Large generated files should generally not be committed, especially raw rasters and cache files.

Recommended `.gitignore` additions:

```gitignore
.venv/
__pycache__/
.ipynb_checkpoints/

cache/
*.sqlite

data/raw/gridmet_output/
data/raw/terrain/*.tif
data/raw/drought/usdm_tiff/*.tif
data/raw/landfire/*.zip
data/raw/landfire/**/*.tif

outputs/
```

If processed CSVs are small enough, they can be committed to make the project easier to review. If they are large, include instructions for regenerating them instead.

## Tech Stack

- Python
- pandas
- numpy
- geopandas
- rasterio
- pygridmet
- py3dep
- xarray
- scikit-learn
- XGBoost
- matplotlib
- requests
- LANDFIRE Product Service
- U.S. Drought Monitor GeoTIFFs

## Resume Summary

Built a point-based wildfire severity prediction pipeline integrating CAL FIRE incident records, gridMET weather/fire-danger data, USGS 3DEP terrain, USDM drought intensity, and LANDFIRE vegetation/fuel rasters. Engineered pre-fire environmental features for 2,397 California wildfire incidents and evaluated severity classification with realistic time-based validation and imbalance-aware metrics.
