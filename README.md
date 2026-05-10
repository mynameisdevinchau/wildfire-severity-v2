# California Wildfire Severity Prediction

A point-based geospatial machine learning project for modeling California wildfire severity using pre-fire weather, terrain, drought, and vegetation/fuel conditions.

> Inspired by recent California wildfire events and my background in fire technology, this project explores how data science can support wildfire risk analysis and preparedness. The final model is not intended for operational deployment, but it demonstrates a defensible environmental feature pipeline and evaluates whether public geospatial datasets contain meaningful signal for high-severity wildfire triage.

---

## Project Summary

This project predicts wildfire severity for confirmed California wildfire incidents. It began as a coarse county-level analysis, but was rebuilt into a point-based geospatial pipeline that samples environmental features at each fire's ignition coordinates.

The final dataset combines:

- **CAL FIRE incident records**
- **gridMET weather and fire-danger variables**
- **USGS 3DEP terrain features**
- **U.S. Drought Monitor drought intensity**
- **LANDFIRE vegetation and fuel layers**

The final modeling table contains:

```text
2,397 California wildfire incidents
197 total columns after enrichment
No duplicate fire IDs
Low missingness across core modeling features
```

Final enriched table:

```text
data/processed/calfire_with_gridmet_terrain_drought_veg.csv
```

---

## Research Question

Given the location and pre-fire environmental conditions for a confirmed wildfire, can we estimate whether the fire is likely to become high-severity?

The project evaluates two related tasks:

1. **Multiclass severity prediction**
   - Small
   - Medium
   - Large
   - Extreme

2. **Binary high-severity triage**
   - Low: Small or Medium
   - High: Large or Extreme

The binary task is the more practical framing because identifying potentially high-severity fires is more useful than perfectly predicting every fire-size tier.

---

## Why Not Predict Raw Acres Directly?

Final acres burned is extremely right-skewed. Most fires burn relatively small areas, while a small number of massive fires dominate the distribution.

Summary from the final EDA:

```text
Median acres burned:        80
Mean acres burned:       4,332
95th percentile:         8,806
99th percentile:        83,844
Maximum:             1,032,648
```

Because of this skew, raw-acreage regression is unstable and difficult to interpret. The project uses `log_acres` for EDA, but the main target is a severity-tier classification label.

---

## Target Definition

Severity tiers are derived from NWCG-style fire-size classes and collapsed into four modeling groups.

| Severity Tier | Acreage Range |
|---|---:|
| Small | < 100 acres |
| Medium | 100 to < 1,000 acres |
| Large | 1,000 to < 5,000 acres |
| Extreme | >= 5,000 acres |

Final target distribution:

| Severity Tier | Count |
|---|---:|
| Small | 1,311 |
| Medium | 720 |
| Large | 200 |
| Extreme | 166 |

Because the target is imbalanced, the project reports:

- Balanced accuracy
- Macro F1
- Per-class recall
- Large/Extreme recall
- ROC-AUC for the binary task

Raw accuracy is not used as the main metric because a model can score well by predicting only the majority class.

---

## Data Sources

### CAL FIRE Incidents

CAL FIRE incident records form the base wildfire dataset. Each record includes the fire name, start date, location, and final acres burned.

Key fields used:

- Fire name
- Start date
- Latitude
- Longitude
- Acres burned

### gridMET Weather and Fire-Danger Data

gridMET variables were sampled at each wildfire ignition coordinate using `pygridMET`. For each fire, the pipeline computes pre-fire weather and fire-danger aggregates over 7-day, 14-day, and 30-day windows.

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

### USGS 3DEP Terrain

USGS 3DEP terrain rasters were sampled at each fire point using `py3DEP` and `rasterio`.

Features include:

- Elevation
- Slope
- Aspect
- Northness
- Eastness

Aspect was decomposed into northness and eastness because aspect is circular and should not be treated as a simple linear variable.

### U.S. Drought Monitor

Weekly U.S. Drought Monitor GeoTIFFs were downloaded and sampled at each fire point. For each fire, the pipeline selected the most recent drought map available before the fire start date to avoid temporal leakage.

Features include:

- USDM raw raster value
- USDM category
- USDM intensity
- D0+ abnormally dry indicator
- D1+ drought indicator
- D2+ severe drought indicator
- D3+ extreme drought indicator

### LANDFIRE Vegetation and Fuels

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

Raw vegetation codes are retained for EDA, while broader fuel groups are used for more interpretable modeling.

---

## Pipeline

The final workflow follows a structured `raw/interim/processed` data architecture.

```text
CAL FIRE incidents
        |
        v
Clean and standardize fire records
        |
        v
Assign each fire a stable gridmet_id
        |
        v
Sample gridMET daily weather/fire-danger data at ignition coordinates
        |
        v
Aggregate 7-day, 14-day, and 30-day pre-fire windows
        |
        v
Download and sample USGS 3DEP terrain rasters
        |
        v
Download and sample weekly USDM drought rasters
        |
        v
Request and sample LANDFIRE vegetation/fuel rasters
        |
        v
Build final enriched modeling table
        |
        v
Run final EDA and modeling
```

---

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

---

## Exploratory Data Analysis

The final EDA supports the project framing and modeling decisions.

Key findings:

- Final acres burned is highly right-skewed, making exact acreage prediction unstable.
- The severity target is imbalanced, with Small fires forming the largest class.
- Larger fires generally occur under drier and more fire-dangerous conditions.
- Higher-severity fires tend to show higher VPD, higher ERC, higher Burning Index, lower relative humidity, and lower fuel moisture.
- Larger fires tend to occur at higher elevations and steeper slopes.
- Extreme fires show higher average USDM drought intensity than Small fires.
- LANDFIRE fuel groups differ across severity tiers, with larger fires showing greater representation in shrub and timber-related fuel groups.
- Several fire-danger features are highly correlated, especially ERC, Burning Index, fuel moisture, temperature, and VPD.
- The 2016-2021 train period and 2022-2024 test period show distribution shift, so model results must be interpreted carefully.

Final EDA notebook:

```text
notebooks/08_final_eda.ipynb
```

---

## Modeling Setup

The project uses a time-based split:

```text
Train: 2016-2021
Test:  2022-2024
```

This is more realistic than a random split because it tests generalization to future fire seasons.

Models evaluated:

- Majority-class baseline
- Logistic Regression
- Random Forest
- XGBoost

Feature groups compared:

1. `gridMET + terrain`
2. `gridMET + terrain + drought`
3. `gridMET + terrain + drought + LANDFIRE`
4. Expanded LANDFIRE code version

Final modeling notebook:

```text
notebooks/09_final_modeling.ipynb
```

---

## Results

### Multiclass Severity Prediction

Task:

```text
Small vs Medium vs Large vs Extreme
```

Best multiclass model:

| Model | Features | Accuracy | Balanced Accuracy | Macro F1 | Large/Extreme Recall |
|---|---|---:|---:|---:|---:|
| Random Forest | gridMET + terrain + drought + LANDFIRE | 0.4965 | 0.3543 | 0.3353 | 0.4045 |

The majority baseline achieved higher raw accuracy by predicting the most common class, but it had:

```text
Large/Extreme recall: 0.0000
Macro F1: 0.1974
Balanced accuracy: 0.2500
```

This confirms that raw accuracy is misleading for this imbalanced task.

### Binary High-Severity Triage

Task:

```text
Low  = Small or Medium
High = Large or Extreme
```

Best binary model:

| Model | Features | Accuracy | Balanced Accuracy | Macro F1 | High Recall | High Precision | ROC-AUC | Average Precision |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | gridMET + terrain + drought | 0.8640 | 0.6210 | 0.6240 | 0.3146 | 0.3333 | 0.6942 | 0.2732 |

The binary task is the stronger practical framing. It evaluates whether the model can flag potentially high-severity fires rather than perfectly separating all four size classes.

---

## Threshold Tuning

At the default 0.50 threshold, the binary Random Forest is conservative:

| Threshold | High Recall | High Precision | Macro F1 |
|---:|---:|---:|---:|
| 0.35 | 0.5843 | 0.2088 | 0.5692 |
| 0.40 | 0.4944 | 0.2431 | 0.6002 |
| 0.45 | 0.3933 | 0.2536 | 0.6016 |
| 0.50 | 0.3146 | 0.3333 | 0.6240 |

For a triage use case, a threshold near `0.40` may be more useful because it catches nearly half of high-severity fires, though it increases false positives.

---

## Interpretation

The final results suggest that point-based weather, terrain, drought, and fuel variables contain meaningful wildfire severity signal. However, exact four-tier prediction remains difficult because final fire size depends on many factors that are not fully captured in pre-fire environmental data.

The most defensible interpretation is:

```text
This project is a point-based geospatial wildfire severity and high-severity triage pipeline, not an operational wildfire prediction system.
```

The strongest model story is not that the model can perfectly predict final acres burned. Instead, the project demonstrates that a more spatially precise feature pipeline can identify useful environmental risk signals and support a realistic high-severity triage framing.

---

## Limitations

This project only models confirmed wildfire incidents. It does not predict whether a fire will ignite.

Important missing predictors include:

- Ignition cause
- Suppression response
- Initial attack timing
- Fire perimeter growth
- Fuel continuity beyond the ignition point
- Road access
- WUI and population exposure
- Extreme wind direction and wind gust events
- Lightning and human ignition context
- Real-time resource availability

The model should not be used for operational wildfire decision-making.

---

## Future Work

Potential improvements:

- Add ignition cause.
- Add WUI and population exposure.
- Add road access, distance to cities, and distance to fire stations.
- Add buffer-based fuel continuity metrics instead of only point-sampled fuel values.
- Use fire perimeter data to summarize vegetation/fuel conditions across burned area.
- Add multi-week or multi-month drought lag features.
- Add extreme wind and red-flag warning features.
- Use leave-one-year-out cross-validation.
- Tune Random Forest and XGBoost hyperparameters.
- Calibrate predicted probabilities for high-severity triage.
- Build an interactive dashboard or map-based model explainer.

---

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

5. Review outputs.

```text
outputs/final_eda/
outputs/final_modeling/
```

---

## Git and Data Notes

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

Processed CSVs can be committed if they are small enough for GitHub. If they are large, regenerate them using the notebooks.

---

## Tech Stack

- Python
- pandas
- NumPy
- GeoPandas
- rasterio
- pygridMET
- py3DEP
- xarray
- scikit-learn
- XGBoost
- matplotlib
- requests
- LANDFIRE Product Service
- U.S. Drought Monitor GeoTIFFs

---

## Resume Summary

Built a point-based wildfire severity prediction pipeline integrating CAL FIRE incident records, gridMET weather/fire-danger data, USGS 3DEP terrain, USDM drought intensity, and LANDFIRE vegetation/fuel rasters. Engineered pre-fire environmental features for 2,397 California wildfire incidents and evaluated multiclass severity classification and binary high-severity triage models using realistic time-based validation and imbalance-aware metrics.
