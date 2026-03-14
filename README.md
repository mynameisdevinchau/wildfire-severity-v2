# California Wildfire Severity Prediction
### CS163 Final Project

> Given weather and drought conditions at the time a fire is reported, how large is it likely to get?

California has seen a dramatic increase in wildfire severity over the past decade. While predicting *whether* a fire will start is one challenge, first responders and resource dispatchers face a different and equally urgent question the moment ignition is confirmed: **how much should we mobilise?** This project builds a machine learning model to estimate expected fire size from environmental conditions — framing the output as a fire risk tier that maps directly to real dispatch categories used by CAL FIRE.

---

## Project Structure

```
WILDFIRE-SEVERITY-V2/
├── data/
│   └── final_dataset.csv          # cleaned, merged dataset (2,579 fires × 49 features)
├── notebooks/
│   ├── preprocessing.ipynb        # data cleaning and feature engineering pipeline
│   ├── EDA.ipynb                  # exploratory data analysis
│   └── modeling.ipynb             # model training, evaluation, and risk tier analysis
├── requirements.txt
└── README.md
```

---

## Dataset & Data Sources

The final dataset contains **2,579 wildfire incidents** across **57 California counties** from **February 2013 to December 2024**. It was assembled by merging four independent sources:

| Source | Description | Records |
|--------|-------------|---------|
| [CAL FIRE Incidents](https://www.fire.ca.gov/) | Historical fire records with location, date, and acres burned | 2,870 raw incidents |
| [CIMIS](https://cimis.water.ca.gov/) | Daily weather station data (temperature, precipitation, wind, humidity, solar radiation) | 236,215 daily rows across 57 counties |
| [US Drought Monitor](https://droughtmonitor.unl.edu/) | Weekly county-level drought index (D0–D4 scale) | Merged and forward-filled to daily |
| [US Census](https://www.census.gov/) | County population, density, and land area | 57 counties |

### Target Variable

`incident_acres_burned` — highly right-skewed (median 89 acres, max 1.03M acres). All models predict `log(1 + acres)` and predictions are exponentiated back to acres for interpretation.

### Fire Size Tiers

Predictions are bucketed into four operational tiers matching CAL FIRE dispatch categories:

| Tier | Acres | Response Level |
|------|-------|----------------|
| Small | < 100 | Initial attack — 1–2 engines |
| Moderate | 100 – 1,000 | Extended attack — multiple resources |
| Large | 1,000 – 10,000 | Major incident — Type 2 team |
| Extreme | > 10,000 | Complex fire — Unified Command |

---

## Preprocessing Pipeline

The raw data required significant cleaning before it could be merged. Key steps:

**Fire data**
- Removed 77 rows with null or zero acre values
- Dropped 29 records with invalid coordinates (outside valid California bounds: lat 32–42, lon -124 to -114)
- Dropped `incident_type` column (43% null)

**CIMIS weather**
- Normalised county names to match across all sources
- Filled short gaps (≤ 7 days) using per-county time interpolation
- 9 counties with no CIMIS station were filled using the nearest neighbouring county's data:

  | County | Neighbour used |
  |--------|---------------|
  | Lake | Mendocino |
  | Mariposa | Madera |
  | Calaveras | Amador |
  | Tuolumne | Amador |
  | Trinity | Tehama |
  | Nevada | Placer |
  | Glenn | Colusa |
  | Mono | Inyo |
  | Sierra | Plumas |

**Drought index**
- Expanded weekly drought data to daily via forward-fill
- Computed composite `Drought_Score = D0 + D1 + D2 + D3 + D4` (range 0–500)
- Note: D0–D4 are cumulative percentages, not exclusive categories

**Feature engineering**
- `Temp_Range`: daily temperature swing (Max − Min)
- `Avg_Temp`: average of max and min
- `Is_Fire_Season`: binary flag for June–October
- `Season`: meteorological season
- `Fire_Impact_Per_Capita`: acres burned per county resident
- **7-day rolling weather windows**: for 10 weather variables, computed mean/max/sum over the 7 days *prior* to the fire date (using `shift(1)` before rolling to exclude the fire day itself — preventing data leakage)

**Collinear feature removal**
Before modeling, 16 features were dropped for having |r| > 0.85 with another feature. When a pair was redundant, the weaker predictor of the target was dropped. Final model uses 25 features.

---

## How to Run

### Requirements

```bash
pip install -r requirements.txt
```

`requirements.txt` should include:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
scipy
```

### Order of execution

Run the notebooks in this order:

```
1. notebooks/preprocessing.ipynb   — builds final_dataset.csv from raw sources
2. notebooks/EDA.ipynb             — exploratory analysis (optional, no outputs required)
3. notebooks/modeling.ipynb        — trains all models and produces results
```

Each notebook reads from `../data/` and expects the working directory to be `notebooks/`.

---

## Limitations

These are not afterthoughts — they are genuine constraints on how the model should be interpreted.

**1. Only fires are in the dataset**
Every row represents a confirmed ignition. The model has never seen a day where conditions were dangerous but no fire started. This means it cannot be used as an ignition predictor — it only estimates scale *given* that a fire has already been reported.

**2. The time-based split may flatter performance**
Training covers 2013–2021, which includes the catastrophic 2020 season (12.4M acres — roughly 5× the annual average). Testing covers 2022–2024, which were comparatively quieter years. The model may score well partly because the test set is easier, not because it has truly learned to generalise to extreme fire years.

**3. XGBoost near-perfectly fits training data**
XGBoost achieved R²=0.9992 on the training set, which indicates memorisation. While test performance remains strong (R²=0.97), this gap signals overfitting. The stacking ensemble partially mitigates this by blending XGBoost with Random Forest through cross-validated out-of-fold predictions.

**4. County-level resolution is coarse**
Weather and drought data are averaged to the county level. Two fires in the same county on the same day get identical feature values even if one is in a coastal valley and one is in a dry mountain range. Finer spatial resolution (grid-level or station-level) would improve signal considerably.

**5. No suppression or terrain data**
The model knows nothing about fuel load, slope, aspect, road access, or how quickly resources arrived. These factors heavily influence final fire size and explain most of the model's worst predictions — fires that burned far beyond what conditions alone would suggest.

---

## Future Work

- **Fire occurrence modeling**: rebuild the dataset with negative examples (non-fire days) to predict ignition probability, not just size given ignition
- **Leave-one-year-out cross-validation**: a more honest evaluation strategy that tests generalisation across different fire years rather than relying on a single train/test split
- **Finer spatial resolution**: replace county-level aggregation with grid-level weather data (e.g. PRISM, ERA5) to capture within-county variation
- **Fuel and terrain features**: integrate LANDFIRE fuel load data and DEM-derived slope/aspect features
- **Real-time inference**: wrap the stacking ensemble in a simple API that accepts current CIMIS readings and returns a predicted tier with confidence bounds
