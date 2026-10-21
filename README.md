# Wind Farm Renewable Energy Optimization Prototype

A Streamlit prototype for exploring a wind-farm dataset, visualizing it, comparing four regression models, and reviewing model-based insights.

## Features
- Upload CSV or Excel (`.csv`, `.xlsx`, `.xls`) data.
- Exploratory data analysis: histogram, box plot, bar chart, scatter plot, line graph, correlation heatmap.
- Four regression models:
  - Linear Regression (one selected primary feature)
  - Multiple Linear Regression (all selected features)
  - Decision Tree Regression
  - Random Forest Regression
- Regression metrics: MAE, MSE, RMSE, R².
- Accuracy and F1 Score from a separate binary interpretation of power generation:
  - `1`: high generation (power at or above the median of the training target)
  - `0`: normal/low generation (below that threshold)
- Download model comparison, predictions, and feature-importance CSV files.

## Dataset requirements
The dataset must include the numeric target column `power_29_avg`. The app starts with common wind/sensor columns if present and lets you select other numeric features.

For reliable results, use data you are authorized to share and check for missing values, data leakage, and sensor/asset/time identifiers. Power-derived columns are not selected by default as features.

## Run locally
Python 3.10 or newer is recommended.

### Windows PowerShell
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run src/app.py
```

### Linux/macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run src/app.py
```

Open the local URL printed by Streamlit, then upload your dataset.

## Important metric note
MAE, MSE, RMSE, and R² evaluate continuous power prediction. Accuracy and F1 Score are classification metrics, so this app first converts actual and predicted power into two classes using the median of the training target as a threshold. Report these as threshold-based high/normal-generation classification metrics, not as ordinary regression accuracy.

## Project structure
```text
wind_farm_optimization_prototype/
├── src/
│   └── app.py
├── data/
├── results/
├── docs/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

## Limitations
This is an academic prototype, not a turbine-control system. Model associations do not establish causation, and recommendations should be validated by qualified wind-farm engineers before operational use.
