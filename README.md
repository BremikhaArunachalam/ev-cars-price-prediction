# EV Cars Price Prediction

A machine learning project to predict the price of electric vehicles in the German market based on battery capacity, range, efficiency, speed, and charging specifications.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [ML Pipeline](#ml-pipeline)
- [Model Results](#model-results)
- [Author](#author)

---

## Project Overview

This project builds an end-to-end ML regression pipeline to predict the market price of electric vehicles (in Germany) based on technical specifications. The dataset contains 360 EV records with features spanning battery capacity, driving range, energy efficiency, fast charge speed, top speed, and acceleration.

The target variable is `Price.DE.` — a continuous value representing the price of the EV in the German market (EUR). The pipeline covers the full ML workflow: data loading, missing value handling, categorical encoding, outlier treatment, feature scaling, feature selection, model training, and R² evaluation across 7 regression models.

---

## Repository Structure

```
ev-cars-price-prediction/
├── EV_Cars_Sales.ipynb           # End-to-end ML pipeline notebook
└── EV_cars.csv                   # Dataset (360 EV records)
```

---

## Dataset

**File:** `EV_cars.csv`  
**Source:** Kaggle  
**Total Records:** 360  
**Target Variable:** `Price.DE.` — EV price in the German market (EUR 22,550 – EUR 2,18,000)

| Feature | Type | Description |
|---|---|---|
| `Car_name` | Categorical | Name of the electric vehicle |
| `Car_name_link` | Categorical | URL link to the EV database entry |
| `Battery` | Float | Battery capacity in kWh (21.3 – 123.0) |
| `Efficiency` | Integer | Energy consumption in Wh/km (137 – 295) |
| `Fast_charge` | Float | Fast charging speed in km/h |
| `Range` | Integer | Driving range on full charge in km (135 – 685) |
| `Top_speed` | Integer | Maximum speed in km/h (125 – 320) |
| `acceleration..0.100.` | Float | Time to accelerate from 0 to 100 km/h in seconds |
| `Price.DE.` | Float | Target — Vehicle price in Germany (EUR) |

### Summary Statistics

| Feature | Mean | Std Dev | Min | Max |
|---|---|---|---|---|
| Battery (kWh) | 71.19 | 20.39 | 21.3 | 123.0 |
| Range (km) | — | — | 135 | 685 |
| Top Speed (km/h) | 180.92 | 36.23 | 125 | 320 |
| Acceleration (0–100s) | 7.29 | 3.01 | 2.1 | 19.1 |
| Efficiency (Wh/km) | 195.18 | 31.91 | 137 | 295 |

---

## ML Pipeline

### 1. Data Loading and Exploration

- Loaded dataset using `pandas`
- Inspected shape, data types, and missing value counts
- Noted missing values: `Fast_charge` (2), `Price.DE.` (51)

### 2. Missing Value Handling

- **Numerical columns** — imputed using mean via `SimpleImputer(strategy='mean')`
- **Categorical columns** — imputed using mode via `SimpleImputer(strategy='most_frequent')`

### 3. Categorical Encoding

- Applied **One-Hot Encoding** on `Car_name` and `Car_name_link` using `pd.get_dummies()` with `drop_first=True` to avoid multicollinearity

### 4. Outlier Treatment

- Used **IQR (Interquartile Range)** method for all numerical columns
- Capped values below `Q1 - 1.5 * IQR` to the lower bound
- Capped values above `Q3 + 1.5 * IQR` to the upper bound

### 5. Feature Selection

- Applied `SelectKBest` with `f_regression` to identify the top 10 most relevant features correlated with `Price.DE.`
- Selected core features: `Battery`, `Efficiency`, `Fast_charge`, `Range`, `Top_speed`, `acceleration..0.100.`
- Computed and sorted a correlation matrix by the target variable

### 6. Feature Scaling

- Applied `StandardScaler` to normalize all feature values to zero mean and unit variance before model training

### 7. Train-Test Split

| Parameter | Value |
|---|---|
| Test size | 30% (108 records) |
| Train size | 70% (252 records) |
| Random state | 100 |

---

## Model Results

All 7 regression models were trained on the same preprocessed dataset and evaluated using **R² Score (Coefficient of Determination)**.

| # | Model | R² Score |
|---|---|---|
| 1 | **Gradient Boosting** | **0.7775** |
| 2 | Random Forest | 0.7748 |
| 3 | K-Nearest Neighbors | 0.7668 |
| 4 | Support Vector Regression | 0.7477 |
| 5 | Linear Regression | 0.6978 |
| 6 | Ridge Regression | 0.6976 |
| 7 | Decision Tree | 0.6149 |

**Best performing model: Gradient Boosting — R² Score of 0.7775**

---

## Author

**Bremikha Arunachalam**  
GitHub: [github.com/BremikhaArunachalam](https://github.com/BremikhaArunachalam)
