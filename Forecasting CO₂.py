import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import joblib


DATA_PATH = "data/co2_worldbank.csv"   # put your cleaned CSV here
TARGET_COL = "co2_per_capita"          # or 'co2_total'
COUNTRY_COL = "country"
YEAR_COL = "year"


# 1) Load data
df = pd.read_csv(DATA_PATH)
print("Rows:", len(df), "Columns:", df.columns.tolist())
# Expect columns: country, year, co2_per_capita, gdp_per_capita, population, energy_use, ...

# 2) Simple cleaning
df = df.dropna(subset=[COUNTRY_COL, YEAR_COL, TARGET_COL])
df[YEAR_COL] = df[YEAR_COL].astype(int)

# 3) Create lag features per country
def add_lags(group, col, lags=[1,2]):
    for lag in lags:
        group[f"{col}_lag{lag}"] = group[col].shift(lag)
    return group

df = df.groupby(COUNTRY_COL).apply(lambda g: add_lags(g, TARGET_COL, lags=[1,2])).reset_index(drop=True)
df = df.dropna()  # drop rows where lag features are missing

# 4) Features and target
FEATURES = [c for c in df.columns if c not in [COUNTRY_COL, YEAR_COL, TARGET_COL]]
X = df[FEATURES]
y = df[TARGET_COL]

# 5) Time-based train/test split
# We'll split by year: train <= 2015, test >= 2016 (adjust as needed)
train_mask = df[YEAR_COL] <= 2015
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[~train_mask], y[~train_mask]
print("Train:", X_train.shape, "Test:", X_test.shape)

# 6) Baseline model
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
pred = model.predict(X_test)

# 7) Evaluate
mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred, squared=False)
mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

# 8) Plot for a sample 
sample_country = "Kenya"
df_country = df[df[COUNTRY_COL] == sample_country].sort_values(YEAR_COL)
if not df_country.empty:
    years = df_country[YEAR_COL]
    actual = df_country[TARGET_COL]
    # For predicted values we need to predict on that subset's test slice
    mask_country_test = (df_country[YEAR_COL] >= 2016)
    X_country_test = df_country[FEATURES][mask_country_test]
    pred_country = model.predict(X_country_test)
    plt.figure(figsize=(8,4))
    plt.plot(years, actual, label='Actual', marker='o')
    plt.plot(years[mask_country_test], pred_country, label='Predicted', marker='x')
    plt.title(f"{sample_country} - {TARGET_COL} Actual vs Predicted")
    plt.xlabel("Year"); plt.ylabel(TARGET_COL)
    plt.legend(); plt.grid(True)
    plt.show()

# 9) Save model
joblib.dump(model, "rf_co2_model.joblib")
