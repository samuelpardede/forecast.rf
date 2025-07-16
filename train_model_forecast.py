import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

print("Mempersiapkan model FORECASTING untuk data ME 48...")

# --- 1. Persiapan Data ---
file_path = "E:/STMKG/SKRIPSI/datahujan/ME_48/Data_ME48_Bersih_Final.xlsx"
try:
    data = pd.read_excel(file_path, engine='openpyxl')
except Exception as e:
    print(f"Error membaca file: {e}")
    exit()

# Preprocessing
features = [
    "CLOUD_LOW_TYPE_CL", "CLOUD_LOW_MED_AMT_OKTAS", "CLOUD_MED_TYPE_CM", "CLOUD_HIGH_TYPE_CH",
    "CLOUD_COVER_OKTAS_M", "LAND_COND", "PRESENT_WEATHER_WW", "TEMP_DEWPOINT_C_TDTDTD",
    "TEMP_DRYBULB_C_TTTTTT", "TEMP_WETBULB_C", "WIND_SPEED_FF", "RELATIVE_HUMIDITY_PC",
    "PRESSURE_QFF_MB_DERIVED", "PRESSURE_QFE_MB_DERIVED"
]
target = "RR"

data.replace([8888, 9999], np.nan, inplace=True)
for col in features + [target]:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data.interpolate(method='linear', inplace=True)
data.dropna(subset=features + [target], inplace=True)

# --- 2. Restrukturisasi Data untuk Forecasting ---
df_forecast = data[features].copy()
df_forecast['target_RR'] = data[target].shift(-1)
df_forecast.dropna(inplace=True)

X = df_forecast[features]
y = df_forecast['target_RR']

# --- 3. Normalisasi & Pelatihan pada Seluruh Data ---
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

print("Melatih model dengan seluruh data forecasting...")
# Menggunakan hyperparameter terbaik dari hasil tuning forecasting
params_rf = {
    'max_depth': 15, 'max_features': 'sqrt', 'min_samples_leaf': 5, 
    'min_samples_split': 17, 'n_estimators': 262
}
model = RandomForestRegressor(**params_rf, random_state=42, n_jobs=-1)
model.fit(X_scaled, y_scaled.ravel())

# --- 4. Simpan Model dan Scaler ---
# Beri nama berbeda untuk membedakan dengan model hindcasting
joblib.dump(model, 'rf_forecast_model.pkl')
joblib.dump(scaler_X, 'scaler_X_forecast.pkl')
joblib.dump(scaler_y, 'scaler_y_forecast.pkl')

print("\nModel FORECASTING dan scaler telah berhasil disimpan!")