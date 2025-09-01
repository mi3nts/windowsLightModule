# LOAD PACKAGES
import os
import time
import pickle
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
import numpy as np
from pvlib.solarposition import get_solarposition
#import joblib 
import joblib
from io import StringIO




# USER CONFIGURATIONS 
BASE_DIR = "C:/Users/yichao/Desktop/mintsData/raw/001e06453e58"
DEVICE_ID = "MINTS_001e06453e58"
SENSOR_FILES = {
    'AS7265X': 'AS7265X',
    'GUVAS12SDV2': 'GUVAS12SDV2',
    'LTR390V2': 'LTR390V2',
    'GPS': 'GPSGPGGA2'
}
RESAMPLE_INTERVAL = '10s'
BUFFER_SECONDS = 60
MODEL_PATH = 'ConformalPredictionModel.pkl'
OUTPUT_FOLDER = BASE_DIR
#os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# LOAD CSV WITH SR DATA
dfSat = pd.read_csv('SurfaceReflectance.csv')

# LOAD ML MODEL 
with open(MODEL_PATH, 'rb') as f:
    model = joblib.load(f)
print("Loaded model type:", type(model))

#  BUFFER AND FILE POSITION TRACKING 
buffers = {sensor: deque() for sensor in SENSOR_FILES}
file_positions = {sensor: 0 for sensor in SENSOR_FILES}

#  FEATURE COLUMN GROUPS 
AS7265X_COLS = [
    'channelA410nm', 'channelA435nm', 'channelA460nm', 'channelA485nm',
    'channelA510nm', 'channelA535nm', 'channelA560nm', 'channelA585nm',
    'channelA610nm', 'channelA645nm', 'channelA680nm', 'channelA705nm',
    'channelA730nm', 'channelA760nm', 'channelA810nm', 'channelA860nm',
    'channelA900nm', 'channelA940nm'
]
GUVAS_COLS = ['uvShunt', 'uvBus']
LTR390_COLS = ['als', 'uvs']
SR_COLS = ['B1', 'B2', 'B3', 'B4','B5','B6','B7','B8','B8A']

#  GET TODAY'S FILE PATHS 
today = datetime.now().strftime('%Y/%m/%d')
folder_path = os.path.join(BASE_DIR, today)
file_paths = {
    sensor: os.path.join(folder_path, f"{DEVICE_ID}_{filename}_{today.replace('/', '_')}.csv")
    for sensor, filename in SENSOR_FILES.items()
}

# READ COLUMN NAMES FOR EACH SENSOR ONCE 
SENSOR_COLNAMES = {}

def get_sensor_colnames(sensor, path):
    with open(path, 'r') as f:
        header_line = f.readline()
    return header_line.strip().split(',')

for sensor, path in file_paths.items():
    try:
        if os.path.exists(path):
            SENSOR_COLNAMES[sensor] = get_sensor_colnames(sensor, path)
        else:
            SENSOR_COLNAMES[sensor] = None
            print(f"File for sensor {sensor} does not exist yet: {path}")
    except Exception as e:
        SENSOR_COLNAMES[sensor] = None
        print(f"Could not read header for {sensor}: {e}")

# HELPER FUNCTIONS 

def read_new_data(sensor, path):
    global file_positions
    try:
        if not os.path.exists(path):
            # File might not exist yet if it is still early in the day
            return pd.DataFrame()
        with open(path, 'r') as f:
            f.seek(file_positions[sensor])
            lines = f.readlines()
            file_positions[sensor] = f.tell()
        if not lines:
            return pd.DataFrame()
        # Skip header line if present in chunk
        if lines[0].startswith('dateTime'):
            lines = lines[1:]
        if not lines:
            return pd.DataFrame()
        # Read without header, assign column names explicitly
        if SENSOR_COLNAMES[sensor] is None:
            print(f"No column names for sensor {sensor}, skipping read")
            return pd.DataFrame()
        df = pd.read_csv(StringIO(''.join(lines)), header=None, names=SENSOR_COLNAMES[sensor])
        df['dateTime'] = pd.to_datetime(df['dateTime'])
        return df
    except Exception as e:
        print(f"Error reading {sensor}: {e}")
        return pd.DataFrame()

def trim_buffer(sensor):
    now = datetime.now()
    buffers[sensor] = deque([row for row in buffers[sensor] if row['dateTime'] > now - timedelta(seconds=BUFFER_SECONDS)])

def get_df_from_buffer(sensor):
    if not buffers[sensor]:
        return pd.DataFrame()
    return pd.DataFrame(buffers[sensor])

def calculate_solar_angles(timestamp, lat, lon, alt):
    ts = pd.DatetimeIndex([timestamp])
    solpos = get_solarposition(ts, latitude=lat, longitude=lon, altitude=alt)
    return solpos['zenith'].values[0], solpos['azimuth'].values[0]

# === OUTPUT SETUP ===
now = datetime.now()
year = now.strftime("%Y")
month = now.strftime("%m")
day = now.strftime("%d")

output_subfolder = os.path.join(OUTPUT_FOLDER, year, month, day)
os.makedirs(output_subfolder, exist_ok=True)

output_file = os.path.join(output_subfolder, f"MINTS_001e06453e58_CP777_{year}_{month}_{day}.csv")

wavelengths = [f"{wl} nm" for wl in range(360, 781)]  # 360 to 780 inclusive
output_columns = ['dateTime'] + wavelengths

if not os.path.exists(output_file):
    pd.DataFrame(columns=output_columns).to_csv(output_file, index=False)

SR_Bands = dfSat[SR_COLS].iloc[0]

# MAIN LOOP 
print("Starting real-time inference loop...")
predicted_timestamps = set()

while True:
    now = datetime.now()
    for sensor, path in file_paths.items():
        new_data = read_new_data(sensor, path)
        if not new_data.empty:
            buffers[sensor].extend(new_data.to_dict('records'))
            trim_buffer(sensor)

    # Convert buffers to resampled DataFrames
    dfs = {}
    for s in buffers:
        df = get_df_from_buffer(s)
        if df.empty or 'dateTime' not in df.columns:
            dfs[s] = pd.DataFrame()
        else:
            df = df.set_index('dateTime')
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            dfs[s] = df[numeric_cols].resample(RESAMPLE_INTERVAL).mean()

    # Find latest valid timestamp for AS7265X sensor 
    common_index = dfs['AS7265X'].last_valid_index()
    if common_index is None or common_index in predicted_timestamps:
        time.sleep(10)
        continue

    predicted_timestamps.add(common_index)
    # Limit set size to last 500 timestamps
    if len(predicted_timestamps) > 1000:
        predicted_timestamps = set(list(predicted_timestamps)[-500:])

    try:
        # Extract features for prediction
        x_as = dfs['AS7265X'].loc[common_index, AS7265X_COLS]
        x_uv = dfs['GUVAS12SDV2'].loc[common_index, GUVAS_COLS]
        x_ltr = dfs['LTR390V2'].loc[common_index, LTR390_COLS]

        gps_df = dfs['GPS']
        gps_row = gps_df.loc[common_index]
        lat = gps_row['latitudeCoordinate']
        lon = gps_row['longitudeCoordinate']
        alt = gps_row['altitude']

        zen, azi = calculate_solar_angles(common_index, lat, lon, alt)

        feature_dict = {
            **x_as.to_dict(),
            **x_uv.to_dict(),
            **x_ltr.to_dict(),
            **SR_Bands.to_dict(),
            'solarZenith': zen,
            'solarAzimuth': azi
        }
        feature_df = pd.DataFrame([feature_dict])

        prediction = model.predict(feature_df, alpha=0.2)[0]
        prediction = np.array(prediction).flatten()

        output_row = pd.DataFrame([[common_index.isoformat()] + list(prediction)], columns=output_columns)
        output_row.to_csv(output_file, mode='a', header=False, index=False)

        print(f"[{common_index}] Predicted Irradiance shape: {prediction.shape}")
        print(f"[{common_index}] Predicted Irradiance (first 5 values): {prediction[:5]}")

    except Exception as e:
        print(f"Skipping due to error: {e}")

    time.sleep(10)
