import os
import glob
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

def ingest_strengthsense():
    base_dir = "data/StrengthSense"
    output_data_npy = "data/strength_sense_data.npy"
    output_labels_npy = "data/strength_sense_labels.npy"
    output_sensors_npy = "data/strength_sense_sensors.npy"
    output_csv = "data/df_strength_sense.csv"
    
    # Define mapping from string activity labels to integers
    # 13 semantic activities identified from metadata. 
    # 1w and 1wid are counted separately here for fine-grained labels.
    activity_map = {
        "1(w)": 0,    # Flat walking
        "1(wid)": 1,  # Incline/Decline walking
        "2": 2,       # Rise from seat
        "3": 3,       # Walk with shopping cart
        "4": 4,       # Vacuuming
        "5": 5,       # Squat to lying
        "6": 6,       # Stand to sit
        "7": 7,       # Stand to sit to lying
        "8": 8,       # Greet camera
        "9": 9,       # Drink water
        "10": 10,     # Stairs (up/down)
        "11": 11,     # Push-ups
        "12": 12,     # Sit-ups
        "13": 13      # Squats
    }
    
    # 10 sensor placements defined in metadata
    sensor_map = {
        "CHS": 0, # Chest
        "LU": 1,  # Left Upper Arm
        "RU": 2,  # Right Upper Arm
        "LF": 3,  # Left Forearm
        "RF": 4,  # Right Forearm
        "WAS": 5, # Waist
        "LT": 6,  # Left Thigh
        "RT": 7,  # Right Thigh
        "LC": 8,  # Left Calf/Shin
        "RC": 9   # Right Calf/Shin
    }
    
    # Inverse map for logging
    inv_sensor_map = {v: k for k, v in sensor_map.items()}
    
    all_data = []    # List of (T, 6) arrays
    all_labels = []  # List of int
    all_sensors = [] # List of int
    metadata = []    # List of dicts
    
    # Constants for unit conversion
    G_TO_MPS2 = 9.80665
    DEG_TO_RAD = np.pi / 180.0
    
    print("Scanning for files...")
    search_pattern = os.path.join(base_dir, "subject*", "laptop*", "IMU9", "*.[cC][sS][vV]")
    files = glob.glob(search_pattern)
    print(f"Found {len(files)} CSV files.")
    
    if len(files) == 0:
        print("No files found. Exiting.")
        return

    for fpath in tqdm(files):
        try:
            basename = os.path.basename(fpath)
            # Example filename: s1_a1(w)_t1_u.csv
            match = re.match(r's(\d+)_a(.*?)_t(\d+)_([ul]).*', basename, re.IGNORECASE)
            if not match:
                continue
                
            subject_id = int(match.group(1))
            act_str = match.group(2).lower()
            trial_id = int(match.group(3))
            body_part = match.group(4).lower() # 'u' or 'l'
            
            if act_str not in activity_map:
                print(f"Warning: Unknown activity '{act_str}' in {basename}")
                continue
                
            activity_label = activity_map[act_str]
            
            # Read CSV
            df = pd.read_csv(fpath)
            if df.empty:
                continue
                
            # Drop rows with NaN if any sensor failed momentarily
            df = df.dropna()
            
            # Find all sensor prefixes in this file 
            # Columns look like: CHS_IMU9_Acc_X
            prefixes = set()
            for col in df.columns:
                if "_IMU9_Acc_" in col:
                    prefixes.add(col.split("_")[0])
                    
            for prefix in prefixes:
                if prefix not in sensor_map:
                    continue
                    
                sensor_id = sensor_map[prefix]
                
                # Extract Acc (X, Y, Z) and Gyro (X, Y, Z) for this specific sensor
                cols = [
                    f"{prefix}_IMU9_Acc_X", f"{prefix}_IMU9_Acc_Y", f"{prefix}_IMU9_Acc_Z",
                    f"{prefix}_IMU9_Gyro_X", f"{prefix}_IMU9_Gyro_Y", f"{prefix}_IMU9_Gyro_Z"
                ]
                
                # Check if all 6 columns exist
                if not all(c in df.columns for c in cols):
                    continue
                    
                sensor_data = df[cols].values.copy()
                
                # Unit Conversions
                # Accelerometer: G -> m/s^2
                sensor_data[:, 0:3] *= G_TO_MPS2
                # Gyroscope: deg/s -> rad/s
                sensor_data[:, 3:6] *= DEG_TO_RAD
                
                # Append to datasets
                all_data.append(sensor_data)
                all_labels.append(activity_label)
                all_sensors.append(sensor_id)
                
                metadata.append({
                    "subject": subject_id,
                    "activity_str": act_str,
                    "activity_id": activity_label,
                    "trial": trial_id,
                    "body_part_file": body_part,
                    "sensor_prefix": prefix,
                    "sensor_id": sensor_id,
                    "length": sensor_data.shape[0]
                })
                
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    # Save to disk
    final_data = np.array(all_data, dtype=object)
    final_labels = np.array(all_labels, dtype=int)
    final_sensors = np.array(all_sensors, dtype=int)
    
    print(f"\nSaving {len(final_data)} sensor-specific samples...")
    np.save(output_data_npy, final_data)
    np.save(output_labels_npy, final_labels)
    np.save(output_sensors_npy, final_sensors)
    
    df_meta = pd.DataFrame(metadata)
    df_meta.to_csv(output_csv, index=False)
    
    print("Ingestion Complete.")
    print("\nClass Distribution:")
    print(df_meta['activity_str'].value_counts())
    print("\nSensor Distribution:")
    print(df_meta['sensor_prefix'].value_counts())

if __name__ == "__main__":
    ingest_strengthsense()
