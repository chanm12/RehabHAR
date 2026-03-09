
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

def ingest_keen_pad():
    base_dir = "data/12112951/dataset"
    output_npy = "data/keen_pad_data.npy"
    output_csv = "data/df_keen_pad.csv"
    
    # Activity Mapping (from metadata.txt)
    activity_map = {
        0: "Squat (Correct)",
        1: "Squat (Weight on Healthy)",
        2: "Squat (Injured Forward)",
        3: "Seated Ext (Correct)",
        4: "Seated Ext (No ROM)",
        5: "Seated Ext (Lift Limb)",
        6: "Walking (Correct)",
        7: "Walking (Not Extended)",
        8: "Walking (Hip Abduction)"
    }
    
    # Storage
    all_data = [] # List of (T, 6) arrays
    metadata = [] # List of dicts
    
    # Constants for Unit Conversion
    G_TO_MPS2 = 9.81
    DEG_TO_RAD = np.pi / 180.0
    
    # 1. Walk through directories
    # Structure: Subject_X / Activity_ID / Trial_Y / imu.npy
    print("Scanning for files...")
    search_pattern = os.path.join(base_dir, "Subject_*", "*", "Trial_*", "imu.npy")
    files = glob.glob(search_pattern)
    print(f"Found {len(files)} IMU files.")
    
    for fpath in tqdm(files):
        try:
            # Parse Path
            parts = fpath.split(os.sep)
            # parts[-1] = imu.npy
            # parts[-2] = Trial_Y
            # parts[-3] = Activity_ID
            # parts[-4] = Subject_X
            
            subject_id = parts[-4]
            activity_id_str = parts[-3]
            trial_id = parts[-2]
            
            try:
                activity_id = int(activity_id_str)
                label_text = activity_map.get(activity_id, "Unknown")
            except:
                label_text = "Unknown"
            
            # Load Data
            # Shape is (48, T) where 48 = 8 sensors * 6 channels
            raw = np.load(fpath)
            
            # Transpose to (T, 48) for easier slicing
            # Assuming raw order is Channels x Time given previous check (48, 448)
            data_t = raw.T # Now (T, 48)
            
            # Extract Sensor 1 (Channels 0-5)
            # 0-2: Acc X,Y,Z
            # 3-5: Gyro X,Y,Z
            s1_data = data_t[:, 0:6].copy()
            
            # Unit Conversion
            # Acc (cols 0-2): G -> m/s^2
            s1_data[:, 0:3] *= G_TO_MPS2
            
            # Gyro (cols 3-5): Deg/s -> Rad/s
            s1_data[:, 3:6] *= DEG_TO_RAD
            
            # Add to list
            all_data.append(s1_data)
            
            metadata.append({
                "subject": subject_id,
                "activity_id": activity_id,
                "label": label_text,
                "trial": trial_id,
                "length": s1_data.shape[0]
            })
            
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    # 2. Pad Signals (to max length) or Object Array?
    # Our batch script handles object array (list of arrays)
    # prompt.py expects (N, T, 6) if typically fixed, or object array if variable.
    # Given 'Trial' nature, lengths vary. Let's save as object array to preserve raw lengths.
    
    final_data = np.array(all_data, dtype=object)
    
    # 3. Save
    print(f"Saving {len(final_data)} samples to {output_npy}...")
    np.save(output_npy, final_data)
    
    df = pd.DataFrame(metadata)
    print(f"Saving metadata to {output_csv}...")
    df.to_csv(output_csv, index=False)
    
    print("\nIngestion Complete.")
    print("Class Distribution:")
    print(df['label'].value_counts())

if __name__ == "__main__":
    ingest_keen_pad()
