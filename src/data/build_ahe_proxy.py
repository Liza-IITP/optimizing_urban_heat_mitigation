import os
import pandas as pd
import numpy as np

def main():
    print("--- Starting AHE Proxy Generation (Offline Mode) ---")
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(base_dir, 'data', 'processed', 'delhi_thermal_features.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError("delhi_thermal_features.csv not found.")

    print("1. Loading processed data...")
    df = pd.read_csv(data_path)

    print("2. Calculating empirical proxies for Urban Metabolism...")
    # Normalize features between 0 and 1 to create clean indices
    ndvi_norm = (df['NDVI'] - df['NDVI'].min()) / (df['NDVI'].max() - df['NDVI'].min())
    albedo_norm = (df['Albedo'] - df['Albedo'].min()) / (df['Albedo'].max() - df['Albedo'].min())

    # BAH (Building AHE) Proxy:
    # Logic: Buildings have extremely low vegetation. We invert NDVI and scale it.
    df['BAH'] = (1.0 - ndvi_norm) * 100.0  

    # TAH (Transportation AHE) Proxy:
    # Logic: Asphalt roads have ZERO vegetation and VERY LOW albedo (dark surfaces).
    df['TAH'] = ((1.0 - ndvi_norm) * (1.0 - albedo_norm)) * 80.0

    # IAH (Industrial AHE) & MAH (Metabolic AHE):
    # Logic: Industries cluster in massive concrete zones (BAH > 85). Metabolic heat scales with buildings.
    df['IAH'] = np.where(df['BAH'] > 85, np.random.uniform(20, 50, size=len(df)), 0.0)
    df['MAH'] = df['BAH'] * 0.25 

    print("3. Saving injected features...")
    df.to_csv(data_path, index=False)
    
    print(f"Success! Structural Proxies successfully injected into {data_path}")
    print("You are clear to retrain the LightGBM model!")

if __name__ == "__main__":
    main()