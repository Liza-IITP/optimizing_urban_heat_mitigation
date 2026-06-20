import os
import rasterio
import numpy as np
import pandas as pd

def read_tif(file_path):
    """Reads a single-band or multi-band tif and returns a flattened array."""
    with rasterio.open(file_path) as src:
        # Read all bands
        image = src.read()
        # Flatten the arrays (Band, Height, Width) -> (Band, Pixels)
        flattened = image.reshape(image.shape[0], -1)
        return flattened

def main():
    print("--- Starting Day 2 Feature Engineering ---")
    
    # Define paths
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'raw')
    lst_path = os.path.join(raw_dir, 'LST_Kelvin.tif')
    s2_path = os.path.join(raw_dir, 'Sentinel2_Features.tif')
    
    if not os.path.exists(lst_path) or not os.path.exists(s2_path):
        raise FileNotFoundError("Raw TIF files missing. Did Day 1 extraction complete?")

    print("1. Reading raw geospatial matrices...")
    lst_data = read_tif(lst_path)[0]  # Single band (Kelvin)
    s2_data = read_tif(s2_path)       # Band 1: NDVI, Band 2: Albedo
    
    ndvi_data = s2_data[0]
    albedo_data = s2_data[1]

    # Create a DataFrame
    df = pd.DataFrame({
        'LST_Kelvin': lst_data,
        'NDVI': ndvi_data,
        'Albedo': albedo_data
    })

    # Drop nulls (pixels outside the bounding box or masked by clouds)
    df = df.dropna()
    print(f"Total valid pixels loaded: {len(df)}")

    print("2. Applying physical transformations...")
    # Convert Kelvin to Celsius
    df['LST_Celsius'] = df['LST_Kelvin'] - 273.15
    
    # Identify Cropland Baseline for RHII
    # Note: In a full pipeline, we'd overlay an LULC raster to find exact cropland pixels.
    # For now, we use a proxy: high NDVI (>0.4) and moderate Albedo (typical of crops in Delhi peripheral)
    cropland_mask = (df['NDVI'] > 0.4) & (df['Albedo'] < 0.25)
    lst_cropland_mean = df.loc[cropland_mask, 'LST_Celsius'].mean()
    print(f"Calculated Regional Cropland Baseline LST: {lst_cropland_mean:.2f} °C")

    # Calculate RHII
    df['RHII'] = df['LST_Celsius'] - lst_cropland_mean

    print("3. Initializing Anthropogenic Heat Emission (AHE) structure...")
    # These will be populated from building/traffic densities, initialized as 0 for the schema
    df['BAH'] = 0.0 # Building AHE
    df['TAH'] = 0.0 # Transportation AHE
    df['IAH'] = 0.0 # Industrial AHE
    df['MAH'] = 0.0 # Metabolic AHE

    # Save engineered features
    processed_dir = os.path.join(os.path.dirname(raw_dir), 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, 'delhi_thermal_features.csv')
    df.to_csv(out_path, index=False)
    
    print(f"Success! Engineered feature matrix saved to: {out_path}")
    print("Ready for LightGBM and SHAP analysis.")

if __name__ == "__main__":
    main()