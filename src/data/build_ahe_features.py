import os
import rasterio
from rasterio.features import rasterize
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def get_mask(lst_path, s2_path):
    """Recreates the exact valid pixel mask used in build_features.py"""
    with rasterio.open(lst_path) as src:
        lst = src.read(1).flatten()
    with rasterio.open(s2_path) as src:
        s2_0 = src.read(1).flatten()
        s2_1 = src.read(2).flatten()
    
    df_temp = pd.DataFrame({'a': lst, 'b': s2_0, 'c': s2_1})
    return df_temp.notna().all(axis=1)

def main():
    print("--- Starting Day 3 AHE Feature Engineering ---")
    
    # Configure OSMnx for large queries
    ox.settings.timeout = 1800  # Allow 30 minutes for large Overpass queries
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    
    lst_path = os.path.join(raw_dir, 'LST_Kelvin.tif')
    s2_path = os.path.join(raw_dir, 'Sentinel2_Features.tif')
    csv_path = os.path.join(processed_dir, 'delhi_thermal_features.csv')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Processed CSV not found. Run build_features.py first.")

    # Get raster profile
    print("Reading raster profile for spatial alignment...")
    with rasterio.open(lst_path) as src:
        transform = src.transform
        width = src.width
        height = src.height
        crs = src.crs

    # Delhi-NCR bounding box from gee_extraction.py
    west, south = 77.00, 28.40
    east, north = 77.35, 28.75
    
    # Robust OSMnx querying to handle different version signatures
    print("1. Downloading building footprints from OSM (BAH)...")
    print("   Note: This query spans Delhi-NCR and may take 5-15 minutes to download.")
    try:
        try:
            # OSMnx >= 2.0
            buildings = ox.features_from_bbox(bbox=(north, south, east, west), tags={'building': True})
        except TypeError:
            try:
                # OSMnx >= 1.5
                buildings = ox.features_from_bbox(north, south, east, west, tags={'building': True})
            except AttributeError:
                # OSMnx < 1.5
                buildings = ox.geometries_from_bbox(north, south, east, west, tags={'building': True})
                
        print("   Projecting and rasterizing buildings...")
        buildings = buildings.to_crs(crs)
        building_shapes = [(geom, 1) for geom in buildings.geometry if geom.is_valid and not geom.is_empty]
        
        if building_shapes:
            # all_touched=True ensures any pixel touched by a building footprint gets marked
            bah_raster = rasterize(building_shapes, out_shape=(height, width), transform=transform, fill=0, dtype='float32', all_touched=True)
        else:
            bah_raster = np.zeros((height, width), dtype='float32')
    except Exception as e:
        print(f"   Failed to download or process buildings. Error: {e}")
        bah_raster = np.zeros((height, width), dtype='float32')

    print("2. Downloading road network from OSM (TAH)...")
    try:
        try:
            # OSMnx >= 2.0
            G = ox.graph_from_bbox(bbox=(north, south, east, west), network_type='drive')
        except TypeError:
            # OSMnx < 2.0
            G = ox.graph_from_bbox(north, south, east, west, network_type='drive')
            
        roads = ox.graph_to_gdfs(G, nodes=False, edges=True)
        roads = roads.to_crs(crs)
        
        print("   Buffering roads and rasterizing...")
        # Buffer roads by 5 meters to simulate typical road footprint width
        roads['geometry'] = roads.geometry.buffer(5)
        road_shapes = [(geom, 1) for geom in roads.geometry if geom.is_valid and not geom.is_empty]
        
        if road_shapes:
            tah_raster = rasterize(road_shapes, out_shape=(height, width), transform=transform, fill=0, dtype='float32', all_touched=True)
        else:
            tah_raster = np.zeros((height, width), dtype='float32')
    except Exception as e:
        print(f"   Failed to process road network. Error: {e}")
        tah_raster = np.zeros((height, width), dtype='float32')

    # Flatten the newly generated rasters
    bah_flat = bah_raster.flatten()
    tah_flat = tah_raster.flatten()
    
    print("3. Aligning AHE features with the thermal feature matrix...")
    mask = get_mask(lst_path, s2_path)
    bah_valid = bah_flat[mask]
    tah_valid = tah_flat[mask]
    
    df = pd.read_csv(csv_path)
    
    if len(df) != len(bah_valid):
        print(f"Warning: Length mismatch! CSV has {len(df)} rows, but mask produced {len(bah_valid)} valid pixels.")
    
    # Overwrite the placeholder 0.0 values with our new normalized raster metrics
    df['BAH'] = bah_valid
    df['TAH'] = tah_valid
    
    df.to_csv(csv_path, index=False)
    print(f"Success! Updated {csv_path} with calculated Building (BAH) and Transportation (TAH) features.")
    print("Ready to retrain the LightGBM model.")

if __name__ == "__main__":
    main()
