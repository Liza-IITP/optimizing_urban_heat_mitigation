import ee
import os
import requests
def authenticate_gee():
    print("--- Google Earth Engine Authentication ---")
    project_id = "mediq-app-a3a52"
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)
    print("Earth Engine Initialized Successfully!")

def get_landsat_lst(aoi, start_date, end_date):
    """
    Fetch Landsat 8 Collection 2 Level 2 Surface Temperature.
    Applies scaling factors to convert to Kelvin.
    """
    collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUD_COVER', 10))

    # Function to apply scaling factors for LST (ST_B10)
    # The output will be directly in Kelvin
    def apply_scale_factors(image):
        # ST_B10 scale is 0.00341802 and offset is 149.0
        # This yields Kelvin directly.
        lst_kelvin = image.select('ST_B10').multiply(0.00341802).add(149.0).rename('LST_Kelvin')
        return image.addBands(lst_kelvin)

    scaled_collection = collection.map(apply_scale_factors)
    # Take median over the period to get a clean cloud-free image
    median_image = scaled_collection.select('LST_Kelvin').median()
    return median_image.clip(aoi)

def get_sentinel2_features(aoi, start_date, end_date):
    """
    Fetch Sentinel-2 Harmonized Surface Reflectance to compute
    NDVI and the 5-band Liang (2001) Albedo.
    """
    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))

    def compute_indices(image):
        # Scale factor for Sentinel-2 SR
        img_scaled = image.select(['B2', 'B4', 'B8', 'B11', 'B12']).multiply(0.0001)

        # NDVI: (NIR - RED) / (NIR + RED)
        ndvi = img_scaled.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        # Albedo: Liang (2001) 5-band formula
        # α = 0.356·ρ_blue + 0.130·ρ_red + 0.373·ρ_nir + 0.085·ρ_swir1 + 0.072·ρ_swir2 - 0.0018
        albedo = img_scaled.expression(
            '0.356 * B2 + 0.130 * B4 + 0.373 * B8 + 0.085 * B11 + 0.072 * B12 - 0.0018',
            {
                'B2': img_scaled.select('B2'),
                'B4': img_scaled.select('B4'),
                'B8': img_scaled.select('B8'),
                'B11': img_scaled.select('B11'),
                'B12': img_scaled.select('B12')
            }
        ).rename('Albedo')
        
        return image.addBands([ndvi, albedo])

    processed_collection = collection.map(compute_indices)
    median_image = processed_collection.select(['NDVI', 'Albedo']).median()
    return median_image.clip(aoi)

def download_image(image, aoi, scale, filename, output_dir):
    """
    Generate download URL and save the GeoTIFF locally.
    """
    print(f"Generating download URL for {filename}...")
    try:
        url = image.getDownloadURL({
            'scale': scale,
            'crs': 'EPSG:32643', # UTM Zone 43N (Delhi)
            'region': aoi,
            'format': 'GEO_TIFF'
        })
        print(f"Downloading from: {url}")
        response = requests.get(url)
        response.raise_for_status()
        
        # Save the content directly as a .tif file (no unzipping needed)
        file_path = os.path.join(output_dir, f"{filename}.tif")
        with open(file_path, 'wb') as fd:
            fd.write(response.content)
            
        print(f"Successfully downloaded to {file_path}")
        
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
def main():
    authenticate_gee()
    
    # Target AOI: Delhi-NCR
    min_lon, min_lat = 77.00, 28.40
    max_lon, max_lat = 77.35, 28.75
    aoi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
    
    # Time window: targeting peak summer (May 2023)
    start_date = '2023-05-01'
    end_date = '2023-05-31'
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Fetching Landsat 8 LST for Delhi-NCR from {start_date} to {end_date}...")
    lst_image = get_landsat_lst(aoi, start_date, end_date)
    
    print("Fetching Sentinel-2 NDVI and Albedo...")
    s2_image = get_sentinel2_features(aoi, start_date, end_date)
    
    # Download with appropriate scale (30m to align to Landsat)
    download_image(lst_image, aoi, scale=30, filename="LST_Kelvin", output_dir=output_dir)
    download_image(s2_image, aoi, scale=30, filename="Sentinel2_Features", output_dir=output_dir)
    
    print("Day 1 GEE Extraction Complete!")

if __name__ == "__main__":
    main()
