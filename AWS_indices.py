from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import uvicorn
import json
from pystac_client import Client
from odc.stac import load
import odc.geo
import xarray as xr
import os
from zipfile import ZipFile
import rasterio
from rasterio.transform import from_bounds

app = FastAPI()

async def process_area_of_interest(area_of_interest, date_range):
    client = Client.open("https://earth-search.aws.element84.com/v1")
    collection = "sentinel-2-l2a"
    
    if area_of_interest.get('type') == 'FeatureCollection':
        geometry = area_of_interest['features'][0]['geometry']
    elif area_of_interest.get('type') == 'Feature':
        geometry = area_of_interest['geometry']
    else:
        geometry = area_of_interest

    output_files = []
    output_dir = "output_mean_sentinel_bands"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        search = client.search(collections=[collection], 
                               intersects=geometry, 
                               datetime=date_range, 
                               query={"eo:cloud_cover": {"lt": 20}})
        items = list(search.items())
        if len(items) == 0:
            raise ValueError("No data found for the given parameters.")
        
        bands = ['coastal', 'blue', 'green', 'red', 'rededge1', 'rededge2', 'rededge3', 'nir', 'nir08', 'nir09', 'wvp', 'swir16', 'swir22']
        data = load(items, 
                    measurements=bands, 
                    geopolygon=odc.geo.Geometry(geometry, crs="epsg:4326"), 
                    groupby="solar_day", 
                    chunks={})
        
        mean_data = data.mean(dim='time', keep_attrs=True).compute()
        
        for band in mean_data.data_vars:
            output_file = os.path.join(output_dir, f'{band}_mean.tiff')
            odc.geo.xr.write_cog(mean_data[band], fname=output_file, overwrite=True)
            output_files.append(output_file)
            
        return output_files
    except Exception as e:
        print(f"Error processing area of interest: {e}")
        raise

def zip_output_files(output_files, output_dir):
    zip_path = os.path.join(output_dir, "mean_sentinel_bands.zip")
    with ZipFile(zip_path, 'w') as zipf:
        for file in output_files:
            zipf.write(file, os.path.basename(file))
    return zip_path
def read_band(band_path):
    """Read a band data from a GeoTIFF file."""
    with rasterio.open(band_path) as src:
        return src.read(1)  # Read the first band
def write_index(output_path, data, profile):
    """
    Write index data to a GeoTIFF file.

    Parameters:
    - output_path: The path to the output GeoTIFF file.
    - data: The index data as a NumPy array to write to the file.
    - profile: The profile information from the source raster file.
    """
    # Update the profile to reflect the data's shape and dtype
    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw'
    )

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data, 1)
def calculate_ndvi(red_path, nir_path, output_path):
    """Calculate NDVI and save as a GeoTIFF."""
    # Read the red and NIR bands
    red = read_band(red_path)
    nir = read_band(nir_path)
    
    # Calculate NDVI
    ndvi = (nir.astype(float) - red.astype(float)) / (nir + red)
    
    # Use the profile from one of the bands as a template
    with rasterio.open(red_path) as src:
        profile = src.profile

    # Update the data type and no. of bands in the profile
    profile.update(dtype=rasterio.float32, count=1)
    
    # Write the NDVI to a new GeoTIFF file
    write_index(output_path, ndvi.astype(rasterio.float32), profile)
def calculate_evi(blue_path, red_path, nir_path, output_path, G=2.5, C1=6, C2=7.5, L=1):
    blue = read_band(blue_path)
    red = read_band(red_path)
    nir = read_band(nir_path)
    
    evi = G * (nir - red) / (nir + C1 * red - C2 * blue + L)
    
    with rasterio.open(red_path) as src:
        profile = src.profile
    profile.update(dtype=rasterio.float32, count=1)
    write_index(output_path, evi.astype(rasterio.float32), profile)

def calculate_savi(red_path, nir_path, output_path, L=0.5):
    red = read_band(red_path)
    nir = read_band(nir_path)
    
    savi = ((nir - red) / (nir + red + L)) * (1 + L)
    
    with rasterio.open(red_path) as src:
        profile = src.profile
    profile.update(dtype=rasterio.float32, count=1)
    write_index(output_path, savi.astype(rasterio.float32), profile)

@app.post("/compute-indices/")
async def compute_indices(area_of_interest: UploadFile = File(...), date_range: str = Form(...)):
    geojson_data = await area_of_interest.read()
    area_of_interest_data = json.loads(geojson_data.decode("utf-8"))
    
    # Assume the process_area_of_interest function saves all required band images and returns their paths
    output_files = await process_area_of_interest(area_of_interest_data, date_range)
    output_dir = "output_mean_sentinel_bands"
    
    # Compute NDVI, EVI, SAVI (extend this for NDWI, NDBI, NDMI)
    ndvi_output_path = os.path.join(output_dir, "ndvi.tiff")
    calculate_ndvi(os.path.join(output_dir, "red_mean.tiff"), os.path.join(output_dir, "nir_mean.tiff"), ndvi_output_path)
    
    evi_output_path = os.path.join(output_dir, "evi.tiff")
    calculate_evi(os.path.join(output_dir, "blue_mean.tiff"), os.path.join(output_dir, "red_mean.tiff"), os.path.join(output_dir, "nir_mean.tiff"), evi_output_path)
    
    savi_output_path = os.path.join(output_dir, "savi.tiff")
    calculate_savi(os.path.join(output_dir, "red_mean.tiff"), os.path.join(output_dir, "nir_mean.tiff"), savi_output_path)
    
    zip_path = zip_output_files(output_files + [ndvi_output_path, evi_output_path, savi_output_path], output_dir)
    
    return FileResponse(path=zip_path, filename=os.path.basename(zip_path), media_type='application/octet-stream')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
