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

@app.post("/compute-average-bands/")
async def compute_average_bands(area_of_interest: UploadFile = File(...), date_range: str = Form(...)):
    geojson_data = await area_of_interest.read()
    area_of_interest_data = json.loads(geojson_data.decode("utf-8"))
    
    output_files = await process_area_of_interest(area_of_interest_data, date_range)
    output_dir = "output_mean_sentinel_bands"
    zip_path = zip_output_files(output_files, output_dir)
    
    return FileResponse(path=zip_path, filename=os.path.basename(zip_path), media_type='application/octet-stream')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
