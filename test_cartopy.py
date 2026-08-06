import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import xarray as xr

try:
    print("Loading ETOPO...")
    ds = xr.open_dataset('https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc')
    # Bounds for Ensenada roughly: Lat 29 to 33, Lon -120 to -114
    subset = ds.sel(latitude=slice(29, 33), longitude=slice(-120, -114))
    print(subset)
    print("Success loading bathymetry")
except Exception as e:
    print(f"Error: {e}")
