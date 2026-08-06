import urllib.request
import xarray as xr

# min_lat=29, max_lat=33, min_lon=-120, max_lon=-114
# the dataset id could be etopo180
url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc?altitude[(29):1:(33)][(-120):1:(-114)]"
try:
    urllib.request.urlretrieve(url, "bathy.nc")
    ds = xr.open_dataset("bathy.nc")
    print(ds)
except Exception as e:
    print(f"Error with etopo180: {e}")

url2 = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/USGS_Seafloor_Topography.nc?topo[(29):1:(33)][(-120):1:(-114)]"
try:
    urllib.request.urlretrieve(url2, "bathy2.nc")
    ds = xr.open_dataset("bathy2.nc")
    print(ds)
except Exception as e:
    print(f"Error with USGS: {e}")
