import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import contextily as cx
import xarray as xr

# Load data
file_path = "Cl_Imec98_12.xlsx"
df = pd.read_excel(file_path)
surface_df = df[df['Profundidad'] <= 10]
agg_df = surface_df.groupby(['Latitud', 'Longitud'])['Clorofila'].mean().reset_index()

min_lon, max_lon = agg_df['Longitud'].min() - 0.5, agg_df['Longitud'].max() + 0.5
min_lat, max_lat = agg_df['Latitud'].min() - 0.5, agg_df['Latitud'].max() + 0.5

fig, ax = plt.subplots(figsize=(10, 10))

# Load bathymetry
ds = xr.open_dataset('https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc')
subset = ds.sel(latitude=slice(min_lat, max_lat), longitude=slice(min_lon, max_lon))

lon = subset['longitude'].values
lat = subset['latitude'].values
topo = subset['altitude'].values

# Mask land
topo_sea = np.where(topo < 0, topo, np.nan)

# Plot bathymetry
mesh = ax.pcolormesh(lon, lat, topo_sea, cmap='Blues_r', vmin=-4000, vmax=0, alpha=0.7)
cbar_bathy = plt.colorbar(mesh, ax=ax, shrink=0.5, pad=0.01)
cbar_bathy.set_label('Batimetría (m)')

# Plot stations
scatter = ax.scatter(agg_df['Longitud'], agg_df['Latitud'], 
                      c=agg_df['Clorofila'], cmap='viridis', s=50, 
                      edgecolor='black', vmin=0, vmax=2)
cbar_chl = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.05)
cbar_chl.set_label('Clorofila media (superficie) [mg/m³]')

# Add satellite basemap
cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.Esri.WorldImagery)

ax.set_xlim(min_lon, max_lon)
ax.set_ylim(min_lat, max_lat)
ax.set_title('Mapa de Estaciones y Batimetría')
ax.set_xlabel('Longitud')
ax.set_ylabel('Latitud')

plt.savefig('test_map.png', dpi=300, bbox_inches='tight')
print("test_map.png saved")
