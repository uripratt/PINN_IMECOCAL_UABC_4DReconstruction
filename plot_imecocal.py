import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuración científica para gráficos profesionales
sns.set_theme(style="ticks", context="paper", font_scale=1.0)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['mathtext.fontset'] = 'cm'

import contextily as cx
import xarray as xr
import urllib.request
import os
from matplotlib.ticker import FormatStrFormatter
from matplotlib_scalebar.scalebar import ScaleBar

# Load data
file_path = "Cl_Imec98_12.xlsx"
df = pd.read_excel(file_path)

# Crear IDs unicos para perfiles y estaciones
df['Profile_ID'] = df['Linea'].astype(str) + '_' + df['Estacion'].astype(str) + '_' + df['Año'].astype(str) + '_' + df['mes'].astype(str) + '_' + df['Dia'].astype(str)
df['Station_ID'] = df['Linea'].astype(str) + '_' + df['Estacion'].astype(str)
df = df.groupby(['Profile_ID', 'Station_ID', 'Año', 'mes', 'Profundidad', 'Latitud', 'Longitud'], observed=True)['Clorofila'].mean().reset_index()
df = df.sort_values(by=['Profile_ID', 'Profundidad'])

# Map depths to standard depths for a clean mean profile (combining 0m and 1m, and snapping odd depths)
standard_depths = np.array([0, 10, 20, 50, 100])
def get_closest_std_depth(d):
    return standard_depths[np.abs(standard_depths - d).argmin()]

df['Profundidad_Std'] = df['Profundidad'].apply(get_closest_std_depth)
df = df[df['Profundidad_Std'] <= 100]

def get_season(month):
    if pd.isna(month): return 'Desconocido'
    m = int(month)
    if m in [3, 4, 5]: return 'Primavera'
    elif m in [6, 7, 8]: return 'Verano'
    elif m in [9, 10, 11]: return 'Otoño'
    else: return 'Invierno'

df['Temporada'] = df['mes'].apply(get_season)

# ==========================================
# Plot 1: Spatial Map of Stations
# ==========================================
print("Generando mapa de estaciones con batimetría y foto satelital...")
surface_df = df[df['Profundidad'] <= 10]
agg_df = surface_df.groupby(['Station_ID', 'Latitud', 'Longitud'])['Clorofila'].mean().reset_index()
agg_df['Linea'] = agg_df['Station_ID'].apply(lambda x: int(x.split('_')[0]))
agg_df['Estacion'] = agg_df['Station_ID'].apply(lambda x: int(x.split('_')[1]))

min_lon = agg_df['Longitud'].min() - 0.5
max_lon = agg_df['Longitud'].max() + 0.5
min_lat = agg_df['Latitud'].min() - 0.5
max_lat = agg_df['Latitud'].max() + 0.5

# Fetch ETOPO bathymetry
url = f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc?altitude[({min_lat}):1:({max_lat})][({min_lon}):1:({max_lon})]"
try:
    urllib.request.urlretrieve(url, "bathy.nc")
    ds = xr.open_dataset("bathy.nc")
    lon = ds['longitude'].values
    lat = ds['latitude'].values
    topo = ds['altitude'].values
    # Mask land (altitude > 0)
    topo_sea = np.where(topo <= 0, topo, np.nan)
    has_bathy = True
except Exception as e:
    print(f"Advertencia: No se pudo descargar la batimetría ({e})")
    has_bathy = False

fig, ax = plt.subplots(figsize=(10, 10))

if has_bathy:
    # Plot bathymetry (sea only) with isobaths every 500m
    levels = np.arange(-4500, 1, 500)
    mesh = ax.contourf(lon, lat, topo_sea, levels=levels, cmap='Blues_r', alpha=0.6, extend='min')
    contours = ax.contour(lon, lat, topo_sea, levels=levels, colors='black', linewidths=0.5, alpha=0.7)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%1.0f m')
    
    # Properly positioned horizontal colorbar for bathymetry
    cax_bathy = ax.inset_axes([0, -0.1, 1, 0.03])
    cbar_bathy = plt.colorbar(mesh, cax=cax_bathy, orientation="horizontal")
    cbar_bathy.set_label('Profundidad (m) [Fuente: Modelo ETOPO1, NOAA NCEI / CoastWatch]')

# Plot stations
scatter = ax.scatter(agg_df['Longitud'], agg_df['Latitud'], 
                      c=agg_df['Clorofila'], cmap='viridis', s=60, 
                      edgecolor='black', linewidth=0.8, vmin=0, vmax=2, zorder=5)

import matplotlib.patheffects as pe
for linea, group in agg_df.groupby('Linea'):
    costera = group.loc[group['Estacion'].idxmin()]
    lat_txt = costera['Latitud']
    lon_txt = costera['Longitud'] + 0.08 # Hacia la costa
    ax.text(lon_txt, lat_txt, f"L{int(linea)}", fontsize=10, color='white', weight='bold',
            ha='left', va='center', path_effects=[pe.withStroke(linewidth=2, foreground="black")])

# Properly positioned vertical colorbar for chlorophyll
cax_chl = ax.inset_axes([1.03, 0, 0.03, 1])
cbar_chl = plt.colorbar(scatter, cax=cax_chl)
cbar_chl.set_label('Clorofila media (superficie) [mg/m³]', fontsize=10)

# Add satellite basemap for the coast without attribution to avoid unreadable text
cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.Esri.WorldImagery, attribution=False)

ax.set_xlim(min_lon, max_lon)
ax.set_ylim(min_lat, max_lat)
ax.set_title('Mapa de Estaciones IMECOCAL con Isobáticas y Costa\nColor = Clorofila media (<= 10m)', fontsize=11)
ax.set_xlabel('Longitud', fontsize=10, labelpad=15)
ax.set_ylabel('Latitud', fontsize=10, labelpad=15)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f°'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f°'))
ax.grid(True, linestyle=':', alpha=0.6, color='white') # Professional grid

# Add scale bar
mean_lat = agg_df['Latitud'].mean()
dx_meters = np.cos(np.radians(mean_lat)) * 111320.0
scalebar = ScaleBar(dx_meters, "m", location="lower left", length_fraction=0.2,
                    box_alpha=0.7, color="black", box_color="white",
                    border_pad=0.5, font_properties={'size': 10, 'family': 'serif'})
ax.add_artist(scalebar)

# Add North Arrow (Top Right, Smaller)
ax.annotate('N', xy=(0.94, 0.95), xytext=(0.94, 0.90),
            xycoords='axes fraction', ha='center', va='center',
            fontsize=12, weight='bold', color='white',
            arrowprops=dict(facecolor='white', edgecolor='black', width=2.5, headwidth=8),
            path_effects=[pe.withStroke(linewidth=2, foreground="black")])

plt.savefig('mapa_estaciones.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# Plot 2: Vertical Profiles of Chlorophyll per Year
# ==========================================
print("Generando perfiles anuales de clorofila (todas las líneas + media)...")
output_folder = "perfiles_clorofila_anuales"
os.makedirs(output_folder, exist_ok=True)

# Loop through each year
for year in df['Año'].dropna().unique():
    year_df = df[df['Año'] == year].copy()
    # Force month to be categorical so every month appears in the legend
    year_df['Mes'] = pd.Categorical(year_df['mes'].astype(int), categories=sorted(year_df['mes'].dropna().astype(int).unique()), ordered=True)
    
    plt.figure(figsize=(6, 8))
    
    # Todas las lineas individuales coloreadas por mes
    sns.lineplot(data=year_df, x='Clorofila', y='Profundidad', hue='Mes', units='Profile_ID', estimator=None, palette='turbo', alpha=0.4, linewidth=0.8, sort=False)
    
    # Media y desviacion estandar (usando profundidades estándar)
    mean_stats = year_df.groupby('Profundidad_Std')['Clorofila'].agg(['mean', 'std']).reset_index()
    plt.plot(mean_stats['mean'], mean_stats['Profundidad_Std'], color='black', linewidth=3, label='Media')
    plt.fill_betweenx(mean_stats['Profundidad_Std'], 
                      mean_stats['mean'] - mean_stats['std'], 
                      mean_stats['mean'] + mean_stats['std'], 
                      color='gray', alpha=0.3, label='±1 Std Dev', zorder=1)
        
    plt.gca().invert_yaxis()
    n_prof = year_df['Profile_ID'].nunique()
    plt.title(f'Perfil Vertical Año {int(year)} | N = {n_prof} perfiles', fontsize=11)
    plt.xlabel('Clorofila [mg/m³]', fontsize=10)
    plt.ylabel('Profundidad [m]', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Leyenda en la esquina inferior derecha (lineas más gruesas)
    handles, labels = plt.gca().get_legend_handles_labels()
    leg = plt.legend(loc='lower right', frameon=True, edgecolor='black', fancybox=False, framealpha=0.9, fontsize='small')
    for line in leg.get_lines():
        line.set_linewidth(1.5)
        line.set_alpha(1.0)
    
    plt.savefig(f'{output_folder}/perfil_clorofila_{int(year)}.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# Plot 3: Vertical Profiles of Chlorophyll per Station
# ==========================================
print("Generando perfiles por estacion (todas las líneas + media)...")
output_folder_stations = "perfiles_clorofila_estaciones"
os.makedirs(output_folder_stations, exist_ok=True)

for st in df['Station_ID'].dropna().unique():
    st_df = df[df['Station_ID'] == st].copy()
    
    if len(st_df) < 2:
        continue
        
    # Force year to be categorical so every year appears in the legend
    st_df['Año_Cat'] = pd.Categorical(st_df['Año'].astype(int), categories=sorted(st_df['Año'].dropna().astype(int).unique()), ordered=True)
        
    plt.figure(figsize=(6, 8))
    
    # Todas las lineas individuales coloreadas por año
    sns.lineplot(data=st_df, x='Clorofila', y='Profundidad', hue='Año_Cat', units='Profile_ID', estimator=None, palette='viridis', alpha=0.5, linewidth=0.8, sort=False)
    
    # Media y desviacion estandar (usando profundidades estándar)
    mean_stats = st_df.groupby('Profundidad_Std')['Clorofila'].agg(['mean', 'std']).reset_index()
    plt.plot(mean_stats['mean'], mean_stats['Profundidad_Std'], color='black', linewidth=3, label='Media')
    plt.fill_betweenx(mean_stats['Profundidad_Std'], 
                      mean_stats['mean'] - mean_stats['std'], 
                      mean_stats['mean'] + mean_stats['std'], 
                      color='gray', alpha=0.3, label='±1 Std Dev', zorder=1)
    
    plt.gca().invert_yaxis()
    
    # Titulo con coordenadas y N perfiles
    lat_val = st_df['Latitud'].iloc[0]
    lon_val = st_df['Longitud'].iloc[0]
    n_prof = st_df['Profile_ID'].nunique()
    plt.title(f'Estación {st} | Lat: {lat_val:.3f}°, Lon: {lon_val:.3f}°\nN = {n_prof} perfiles', fontsize=11)
    
    plt.xlabel('Clorofila [mg/m³]', fontsize=10)
    plt.ylabel('Profundidad [m]', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Leyenda en la esquina inferior derecha (lineas más gruesas)
    handles, labels = plt.gca().get_legend_handles_labels()
    leg = plt.legend(loc='lower right', frameon=True, edgecolor='black', fancybox=False, framealpha=0.9, fontsize='small')
    for line in leg.get_lines():
        line.set_linewidth(1.5)
        line.set_alpha(1.0)
    
    plt.savefig(f'{output_folder_stations}/perfil_clorofila_{st}.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# Plot 4: Vertical Profiles of Chlorophyll per Season
# ==========================================
print("Generando perfiles por temporada del año (Primavera, Verano, Otoño, Invierno)...")
output_folder_seasons = "perfiles_clorofila_temporadas"
os.makedirs(output_folder_seasons, exist_ok=True)

temporadas_orden = ['Primavera', 'Verano', 'Otoño', 'Invierno']

for temp in temporadas_orden:
    temp_df = df[df['Temporada'] == temp].copy()
    
    if len(temp_df) < 2:
        continue
        
    temp_df['Año_Cat'] = pd.Categorical(temp_df['Año'].astype(int), categories=sorted(temp_df['Año'].dropna().astype(int).unique()), ordered=True)
        
    plt.figure(figsize=(6, 8))
    
    sns.lineplot(data=temp_df, x='Clorofila', y='Profundidad', hue='Año_Cat', units='Profile_ID', estimator=None, palette='viridis', alpha=0.3, linewidth=0.5, sort=False)
    
    mean_stats = temp_df.groupby('Profundidad_Std')['Clorofila'].agg(['mean', 'std']).reset_index()
    plt.plot(mean_stats['mean'], mean_stats['Profundidad_Std'], color='black', linewidth=3, label='Media')
    plt.fill_betweenx(mean_stats['Profundidad_Std'], 
                      mean_stats['mean'] - mean_stats['std'], 
                      mean_stats['mean'] + mean_stats['std'], 
                      color='gray', alpha=0.3, label='±1 Std Dev', zorder=1)
    
    plt.gca().invert_yaxis()
    
    n_prof = temp_df['Profile_ID'].nunique()
    plt.title(f'Perfil Vertical: {temp} | N = {n_prof} perfiles', fontsize=11)
    plt.xlabel('Clorofila [mg/m³]', fontsize=10)
    plt.ylabel('Profundidad [m]', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Leyenda en la esquina inferior derecha (lineas más gruesas)
    handles, labels = plt.gca().get_legend_handles_labels()
    leg = plt.legend(loc='lower right', frameon=True, edgecolor='black', fancybox=False, framealpha=0.9, fontsize='small')
    for line in leg.get_lines():
        line.set_linewidth(1.5)
        line.set_alpha(1.0)
    
    plt.savefig(f'{output_folder_seasons}/perfil_clorofila_{temp}.png', dpi=300, bbox_inches='tight')
    plt.close()

print(f"Gráficos generados: mapa_estaciones.png, '{output_folder}/', '{output_folder_stations}/' y '{output_folder_seasons}/'")
