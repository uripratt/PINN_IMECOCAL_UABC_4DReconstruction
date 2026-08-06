import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata

# Configuración científica profesional
sns.set_theme(style="ticks", context="paper", font_scale=1.1)
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

# Cargar datos
df = pd.read_excel('Cl_Imec98_12.xlsx')

# Fórmula de Haversine
def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Obtener estación más costera por línea
costeras = {}
for linea in df['Linea'].unique():
    linea_df = df[df['Linea'] == linea]
    min_est = linea_df['Estacion'].min()
    est_data = linea_df[linea_df['Estacion'] == min_est].iloc[0]
    costeras[linea] = {
        'Estacion': min_est,
        'Lat': est_data['Latitud'],
        'Lon': est_data['Longitud']
    }

# Distancia a la costa
df['Distancia_Costa_km'] = [
    haversine(row['Longitud'], row['Latitud'], costeras[row['Linea']]['Lon'], costeras[row['Linea']]['Lat'])
    for idx, row in df.iterrows()
]

# Filtrar capa superficial (<= 10 m)
surf_df = df[df['Profundidad'] <= 10].copy()

# ----------------- PREPARAR DATOS PARA PANEL A (HOVMÖLLER) -----------------
# Crear bins de distancia a la costa
bin_size = 20  # km
max_dist = 280 # km
bins = np.arange(0, max_dist + bin_size, bin_size)
surf_df['Distancia_Bin'] = pd.cut(surf_df['Distancia_Costa_km'], bins=bins, include_lowest=True)
surf_df['Distancia_Bin_Center'] = surf_df['Distancia_Bin'].apply(lambda x: x.mid if not pd.isna(x) else np.nan)

# Agrupar por Año y Bin de distancia
hov_data = surf_df.groupby(['Año', 'Distancia_Bin_Center'], observed=True)['Clorofila'].mean().reset_index()

# Rejilla regular para contornos/heatmap
grid_years = sorted(hov_data['Año'].unique())
grid_dists = np.arange(bin_size/2, max_dist, bin_size)

# Pivotar datos
pivot_hov = hov_data.pivot(index='Distancia_Bin_Center', columns='Año', values='Clorofila')
# Rellenar nans mediante interpolación lineal bidireccional para suavizar el contorno
pivot_hov = pivot_hov.interpolate(method='linear', limit_direction='both', axis=0)
pivot_hov = pivot_hov.interpolate(method='linear', limit_direction='both', axis=1)

# ----------------- PREPARAR DATOS PARA PANEL B (EXTREMOS VS CLIMA) -----------------
# Promedio Climatológico (todos los años)
clima_grad = surf_df.groupby('Distancia_Bin_Center', observed=True)['Clorofila'].agg(['mean', 'std']).reset_index()

# Año El Niño Extremo (1998)
nino_grad = surf_df[surf_df['Año'] == 1998].groupby('Distancia_Bin_Center', observed=True)['Clorofila'].mean().reset_index()

# Año La Niña Fuerte (1999)
nina_grad = surf_df[surf_df['Año'] == 1999].groupby('Distancia_Bin_Center', observed=True)['Clorofila'].mean().reset_index()

# Año de transición / Neutral (e.g. 2008)
neutral_grad = surf_df[surf_df['Año'] == 2008].groupby('Distancia_Bin_Center', observed=True)['Clorofila'].mean().reset_index()

# ----------------- GRAFICAR PANEL COMBINADO -----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# --- PANEL A: Diagrama de Hovmöller (Espacio-Tiempo) ---
X, Y = np.meshgrid(pivot_hov.columns, pivot_hov.index)
Z = pivot_hov.values

# Graficar contorno relleno
contour = ax1.contourf(X, Y, Z, levels=np.linspace(0, 3.5, 15), cmap='YlGnBu', extend='max')
cbar = fig.colorbar(contour, ax=ax1, pad=0.02)
cbar.set_label('Clorofila-a superficial [mg/m³]', fontsize=10)

# Dibujar líneas de contorno de referencia
contours = ax1.contour(X, Y, Z, levels=[0.5, 1.0, 2.0], colors='black', linewidths=0.5, alpha=0.6)
ax1.clabel(contours, inline=True, fontsize=8, fmt='%1.1f')

ax1.set_xlabel('Año', fontsize=11)
ax1.set_ylabel('Distancia a la costa [km]', fontsize=11)
ax1.set_title('a) Evolución Temporal del Gradiente Costa-Océano', fontsize=12, weight='bold')
ax1.set_xticks(grid_years)
ax1.set_xticklabels([str(int(y)) for y in grid_years], rotation=45)
ax1.grid(True, linestyle=':', alpha=0.5)

# --- PANEL B: Perfiles de Gradiente Clásico (Clima vs Extremos) ---
# Climatología histórica
ax2.plot(clima_grad['Distancia_Bin_Center'], clima_grad['mean'], color='black', linewidth=2.5, label='Media Climatológica (1998-2012)')
ax2.fill_between(clima_grad['Distancia_Bin_Center'].astype(float), 
                 np.maximum(0, clima_grad['mean'] - clima_grad['std']), 
                 clima_grad['mean'] + clima_grad['std'], 
                 color='gray', alpha=0.15, label='±1 Desv. Est. Climatológica')

# Año El Niño 1998
ax2.plot(nino_grad['Distancia_Bin_Center'], nino_grad['Clorofila'], color='crimson', linewidth=2.0, marker='o', label='El Niño Extremo (1998)')

# Año La Niña 1999
ax2.plot(nina_grad['Distancia_Bin_Center'], nina_grad['Clorofila'], color='teal', linewidth=2.0, marker='^', label='La Niña Fuerte (1999)')

# Año Neutral 2008
ax2.plot(neutral_grad['Distancia_Bin_Center'], neutral_grad['Clorofila'], color='goldenrod', linewidth=1.5, linestyle='--', marker='s', label='Año Neutral (2008)')

ax2.set_xlabel('Distancia a la costa [km]', fontsize=11)
ax2.set_ylabel('Clorofila-a superficial [mg/m³]', fontsize=11)
ax2.set_title('b) Estructura Transversal en Anomalías Climáticas', fontsize=12, weight='bold')
ax2.set_xlim(0, max_dist)
ax2.set_ylim(0, 7) # Límite superior holgado para ver los picos costeros
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False, fontsize=9)

plt.tight_layout()
plt.savefig('gradiente_costa_anual.png', dpi=300)
plt.close()

print("Gráfico de gradiente costa-océano (gradiente_costa_anual.png) generado exitosamente.")
