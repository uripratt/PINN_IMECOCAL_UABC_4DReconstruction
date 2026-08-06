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

# Cargar y preparar datos (igual que plot_imecocal)
file_path = "Cl_Imec98_12.xlsx"
df = pd.read_excel(file_path)
df['Profile_ID'] = df['Linea'].astype(str) + '_' + df['Estacion'].astype(str) + '_' + df['Año'].astype(str) + '_' + df['mes'].astype(str) + '_' + df['Dia'].astype(str)
df = df.groupby(['Profile_ID', 'Año', 'mes', 'Profundidad'], observed=True)['Clorofila'].mean().reset_index()
df = df.sort_values(by=['Profile_ID', 'Profundidad'])

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
# Grid Anual (5 filas x 3 columnas = 15 años)
# ==========================================
años = sorted(df['Año'].dropna().unique())
fig, axes = plt.subplots(5, 3, figsize=(12, 18), sharex=True, sharey=True)
axes = axes.flatten()

# Preparar una lista de colores/meses para la leyenda global
import matplotlib.lines as mlines
from matplotlib.colors import Normalize

# Asignar meses como categoricos
meses_unicos = sorted(df['mes'].dropna().astype(int).unique())
palette_anual = sns.color_palette('turbo', n_colors=len(meses_unicos))
color_dict_anual = dict(zip(meses_unicos, palette_anual))

for i, year in enumerate(años):
    if i >= len(axes): break
    ax = axes[i]
    year_df = df[df['Año'] == year].copy()
    
    if len(year_df) > 0:
        year_df['Mes'] = year_df['mes'].astype(int)
        for mes, mes_df in year_df.groupby('Mes'):
            sns.lineplot(data=mes_df, x='Clorofila', y='Profundidad', units='Profile_ID', estimator=None, 
                         color=color_dict_anual[mes], alpha=0.4, linewidth=0.8, sort=False, ax=ax)
        
        mean_stats = year_df.groupby('Profundidad_Std')['Clorofila'].agg(['mean', 'std']).reset_index()
        ax.plot(mean_stats['mean'], mean_stats['Profundidad_Std'], color='black', linewidth=3, zorder=10)
        ax.fill_betweenx(mean_stats['Profundidad_Std'], 
                          mean_stats['mean'] - mean_stats['std'], 
                          mean_stats['mean'] + mean_stats['std'], 
                          color='gray', alpha=0.3, zorder=1)
        
    ax.set_title(f'Año {int(year)}', fontsize=10)
    ax.set_ylim(100, 0)
    ax.set_xlim(0, 20)
    ax.grid(True, linestyle=':', alpha=0.6)
    if i % 3 == 0:
        ax.set_ylabel('Profundidad [m]', fontsize=9)
    if i >= 12:
        ax.set_xlabel('Clorofila [mg/m³]', fontsize=9)

# plt.gca().invert_yaxis() - ya lo hace set_ylim(100, 0)
plt.tight_layout()
fig.subplots_adjust(bottom=0.08)

# Leyenda global Anual
legend_elements = [mlines.Line2D([0], [0], color=color_dict_anual[m], lw=2, label=f'Mes {m}') for m in meses_unicos]
fig.legend(handles=legend_elements, loc='lower center', ncol=len(meses_unicos), bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=9)
plt.savefig('grid_anual.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# Grid Estacional (2 filas x 2 columnas)
# ==========================================
temporadas_orden = ['Primavera', 'Verano', 'Otoño', 'Invierno']
fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
axes = axes.flatten()

años_unicos = sorted(df['Año'].dropna().astype(int).unique())
palette_temp = sns.color_palette('viridis', n_colors=len(años_unicos))
color_dict_temp = dict(zip(años_unicos, palette_temp))

for i, temp in enumerate(temporadas_orden):
    ax = axes[i]
    temp_df = df[df['Temporada'] == temp].copy()
    
    if len(temp_df) > 0:
        temp_df['Año_int'] = temp_df['Año'].astype(int)
        for año, año_df in temp_df.groupby('Año_int'):
            sns.lineplot(data=año_df, x='Clorofila', y='Profundidad', units='Profile_ID', estimator=None, 
                         color=color_dict_temp[año], alpha=0.3, linewidth=0.5, sort=False, ax=ax)
            
        mean_stats = temp_df.groupby('Profundidad_Std')['Clorofila'].agg(['mean', 'std']).reset_index()
        ax.plot(mean_stats['mean'], mean_stats['Profundidad_Std'], color='black', linewidth=3, zorder=10)
        ax.fill_betweenx(mean_stats['Profundidad_Std'], 
                          mean_stats['mean'] - mean_stats['std'], 
                          mean_stats['mean'] + mean_stats['std'], 
                          color='gray', alpha=0.3, zorder=1)
        
    ax.set_title(f'{temp}', fontsize=12)
    ax.set_ylim(100, 0)
    ax.set_xlim(0, 20)
    ax.grid(True, linestyle=':', alpha=0.6)
    if i >= 2:
        ax.set_xlabel('Clorofila [mg/m³]', fontsize=10)
    if i % 2 == 0:
        ax.set_ylabel('Profundidad [m]', fontsize=10)

plt.tight_layout()
fig.subplots_adjust(bottom=0.15)

# Leyenda global Estacional
legend_elements_temp = [mlines.Line2D([0], [0], color=color_dict_temp[a], lw=2, label=str(a)) for a in años_unicos]
fig.legend(handles=legend_elements_temp, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=9)
plt.savefig('grid_temporadas.png', dpi=300, bbox_inches='tight')
plt.close()

print("Grids generados con éxito: grid_anual.png y grid_temporadas.png")
