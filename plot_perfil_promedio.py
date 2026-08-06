import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

# Cargar datos
df = pd.read_excel('Cl_Imec98_12.xlsx')

# Crear IDs unicos
df['Profile_ID'] = df['Linea'].astype(str) + '_' + df['Estacion'].astype(str) + '_' + df['Año'].astype(str) + '_' + df['mes'].astype(str) + '_' + df['Dia'].astype(str)

# Promediar clorofila repetida por perfil e interpolar a profundidades estándar
df = df.groupby(['Profile_ID', 'Profundidad'], observed=True)['Clorofila'].mean().reset_index()

standard_depths = np.array([0, 10, 20, 50, 100])
def get_closest_std_depth(d):
    return standard_depths[np.abs(standard_depths - d).argmin()]

df['Profundidad_Std'] = df['Profundidad'].apply(get_closest_std_depth)
df = df[df['Profundidad_Std'] <= 100]

# Agrupar a nivel de perfil en cada profundidad estándar
agg_df = df.groupby(['Profile_ID', 'Profundidad_Std'])['Clorofila'].mean().reset_index()

# Calcular estadísticas climatológicas globales (todos los perfiles, todos los años)
stats = agg_df.groupby('Profundidad_Std')['Clorofila'].agg(['mean', 'std', 'median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)]).reset_index()
stats.columns = ['Profundidad_Std', 'mean', 'std', 'median', 'q25', 'q75']

# Crear figura
plt.figure(figsize=(5, 7))

# Rellenar desviación estándar
plt.fill_betweenx(stats['Profundidad_Std'], 
                  stats['mean'] - stats['std'], 
                  stats['mean'] + stats['std'], 
                  color='lightblue', alpha=0.4, label='±1 Desv. Est.')

# Rellenar rango intercuartil (Q25 - Q75) para mayor rigor estadístico
plt.fill_betweenx(stats['Profundidad_Std'], 
                  stats['q25'], 
                  stats['q75'], 
                  color='blue', alpha=0.15, label='Rango Intercuartil (25-75%)')

# Dibujar mediana y media
plt.plot(stats['mean'], stats['Profundidad_Std'], color='navy', linewidth=2.5, marker='o', label='Media Climatológica')
plt.plot(stats['median'], stats['Profundidad_Std'], color='darkorange', linewidth=1.8, linestyle='--', marker='s', label='Mediana')

# Ajustar ejes
plt.ylim(100, 0) # Invertir eje Y
plt.xlim(0, 8)    # Ajustar límite de clorofila para apreciar los detalles de la capa de mezcla
plt.xlabel('Concentración de Clorofila-a [mg/m³]', fontsize=11)
plt.ylabel('Profundidad [m]', fontsize=11)
plt.title('Perfil Vertical Climatológico de Clorofila-a\nIMECOCAL Histórico (1998-2012)', fontsize=12, weight='bold')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right', frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig('perfil_clorofila_promedio.png', dpi=300)
plt.close()

print("Gráfico de perfil climatológico promedio generado con éxito.")
