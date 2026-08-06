import pandas as pd
import numpy as np

# Cargar datos
df = pd.read_excel('Cl_Imec98_12.xlsx')

# Crear IDs unicos
df['Profile_ID'] = df['Linea'].astype(str) + '_' + df['Estacion'].astype(str) + '_' + df['Año'].astype(str) + '_' + df['mes'].astype(str) + '_' + df['Dia'].astype(str)
df['Station_ID'] = df['Linea'].astype(str) + '_' + df['Estacion'].astype(str)

df = df.dropna(subset=['Clorofila', 'Profundidad'])

estaciones_unicas = sorted(df['Station_ID'].unique())
años_unicos = sorted(df['Año'].unique())

# Crear un MultiIndex con TODAS las combinaciones posibles
idx = pd.MultiIndex.from_product([estaciones_unicas, años_unicos], names=['Estacion', 'Año'])

# Agrupar los datos reales
grupos = df.groupby(['Station_ID', 'Año'])

resultados = []
for st, year in idx:
    if (st, year) in grupos.groups:
        group = grupos.get_group((st, year))
        num_perfiles = group['Profile_ID'].nunique()
        
        profs = np.sort(group['Profundidad'].unique())
        profs_str = ", ".join([f"{p:g}" for p in profs])
        
        max_idx = group['Clorofila'].idxmax()
        max_chl = group.loc[max_idx, 'Clorofila']
        depth_max_chl = group.loc[max_idx, 'Profundidad']
        
        resultados.append({
            'Estacion': st,
            'Año': int(year),
            'Num_Perfiles': num_perfiles,
            'Max_Clorofila_mg_m3': round(max_chl, 3),
            'Profundidad_del_Maximo_m': depth_max_chl,
            'Profundidades_Muestreadas_m': profs_str
        })
    else:
        # Aquí está la clave para mostrar la realidad de los gaps
        resultados.append({
            'Estacion': st,
            'Año': int(year),
            'Num_Perfiles': 0,
            'Max_Clorofila_mg_m3': np.nan,
            'Profundidad_del_Maximo_m': np.nan,
            'Profundidades_Muestreadas_m': "Sin datos"
        })

res_df = pd.DataFrame(resultados)

# 1. Guardar la lista larga completa con Gaps
res_df.to_csv('Cobertura_Completa_IMECOCAL.csv', index=False)
res_df.to_excel('Cobertura_Completa_IMECOCAL.xlsx', index=False)

# 2. Generar una Matriz (Pivot Table) visual de Esfuerzo de Muestreo (Num Perfiles)
# Para que de un solo vistazo el usuario vea la "realidad" del dataset
matriz_esfuerzo = res_df.pivot(index='Estacion', columns='Año', values='Num_Perfiles').fillna(0).astype(int)
matriz_esfuerzo.to_excel('Matriz_Esfuerzo_Muestreo.xlsx')

print("Generación completada.")
