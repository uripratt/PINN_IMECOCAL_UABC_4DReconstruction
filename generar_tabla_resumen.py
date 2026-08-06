import pandas as pd
import numpy as np

# Cargar datos
df = pd.read_excel('Cl_Imec98_12.xlsx')

# Crear IDs unicos
df['Profile_ID'] = df['Linea'].astype(str) + '_' + df['Estacion'].astype(str) + '_' + df['Año'].astype(str) + '_' + df['mes'].astype(str) + '_' + df['Dia'].astype(str)
df['Station_ID'] = df['Linea'].astype(str) + '_' + df['Estacion'].astype(str)

# Quitar NAs en clorofila y profundidad para no tener errores
df = df.dropna(subset=['Clorofila', 'Profundidad'])

resultados = []

for (st, year), group in df.groupby(['Station_ID', 'Año']):
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

res_df = pd.DataFrame(resultados)
res_df.to_csv('Resumen_Estaciones_Años_IMECOCAL.csv', index=False)
res_df.to_excel('Resumen_Estaciones_Años_IMECOCAL.xlsx', index=False)

print("Tabla generada correctamente.")
print(res_df.head(10).to_string())
