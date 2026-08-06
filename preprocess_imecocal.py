import pandas as pd
import numpy as np

# Load data
file_path = "Cl_Imec98_12.xlsx"
df = pd.read_excel(file_path)

# Convert Año, mes, Dia to datetime
df['datetime'] = pd.to_datetime({'year': df['Año'], 'month': df['mes'], 'day': df['Dia']})
epoch = df['datetime'].min()
# Calculate time 't' in hours from the epoch (first date in the dataset)
df['t_hours'] = (df['datetime'] - epoch).dt.total_seconds() / 3600.0

profiles = []

# Group by each unique cast (unique date and station)
grouped = df.groupby(['Linea', 'Estacion', 'datetime'])

for (linea, estacion, dt), group in grouped:
    # Sort by depth
    group = group.sort_values(by='Profundidad')
    
    # Extract coordinates (assuming they are constant for the cast)
    x = group['Longitud'].iloc[0]
    y = group['Latitud'].iloc[0]
    t = group['t_hours'].iloc[0]
    
    # Extract vertical profiles
    z = group['Profundidad'].values
    chlorophyll = group['Clorofila'].values
    
    profile = {
        'x': x,
        'y': y,
        't': t,
        'datetime': str(dt),
        'linea': linea,
        'estacion': estacion,
        'z': z,
        'chlorophyll': chlorophyll
    }
    profiles.append(profile)

output_file = "imecocal_ctd_profiles.npy"
np.save(output_file, profiles, allow_pickle=True)

print(f"Preprocesamiento completado.")
print(f"Se extrajeron {len(profiles)} perfiles.")
print(f"Archivo guardado en: {output_file}")
print(f"Epoch usado para el tiempo t (horas = 0): {epoch}")
