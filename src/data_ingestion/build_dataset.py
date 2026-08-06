import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

def build_augmented_dataset():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    obs_path = os.path.join(project_root, 'Cl_Imec98_12.xlsx')
    bathy_path = os.path.join(project_root, 'data/raw/etopo_bathymetry.nc')
    output_path = os.path.join(project_root, 'data/processed/imecocal_augmented.csv')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Cargando dataset original: {obs_path}")
    df = pd.read_excel(obs_path)
    df['Fecha'] = pd.to_datetime(df[['Año', 'mes', 'Dia']].rename(columns={'Año': 'year', 'mes': 'month', 'Dia': 'day'}))
    
    # Limpiar NaNs
    df = df.dropna(subset=['Clorofila']).copy()
    
    # Inicializar columnas nuevas
    df['uo'] = 0.0
    df['vo'] = 0.0
    df['bathy'] = 0.0
    
    years = sorted(df['Año'].unique())
    
    print(f"Procesando extracciones CMEMS año por año...")
    for y in years:
        cmems_file = os.path.join(project_root, f'data/raw/cmems_yearly/cmems_currents_{int(y)}.nc')
        
        mask = df['Año'] == y
        if not mask.any() or not os.path.exists(cmems_file):
            print(f"  [Aviso] Saltando el año {y} (sin datos o sin archivo CMEMS).")
            continue
            
        print(f"  Procesando Año {y} con archivo {cmems_file}...")
        ds_year = xr.open_dataset(cmems_file)
        
        df_year = df[mask]
        
        # Iterar por filas para extraer de forma segura y sin OOM
        u_list, v_list = [], []
        for _, row in tqdm(df_year.iterrows(), total=len(df_year), desc=f"Extrayendo puntos {y}"):
            try:
                # nearest extraction
                val = ds_year.sel(
                    longitude=row['Longitud'],
                    latitude=row['Latitud'],
                    depth=row['Profundidad'],
                    time=row['Fecha'],
                    method='nearest'
                )
                u_list.append(float(val['uo'].values))
                v_list.append(float(val['vo'].values))
            except Exception as e:
                u_list.append(0.0)
                v_list.append(0.0)
                
        df.loc[mask, 'uo'] = u_list
        df.loc[mask, 'vo'] = v_list
        ds_year.close()
        
    print("Extrayendo Batimetría...")
    if os.path.exists(bathy_path):
        ds_bathy = xr.open_dataset(bathy_path)
        var_name = 'altitude' if 'altitude' in ds_bathy else 'elevation'
        
        b_list = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Batimetría"):
            try:
                val = ds_bathy.sel(
                    longitude=row['Longitud'],
                    latitude=row['Latitud'],
                    method='nearest'
                )
                b_list.append(float(val[var_name].values))
            except:
                b_list.append(0.0)
        df['bathy'] = b_list
        ds_bathy.close()
    
    df.to_csv(output_path, index=False)
    print(f"\\n¡Dataset aumentado guardado exitosamente en: {output_path}!")

if __name__ == "__main__":
    build_augmented_dataset()
