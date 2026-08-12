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
    df['wo'] = 0.0
    df['thetao'] = 15.0 # Dummy inicial
    df['CHL_sat'] = 0.0
    df['bathy'] = 0.0
    
    years = sorted(df['Año'].unique())
    
    print(f"Procesando extracciones CMEMS año por año...")
    for y in years:
        cmems_file = os.path.join(project_root, f'data/raw/cmems_yearly/cmems_currents_{int(y)}_with_w.nc')
        thetao_file = os.path.join(project_root, f'data/raw/cmems_yearly_thetao/cmems_thetao_{int(y)}.nc')
        chl_sat_file = os.path.join(project_root, f'data/raw/satellite_chl_yearly/satellite_chl_{int(y)}.nc')
        
        mask = df['Año'] == y
        if not mask.any() or not os.path.exists(cmems_file):
            print(f"  [Aviso] Saltando el año {y} (sin datos o sin archivo CMEMS _with_w).")
            continue
            
        print(f"  Procesando Año {y}...")
        ds_year = xr.open_dataset(cmems_file)
        
        # Abrir los otros datasets si existen
        ds_thetao = xr.open_dataset(thetao_file) if os.path.exists(thetao_file) else None
        ds_chl = xr.open_dataset(chl_sat_file) if os.path.exists(chl_sat_file) else None
        
        df_year = df[mask]
        
        # Iterar por filas para extraer de forma segura y sin OOM
        u_list, v_list, w_list, t_list, chl_list = [], [], [], [], []
        for _, row in tqdm(df_year.iterrows(), total=len(df_year), desc=f"Extrayendo puntos {y}"):
            # 1. Corrientes 3D (u, v, w)
            try:
                val = ds_year.sel(
                    longitude=row['Longitud'],
                    latitude=row['Latitud'],
                    depth=row['Profundidad'],
                    time=row['Fecha'],
                    method='nearest'
                )
                u_list.append(float(val['uo'].values))
                v_list.append(float(val['vo'].values))
                w_list.append(float(val['wo'].values))
            except Exception:
                u_list.append(0.0)
                v_list.append(0.0)
                w_list.append(0.0)
                
            # 2. Temperatura (thetao)
            try:
                if ds_thetao is not None:
                    val_t = ds_thetao.sel(
                        longitude=row['Longitud'],
                        latitude=row['Latitud'],
                        depth=row['Profundidad'],
                        time=row['Fecha'],
                        method='nearest'
                    )
                    t_list.append(float(val_t['thetao'].values))
                else:
                    t_list.append(15.0)
            except Exception:
                t_list.append(15.0)
                
            # 3. Clorofila Satelital (CHL, siempre en superficie)
            try:
                if ds_chl is not None:
                    val_c = ds_chl.sel(
                        longitude=row['Longitud'],
                        latitude=row['Latitud'],
                        time=row['Fecha'],
                        method='nearest'
                    )
                    chl_list.append(float(val_c['CHL'].values))
                else:
                    chl_list.append(0.0)
            except Exception:
                chl_list.append(0.0)
                
        df.loc[mask, 'uo'] = u_list
        df.loc[mask, 'vo'] = v_list
        df.loc[mask, 'wo'] = w_list
        df.loc[mask, 'thetao'] = t_list
        df.loc[mask, 'CHL_sat'] = chl_list
        
        ds_year.close()
        if ds_thetao is not None: ds_thetao.close()
        if ds_chl is not None: ds_chl.close()
        
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
