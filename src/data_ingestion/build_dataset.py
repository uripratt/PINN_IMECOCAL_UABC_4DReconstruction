import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

def build_augmented_dataset():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    ctd_base_dir = os.path.join(project_root, 'data/raw/imecocal/datos imecocal txt/')
    obs_path = os.path.join(project_root, 'Cl_Imec98_12.xlsx')
    bathy_path = os.path.join(project_root, 'data/raw/etopo_bathymetry.nc')
    output_path = os.path.join(project_root, 'data/processed/imecocal_augmented.parquet')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("1. Cargando datos de botellas (Ground Truth) y metadata espacial...")
    df_bottles_raw = pd.read_excel(obs_path)
    df_bottles_raw['Fecha'] = pd.to_datetime(df_bottles_raw[['Año', 'mes', 'Dia']].rename(columns={'Año': 'year', 'mes': 'month', 'Dia': 'day'}))
    df_bottles_raw = df_bottles_raw.rename(columns={'Profundidad': 'Depth', 'Clorofila': 'Chl_Bottle'})
    
    # Mapa de estaciones a lat/lon
    station_map = df_bottles_raw[['Linea', 'Estacion', 'Latitud', 'Longitud']].drop_duplicates().groupby(['Linea', 'Estacion']).first().reset_index()
    # Mapa de fechas por crucero (Año, mes)
    date_map = df_bottles_raw[['Año', 'mes', 'Fecha']].dropna().groupby(['Año', 'mes']).first().reset_index()

    print("2. Parseando 5000+ archivos CTD (Resolución Continua)...")
    txt_files = glob.glob(os.path.join(ctd_base_dir, "**/*.txt"), recursive=True)
    # Filtro estricto dictado por el análisis de calidad
    blacklist = ['0101', '0107', '0204', '0207', '1004']
    
    ctd_chunks = []
    for f in tqdm(txt_files, desc="Leyendo CTD"):
        folder = os.path.basename(os.path.dirname(f)) # YYMM
        fname = os.path.basename(f)
        try:
            estacion_str = fname.replace('.txt', '').split('_')[-1]
            linea, est = map(float, estacion_str.split('.'))
            
            df = pd.read_csv(f, sep='\s+', encoding='latin1', on_bad_lines='skip')
            if df.shape[1] >= 5:
                df = df.iloc[:, :5]
                df.columns = ['Depth', 'Temp_CTD', 'Sal_CTD', 'O2_CTD', 'Chl_CTD']
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna(subset=['Depth'])
                df['Linea'] = linea
                df['Estacion'] = est
                
                y = int(folder[:2])
                y = y + 1900 if y > 50 else y + 2000
                df['Año'] = y
                df['mes'] = int(folder[2:])
                
                # QC de Clorofila: Anular cruceros malos y años previos a 2001
                if y < 2001 or folder in blacklist:
                    df['Chl_CTD'] = np.nan
                else:
                    df.loc[(df['Chl_CTD'] < -5) | (df['Chl_CTD'] > 100), 'Chl_CTD'] = np.nan
                
                ctd_chunks.append(df)
        except Exception:
            pass
            
    df_ctd = pd.concat(ctd_chunks, ignore_index=True)
    df_ctd['Depth_Round'] = df_ctd['Depth'].round(0)
    
    print(f"Total registros CTD: {len(df_ctd)}")
    
    print("3. Cruzando CTD con Lat/Lon y Fechas...")
    df_ctd = df_ctd.merge(station_map, on=['Linea', 'Estacion'], how='left')
    df_ctd = df_ctd.merge(date_map, on=['Año', 'mes'], how='left')
    df_ctd = df_ctd.dropna(subset=['Latitud', 'Longitud', 'Fecha'])
    
    print("4. Integrando Clorofila de Botellas (High Fidelity)...")
    df_bottles = df_bottles_raw.dropna(subset=['Chl_Bottle'])
    df_bottles['Depth_Round'] = df_bottles['Depth'].round(0)
    
    # Promediar botellas en la misma profundidad exacta para evitar duplicados
    df_bottles_agg = df_bottles.groupby(['Linea', 'Estacion', 'Año', 'mes', 'Depth_Round'])['Chl_Bottle'].mean().reset_index()
    
    df_final = df_ctd.merge(df_bottles_agg, on=['Linea', 'Estacion', 'Año', 'mes', 'Depth_Round'], how='left')
    
    print("5. Extrayendo CMEMS (Corrientes y Satélites) de forma Vectorizada...")
    df_final['uo'] = 0.0
    df_final['vo'] = 0.0
    df_final['wo'] = 0.0
    df_final['thetao'] = 15.0
    df_final['CHL_sat'] = 0.0
    
    years = sorted(df_final['Año'].unique())
    for y in years:
        cmems_file = os.path.join(project_root, f'data/raw/cmems_yearly/cmems_currents_{int(y)}_with_w.nc')
        thetao_file = os.path.join(project_root, f'data/raw/cmems_yearly_thetao/cmems_thetao_{int(y)}.nc')
        chl_sat_file = os.path.join(project_root, f'data/raw/satellite_chl_yearly/satellite_chl_{int(y)}.nc')
        
        mask = df_final['Año'] == y
        if not mask.any():
            continue
            
        print(f"  Vectorizando extracciones CMEMS para Año {y}...")
        df_y = df_final[mask]
        
        # xarray advanced vector indexing
        x = xr.DataArray(df_y['Longitud'].values, dims='points')
        y_lat = xr.DataArray(df_y['Latitud'].values, dims='points')
        z = xr.DataArray(df_y['Depth'].values, dims='points')
        t = xr.DataArray(df_y['Fecha'].values, dims='points')
        
        if os.path.exists(cmems_file):
            try:
                ds_year = xr.open_dataset(cmems_file)
            except Exception as e:
                print(f"  [Advertencia] No se pudo leer {cmems_file}: {e}")
                continue
            try:
                val = ds_year.sel(longitude=x, latitude=y_lat, depth=z, time=t, method='nearest')
                df_final.loc[mask, 'uo'] = val['uo'].values
                df_final.loc[mask, 'vo'] = val['vo'].values
                df_final.loc[mask, 'wo'] = val['wo'].values
            except Exception as e:
                print(f"  Error en corrientes {y}: {e}")
            ds_year.close()
            
        if os.path.exists(thetao_file):
            try:
                ds_thetao = xr.open_dataset(thetao_file)
            except Exception as e:
                print(f"  [Advertencia] No se pudo leer {thetao_file}: {e}")
                continue
            try:
                val_t = ds_thetao.sel(longitude=x, latitude=y_lat, depth=z, time=t, method='nearest')
                df_final.loc[mask, 'thetao'] = val_t['thetao'].values
            except: pass
            ds_thetao.close()
            
        if os.path.exists(chl_sat_file):
            try:
                ds_chl = xr.open_dataset(chl_sat_file)
            except Exception as e:
                print(f"  [Advertencia] No se pudo leer {chl_sat_file}: {e}")
                continue
            try:
                # Satélite no tiene profundidad
                val_c = ds_chl.sel(longitude=x, latitude=y_lat, time=t, method='nearest')
                df_final.loc[mask, 'CHL_sat'] = val_c['CHL'].values
            except: pass
            ds_chl.close()

    print("6. Extrayendo Batimetría Vectorizada...")
    if os.path.exists(bathy_path):
        try:
            ds_bathy = xr.open_dataset(bathy_path)
            var_name = 'altitude' if 'altitude' in ds_bathy else 'elevation'
        except Exception as e:
            print(f"Error bathy: {e}")
            ds_bathy = None
        
        if ds_bathy is not None:
            unique_coords = df_final[['Latitud', 'Longitud']].drop_duplicates()
            x = xr.DataArray(unique_coords['Longitud'].values, dims='points')
            y_lat = xr.DataArray(unique_coords['Latitud'].values, dims='points')
            
            try:
                val = ds_bathy.sel(longitude=x, latitude=y_lat, method='nearest')
                unique_coords['bathy'] = val[var_name].values
            except:
                unique_coords['bathy'] = 0.0
                
            df_final = df_final.merge(unique_coords, on=['Latitud', 'Longitud'], how='left')
            ds_bathy.close()
        
    print(f"Guardando dataset consolidado ({len(df_final)} filas) en Parquet...")
    df_final = df_final.drop(columns=['Depth_Round'])
    df_final.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"\n¡Dataset Multi-Fidelidad guardado exitosamente en: {output_path}!")

if __name__ == "__main__":
    build_augmented_dataset()
