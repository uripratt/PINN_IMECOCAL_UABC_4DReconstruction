import os
import glob
import xarray as xr
import numpy as np
from tqdm import tqdm

def compute_w_day(ds_day):
    R_earth = 6371000.0
    dx = ds_day.longitude.diff('longitude') * (np.pi / 180.0) * R_earth * np.cos(np.deg2rad(ds_day.latitude))
    dy = ds_day.latitude.diff('latitude') * (np.pi / 180.0) * R_earth
    
    du_dx = ds_day.uo.differentiate('longitude') / dx.mean().values
    dv_dy = ds_day.vo.differentiate('latitude') / dy.mean().values
    
    div_h = du_dx + dv_dy
    dz = ds_day.depth.diff('depth').fillna(1.0)
    w = - (div_h * dz).isel(depth=slice(None, None, -1)).cumsum('depth').isel(depth=slice(None, None, -1))
    return w.rename('wo')

def process_all_years(input_dir="data/raw/cmems_yearly"):
    files = sorted(glob.glob(os.path.join(input_dir, "cmems_currents_*.nc")))
    print(f"Archivos encontrados: {len(files)}")
    
    for f in files:
        if "with_w" in f: continue
        out_f = f.replace(".nc", "_with_w.nc")
        if os.path.exists(out_f):
            print(f"[{out_f}] Ya procesado. Saltando.")
            continue
            
        print(f"Procesando {f} día a día (Anti-RAM crash)...")
        ds = xr.open_dataset(f)
        if 'wo' in ds:
            ds.close()
            continue
            
        # Procesar día a día
        w_list = []
        for i in tqdm(range(len(ds.time)), desc="Días"):
            w_day = compute_w_day(ds.isel(time=i))
            w_list.append(w_day)
            
        # Concatenar a lo largo del tiempo
        w_full = xr.concat(w_list, dim='time')
        ds['wo'] = w_full
        
        # Guardar
        ds.to_netcdf(out_f)
        ds.close()
        print(f"Guardado {out_f}")

if __name__ == "__main__":
    process_all_years()
