import os
import sys
import numpy as np
import torch
import xarray as xr

try:
    import pygmt
    HAS_PYGMT = True
except Exception as e:
    print(f"[Aviso] No se pudo cargar PyGMT ({e}). Se usará Matplotlib como fallback.")
    HAS_PYGMT = False

# Asegurar que se puede importar src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.pinn_model import CoastalPINNModel

def plot_continuous_field(model_path, lat_range, lon_range, depth=0.0, time_day=100.0, resolution=200, run_name=""):
    """
    Genera predicciones en una malla 2D espaciotemporal continua utilizando la PINN entrenada,
    y visualiza el resultado para evaluar la reconstrucción física utilizando PyGMT (SOTA Mapping).
    """
    print("Cargando arquitectura PINN...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargar los pesos entrenados
    state_dict = torch.load(model_path, map_location=device)
    
    if 'input_mean' in state_dict and 'input_std' in state_dict:
        input_mean = state_dict['input_mean'].cpu().numpy()
        input_std = state_dict['input_std'].cpu().numpy()
        model = CoastalPINNModel(num_layers=4, hidden_dim=64, input_mean=input_mean, input_std=input_std).to(device)
    else:
        model = CoastalPINNModel(num_layers=4, hidden_dim=64).to(device)
        
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Generando malla de interpolación de {resolution}x{resolution} píxeles...")
    lats = np.linspace(lat_range[0], lat_range[1], resolution)
    lons = np.linspace(lon_range[0], lon_range[1], resolution)
    Lon, Lat = np.meshgrid(lons, lats)
    
    flat_lats = Lat.flatten()
    flat_lons = Lon.flatten()
    flat_depth = np.full_like(flat_lats, depth)
    flat_time = np.full_like(flat_lats, time_day)
    
    X_infer = np.column_stack((flat_lats, flat_lons, flat_depth, flat_time))
    X_tensor = torch.tensor(X_infer, dtype=torch.float32).to(device)
    
    print("Calculando predicciones de Clorofila-a (Inferencia continua)...")
    with torch.no_grad():
        preds_log = model(X_tensor).cpu().numpy()
        # Deshacemos la transformación logarítmica para graficar valores reales
        preds = np.expm1(preds_log)
        
    # --- MÁSCARA GEOMÉTRICA (MAR DE CORTÉS) ---
    # Ecuación de la recta que bordea la costa este de la Península de Baja California.
    # Lat 32 -> Lon -116.0 | Lat 23 -> Lon -110.0
    lon_border = -116.0 + (flat_lats - 32.0) * (-0.6666)
    
    # Todo lo que esté al este de esa frontera (mayor longitud) se descarta.
    sea_of_cortez_mask = flat_lons > lon_border
    preds[sea_of_cortez_mask] = np.nan
        
    # Reconstruir el campo 2D
    C_pred = preds.reshape(resolution, resolution)
    
    prefix = f"{run_name}_" if run_name else ""
    out_file = os.path.join(os.path.dirname(__file__), f"{prefix}pinn_inference_z{int(depth)}_t{int(time_day)}.png")
    
    if HAS_PYGMT:
        print("Generando mapa de calor oceanográfico con PyGMT...")
        da_pred = xr.DataArray(C_pred, coords=[lats, lons], dims=["lat", "lon"])
        fig = pygmt.Figure()
        region = [lon_range[0], lon_range[1], lat_range[0], lat_range[1]]
        
        title = f"Reconstruccion 4D PINN - Profundidad: {int(depth)}m, Dia: {int(time_day)}"
        fig.basemap(region=region, projection="M12c", frame=[f'WSne+t"{title}"', "xaf", "yaf"])
        
        pygmt.makecpt(cmap="viridis", series=[float(C_pred.min()), float(C_pred.max())])
        fig.grdimage(grid=da_pred, cmap=True)
        
        bathy_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw/etopo_bathymetry.nc'))
        if os.path.exists(bathy_file):
            ds_bathy = xr.open_dataset(bathy_file)
            var_name = 'altitude' if 'altitude' in ds_bathy else 'elevation'
            fig.grdcontour(
                grid=ds_bathy[var_name], 
                levels=[-4000, -3000, -2000, -1000, -500, -200, -50], 
                pen="0.6p,white,dashed"
            )
            ds_bathy.close()
            
        fig.coast(land="dimgray", shorelines="1.5p,black", borders="1/0.8p,black")
        fig.colorbar(frame=['x+l"Clorofila-a predicha (mg/m@+3@+)"'], position="JMR+o0.5c/0c+w8c/0.5c")
        fig.savefig(out_file, dpi=300)
    else:
        print("PyGMT no disponible. Generando mapa de calor con Matplotlib puro (Fallback)...")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(Lon, Lat, C_pred, levels=60, cmap='viridis')
        plt.colorbar(contour, label='Clorofila-$a$ predicha (mg/m³)')
        
        bathy_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw/etopo_bathymetry.nc'))
        if os.path.exists(bathy_file):
            ds_bathy = xr.open_dataset(bathy_file)
            var_name = 'altitude' if 'altitude' in ds_bathy else 'elevation'
            plt.contour(ds_bathy.longitude, ds_bathy.latitude, ds_bathy[var_name], 
                        levels=[-4000, -2000, -500, -50], colors='white', alpha=0.3)
            plt.contourf(ds_bathy.longitude, ds_bathy.latitude, ds_bathy[var_name], 
                         levels=[0, 10000], colors=['dimgray'])
            ds_bathy.close()
            
        plt.title(f'Reconstruccion 4D PINN - Prof: {depth}m, Dia: {int(time_day)}')
        plt.xlabel('Longitud')
        plt.ylabel('Latitud')
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Inferencia completada. Mapa guardado en: {out_file}")
    return out_file

if __name__ == "__main__":
    lat_bnds = [23.82, 32.75]
    lon_bnds = [-119.85, -111.92]
    model_pth = os.path.join(os.path.dirname(__file__), "pinn_model_final.pth")
    plot_continuous_field(model_pth, lat_bnds, lon_bnds, depth=0.0, time_day=100.0, resolution=250)
