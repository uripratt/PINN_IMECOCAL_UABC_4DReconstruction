import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.pinn_model import CoastalPINNModel

def generate_advanced_validation():
    # Fechas y parametros basicos
    target_date_str = '2005-04-15'
    target_date = pd.to_datetime(target_date_str)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_path = os.path.join(project_root, 'data/processed/imecocal_augmented.csv')
    df = pd.read_csv(csv_path)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    t0 = df['Fecha'].min()
    time_days = (target_date - t0).total_seconds() / (24 * 3600)
    
    # Cargar satelite (Target)
    ds_sat = xr.open_dataset('data/raw/satellite_chl_yearly/satellite_chl_2005.nc')
    chl_var = 'CHL' if 'CHL' in ds_sat.data_vars else 'chl'
    if chl_var not in ds_sat.data_vars:
        chl_var = list(ds_sat.data_vars.keys())[0]
        
    ds_sat_day = ds_sat.sel(time=target_date_str, method='nearest')
    chl_sat = ds_sat_day[chl_var].values
    lat_sat = ds_sat_day.latitude.values if 'latitude' in ds_sat_day else ds_sat_day.lat.values
    lon_sat = ds_sat_day.longitude.values if 'longitude' in ds_sat_day else ds_sat_day.lon.values
    ds_sat.close()
    
    # Cargar PINN
    device = torch.device("cpu")
    # Usar el mejor modelo de la batería Log-Transformed
    model_path = "./experiments/logs_Server/log_error_17/pinn_model_LogPINN_Sat_Medio.pth"
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    input_mean = state_dict['input_mean'].numpy()
    input_std = state_dict['input_std'].numpy()
    
    model = CoastalPINNModel(num_layers=6, hidden_dim=128, input_mean=input_mean, input_std=input_std)
    model.load_state_dict(state_dict)
    model.eval()
    
    # ---------------------------------------------------------
    # 1. SCATTER PLOT SUPERFICIAL (Satélite vs PINN)
    # ---------------------------------------------------------
    Lon_pinn, Lat_pinn = np.meshgrid(lon_sat, lat_sat)
    flat_lat = Lat_pinn.flatten()
    flat_lon = Lon_pinn.flatten()
    flat_z = np.zeros_like(flat_lat)
    flat_t = np.full_like(flat_lat, time_days)
    
    X_tensor = torch.tensor(np.column_stack((flat_lat, flat_lon, flat_z, flat_t)), dtype=torch.float32)
    with torch.no_grad():
        C_pred_log = model(X_tensor).numpy().flatten()
        # Nuevo modelo: Aplicamos expm1 para volver al espacio real (mg/m3)
        C_pred = np.expm1(C_pred_log) 
    
    chl_sat_flat = chl_sat.flatten()
    
    # Filtrar NaN y enmascarar Mar de Cortes
    lon_border = -116.0 + (flat_lat - 32.0) * (-0.6666)
    valid_mask = (flat_lon <= lon_border) & (~np.isnan(chl_sat_flat))
    
    sat_valid = chl_sat_flat[valid_mask]
    pred_valid = C_pred[valid_mask]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(sat_valid, pred_valid, alpha=0.1, s=2, c='blue')
    plt.plot([0, 30], [0, 30], 'r--', lw=2) # Linea 1:1
    plt.xlim(0, 5) # Recortamos a 5 para ver mejor los datos de satelite
    plt.ylim(0, 5)
    plt.xlabel('MODIS Satélite Chl-a (mg/m³)')
    plt.ylabel('PINN Predicción Chl-a (mg/m³)')
    plt.title(f'Scatter Plot Validación: Superficie (Z=0)\n$R^2$ Correlación Espacial')
    plt.grid(True, alpha=0.3)
    out_scatter = './experiments/scatter_validation.png'
    plt.savefig(out_scatter, dpi=150)
    plt.close()
    
    # ---------------------------------------------------------
    # 2. ESTRUCTURA 3D (Submuestreo estricto para ahorrar RAM)
    # ---------------------------------------------------------
    res_xy = 40  # Baja resolución horizontal
    res_z = 20   # Baja resolución vertical (0 a 100m)
    
    lats_3d = np.linspace(23.82, 32.75, res_xy)
    lons_3d = np.linspace(-119.85, -111.92, res_xy)
    depths_3d = np.linspace(0, 100, res_z)
    
    Lon3D, Lat3D, Dep3D = np.meshgrid(lons_3d, lats_3d, depths_3d)
    flat_lat3 = Lat3D.flatten()
    flat_lon3 = Lon3D.flatten()
    flat_z3 = Dep3D.flatten()
    flat_t3 = np.full_like(flat_lat3, time_days)
    
    X_tensor3 = torch.tensor(np.column_stack((flat_lat3, flat_lon3, flat_z3, flat_t3)), dtype=torch.float32)
    with torch.no_grad():
        C_pred3 = model(X_tensor3).numpy().flatten()
        
    C_grid3 = C_pred3.reshape(Lat3D.shape)
    
    # Enmascarar tierra
    lon_border3 = -116.0 + (Lat3D - 32.0) * (-0.6666)
    C_grid3[Lon3D > lon_border3] = np.nan
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Dibujar isosuperficies (Scatter de puntos con clorofila > 1.5)
    mask_high_chl = (C_grid3 > 1.5) & (~np.isnan(C_grid3))
    
    sc = ax.scatter(Lon3D[mask_high_chl], Lat3D[mask_high_chl], -Dep3D[mask_high_chl], 
                    c=C_grid3[mask_high_chl], cmap='viridis', s=15, alpha=0.6, vmin=0, vmax=5)
    
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    ax.set_zlabel('Profundidad (m)')
    ax.set_title('Estructura 3D del DCM (Clorofila > 1.5 mg/m³)')
    plt.colorbar(sc, ax=ax, label='Chl-a (mg/m³)', shrink=0.5)
    
    out_3d = './experiments/3d_structure.png'
    plt.savefig(out_3d, dpi=150)
    plt.close()

if __name__ == "__main__":
    generate_advanced_validation()
