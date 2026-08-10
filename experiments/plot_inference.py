import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

# Asegurar que se puede importar src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.pinn_model import CoastalPINNModel

def plot_continuous_field(model_path, lat_range, lon_range, depth=0.0, time_day=100.0, resolution=200):
    """
    Genera predicciones en una malla 2D espaciotemporal continua utilizando la PINN entrenada,
    y visualiza el resultado para evaluar la reconstrucción física (mesh-free).
    """
    print("Cargando arquitectura PINN...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Cargar los pesos entrenados
    state_dict = torch.load(model_path, map_location=device)
    
    # Extraer parámetros de normalización si existen en el modelo guardado
    if 'input_mean' in state_dict and 'input_std' in state_dict:
        input_mean = state_dict['input_mean'].cpu().numpy()
        input_std = state_dict['input_std'].cpu().numpy()
        model = CoastalPINNModel(num_layers=6, hidden_dim=128, input_mean=input_mean, input_std=input_std).to(device)
    else:
        model = CoastalPINNModel(num_layers=6, hidden_dim=128).to(device)
        
    model.load_state_dict(state_dict)
    model.eval()

    # 1. Generar malla espacial densa (Resolution x Resolution)
    print(f"Generando malla de interpolación de {resolution}x{resolution} píxeles...")
    lats = np.linspace(lat_range[0], lat_range[1], resolution)
    lons = np.linspace(lon_range[0], lon_range[1], resolution)
    Lon, Lat = np.meshgrid(lons, lats)
    
    # 2. Aplanar tensores para alimentar la MLP
    flat_lats = Lat.flatten()
    flat_lons = Lon.flatten()
    flat_depth = np.full_like(flat_lats, depth)
    flat_time = np.full_like(flat_lats, time_day)
    
    # Tensor de entrada: (Lat, Lon, Profundidad, Tiempo)
    X_infer = np.column_stack((flat_lats, flat_lons, flat_depth, flat_time))
    X_tensor = torch.tensor(X_infer, dtype=torch.float32).to(device)
    
    print("Calculando predicciones de Clorofila-a (Inferencia continua)...")
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()
        
    # Reconstruir el campo 2D
    C_pred = preds.reshape(resolution, resolution)
    
    # 3. Visualización Oceanográfica
    print("Generando mapa de calor oceanográfico con batimetría y topografía...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Utilizamos contourf para visualizar las transiciones fluidas de las PDEs
    contour = ax.contourf(Lon, Lat, C_pred, levels=60, cmap='viridis', transform=ccrs.PlateCarree(), alpha=0.85)
    cbar = plt.colorbar(contour, ax=ax, label='Clorofila-$a$ predicha (mg/m³)', shrink=0.8)
    
    # Superponer Topografía y Batimetría
    bathy_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw/etopo_bathymetry.nc'))
    if os.path.exists(bathy_file):
        ds_bathy = xr.open_dataset(bathy_file)
        var_name = 'altitude' if 'altitude' in ds_bathy else 'elevation'
        lons_b = ds_bathy.longitude.values
        lats_b = ds_bathy.latitude.values
        elev = ds_bathy[var_name].values
        
        # Batimetría en curvas de nivel (transparentes)
        ax.contour(lons_b, lats_b, elev, levels=[-4000, -3000, -2000, -1000, -500, -200, -50], colors='white', linewidths=0.6, alpha=0.4, transform=ccrs.PlateCarree())
        
        # Topografía (Tierra firme) con transparencia
        ax.contourf(lons_b, lats_b, elev, levels=[0, 10000], colors=['dimgray'], alpha=0.6, transform=ccrs.PlateCarree())
        ds_bathy.close()

    ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='black', linestyle=':')
    
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5, color='gray')
    gl.top_labels = False
    gl.right_labels = False
    
    plt.title(f'Reconstrucción 4D - Campo Continuo de Clorofila-a (PINN)\\nProfundidad: {depth} m | Día Simulado: {int(time_day)}', fontsize=14, pad=15)
    
    # Guardar en alta calidad
    out_file = os.path.join(os.path.dirname(__file__), f"pinn_inference_z{int(depth)}_t{int(time_day)}.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Inferencia completada. Mapa de alta resolución guardado en: {out_file}")
    return out_file

if __name__ == "__main__":
    # Dominio de Baja California (IMECOCAL + CMEMS Subset)
    lat_bnds = [23.82, 32.75]
    lon_bnds = [-119.85, -111.92]
    
    model_pth = os.path.join(os.path.dirname(__file__), "pinn_model_final.pth")
    
    # Demostración: Reconstruir la superficie del océano (0m) en el día 100 del dataset
    plot_continuous_field(model_pth, lat_bnds, lon_bnds, depth=0.0, time_day=100.0, resolution=250)
