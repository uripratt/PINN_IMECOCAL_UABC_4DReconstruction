import os
import sys
import torch
import torch.optim as optim
import mlflow
from tqdm import tqdm
import numpy as np
import xarray as xr

# Asegurar que se puede importar src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_ingestion.dataloader import get_dataloader
from src.models.pinn_model import CoastalPINNModel
from src.physics.physics_loss import CoastalPhysicsPINN
from experiments.plot_inference import plot_continuous_field
import matplotlib.pyplot as plt

def load_land_points():
    """Carga las coordenadas de tierra firme desde el archivo ETOPO."""
    bathy_file = os.path.join(os.path.dirname(__file__), '../data/raw/etopo_bathymetry.nc')
    if not os.path.exists(bathy_file):
        print("Advertencia: No se encontró etopo_bathymetry.nc. No se usarán puntos de colocación terrestres.")
        return None
    ds = xr.open_dataset(bathy_file)
    var_name = 'altitude' if 'altitude' in ds else 'elevation'
    lats = ds.latitude.values
    lons = ds.longitude.values
    Lon, Lat = np.meshgrid(lons, lats)
    elev = ds[var_name].values
    
    mask = elev > 0
    land_lats = Lat[mask]
    land_lons = Lon[mask]
    land_elevs = elev[mask]
    ds.close()
    return land_lats, land_lons, land_elevs

def get_collocation_batch(land_data, batch_size, max_time_days, max_depth, device):
    """Genera un batch de puntos aleatorios sobre tierra firme para imponer Dirichlet."""
    land_lats, land_lons, land_elevs = land_data
    
    # Muestrear aleatoriamente 'batch_size' índices
    indices = np.random.choice(len(land_lats), batch_size, replace=True)
    
    batch_lats = land_lats[indices]
    batch_lons = land_lons[indices]
    batch_bathy = land_elevs[indices]
    
    # Muestrear tiempo y profundidad aleatoriamente
    batch_times = np.random.uniform(0, max_time_days, batch_size)
    batch_depths = np.random.uniform(0, max_depth, batch_size)
    
    # u y v son 0 en tierra
    batch_u = np.zeros(batch_size)
    batch_v = np.zeros(batch_size)
    
    # Ensamblar tensor X_full: (Lat, Lon, Prof, Tiempo, u, v, bathy)
    X_numpy = np.column_stack((batch_lats, batch_lons, batch_depths, batch_times, batch_u, batch_v, batch_bathy))
    return torch.tensor(X_numpy, dtype=torch.float32).to(device)

def train_pinn(epochs=10, batch_size=256, lr=1e-3, curriculum_epochs=5, colloc_ratio=4, lbfgs_epochs=0):
    """
    Experiment Harness (Agentes 3 y 4): Entrena la PINN usando Curriculum Learning 
    y registra experimentos y métricas en MLflow.
    Incluye una fase final opcional con L-BFGS para eliminar oscilaciones.
    """
    print("Iniciando Experiment Harness (PINN Training)...")
    
    # 1. Configurar MLflow
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        print(f"Conectando al servidor MLflow remoto: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlruns_dir = os.path.join(os.path.dirname(__file__), 'mlruns')
        os.makedirs(mlruns_dir, exist_ok=True)
        print(f"Usando MLflow local: {mlruns_dir}")
        mlflow.set_tracking_uri(f"file://{mlruns_dir}")
        
    mlflow.set_experiment("PINNs_BajaCalifornia")
    
    # 2. Cargar DataLoader
    print("Preparando DataLoader...")
    dataloader = get_dataloader(batch_size=batch_size, shuffle=True)
    
    max_time_days = dataloader.dataset.df['time_days'].max()
    max_depth = dataloader.dataset.df['Profundidad'].max()
    
    print("Cargando malla de tierra para Puntos de Colocación...")
    land_data = load_land_points()
    
    # 3. Inicializar Arquitectura (Agent 3) y Física (Agent 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Extraer estadísticas para normalización
    dataset_x = dataloader.dataset.X[:, 0:4]
    mean_x = dataset_x.mean(dim=0).numpy()
    std_x = dataset_x.std(dim=0).numpy()
    print(f"Normalizando entradas con Mean: {mean_x} y Std: {std_x}")
    
    model = CoastalPINNModel(num_layers=6, hidden_dim=128, input_mean=mean_x, input_std=std_x).to(device)
    physics = CoastalPhysicsPINN(diff_coef=0.1, decay_rate=0.01)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = torch.nn.MSELoss()
    
    with mlflow.start_run(run_name="PINN_Curriculum_Learning"):
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "colloc_ratio": colloc_ratio,
            "learning_rate": lr,
            "curriculum_epochs": curriculum_epochs,
            "model_layers": 6,
            "hidden_dim": 128
        })
        
        history_data_loss = []
        history_phys_loss = []
        history_steps = []
        
        for epoch in range(epochs):
            model.train()
            epoch_data_loss = 0.0
            epoch_physics_loss = 0.0
            
            # Curriculum Learning: El peso de la física aumenta gradualmente
            lambda_phys = (epoch / curriculum_epochs) * 0.1 if epoch < curriculum_epochs else 0.1
                
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_x_full, batch_y in pbar:
                batch_x_full = batch_x_full.to(device)
                batch_y = batch_y.to(device)
                # --- INYECCIÓN DE PUNTOS DE COLOCACIÓN (TIERRA FIRME) ---
                if land_data is not None:
                    # Generamos puntos de tierra proporcionales al batch (ej. 4x más puntos de tierra)
                    num_colloc = batch_x_full.shape[0] * colloc_ratio
                    colloc_x_full = get_collocation_batch(land_data, num_colloc, max_time_days, max_depth, device)
                    physics_x_full = torch.cat([batch_x_full, colloc_x_full], dim=0)
                else:
                    physics_x_full = batch_x_full
                
                # Desacoplar tensores físicos (sobre la unión del mar y tierra)
                x_coords_phys = physics_x_full[:, 0:4].requires_grad_(True)
                u_velocities_phys = physics_x_full[:, 4:6]
                bathy_phys = physics_x_full[:, 6:7]
                
                optimizer.zero_grad()
                
                # --- DATA LOSS (Solo empíricos) ---
                x_coords_data = batch_x_full[:, 0:4].to(device)
                pred_y = model(x_coords_data)
                loss_data = mse_loss(pred_y, batch_y)
                
                # --- PHYSICS LOSS (Empíricos + Tierra firme) ---
                loss_physics = physics.compute_physics_loss(model, x_coords_phys, u_velocities_phys, bathymetry=bathy_phys)
                
                # --- TOTAL LOSS ---
                loss_total = loss_data + lambda_phys * loss_physics
                
                loss_total.backward()
                optimizer.step()
                
                epoch_data_loss += loss_data.item()
                epoch_physics_loss += loss_physics.item()
                
                pbar.set_postfix({"L_data": f"{loss_data.item():.4f}", "L_phys": f"{loss_physics.item():.4f}"})
            
            # Promedios del epoch
            avg_data_loss = epoch_data_loss / len(dataloader)
            avg_phys_loss = epoch_physics_loss / len(dataloader)
            
            mlflow.log_metrics({
                "Data_Loss": avg_data_loss,
                "Physics_Loss": avg_phys_loss,
                "Total_Loss": avg_data_loss + lambda_phys * avg_phys_loss,
                "lambda_phys": lambda_phys
            }, step=epoch)
            
            history_data_loss.append(avg_data_loss)
            history_phys_loss.append(avg_phys_loss)
            history_steps.append(epoch)
            
        # ==========================================
        # FASE 2: Refinamiento con L-BFGS
        # ==========================================
        if lbfgs_epochs > 0:
            print(f"\nIniciando refinamiento de {lbfgs_epochs} epochs con optimizador L-BFGS...")
            optimizer_lbfgs = optim.LBFGS(model.parameters(), 
                                          lr=0.1, 
                                          max_iter=20, 
                                          tolerance_grad=1e-7, 
                                          tolerance_change=1e-9, 
                                          history_size=50,
                                          line_search_fn="strong_wolfe")
            
            # Usamos el peso final del curriculum
            lambda_phys_final = 0.1 
            
            for epoch in range(epochs, epochs + lbfgs_epochs):
                model.train()
                epoch_data_loss = 0.0
                epoch_physics_loss = 0.0
                
                pbar = tqdm(dataloader, desc=f"L-BFGS Epoch {epoch+1}/{epochs + lbfgs_epochs}")
                for batch_x_full, batch_y in pbar:
                    batch_x_full = batch_x_full.to(device)
                    batch_y = batch_y.to(device)
                    
                    if land_data is not None:
                        num_colloc = batch_x_full.shape[0] * colloc_ratio
                        colloc_x_full = get_collocation_batch(land_data, num_colloc, max_time_days, max_depth, device)
                        physics_x_full = torch.cat([batch_x_full, colloc_x_full], dim=0)
                    else:
                        physics_x_full = batch_x_full
                    
                    x_coords_phys = physics_x_full[:, 0:4].requires_grad_(True)
                    u_velocities_phys = physics_x_full[:, 4:6]
                    bathy_phys = physics_x_full[:, 6:7]
                    x_coords_data = batch_x_full[:, 0:4].to(device)
                    
                    def closure():
                        optimizer_lbfgs.zero_grad()
                        pred_y = model(x_coords_data)
                        loss_d = mse_loss(pred_y, batch_y)
                        loss_p = physics.compute_physics_loss(model, x_coords_phys, u_velocities_phys, bathymetry=bathy_phys)
                        loss_t = loss_d + lambda_phys_final * loss_p
                        loss_t.backward()
                        return loss_t
                    
                    # L-BFGS ejecuta el closure múltiples veces internamente
                    optimizer_lbfgs.step(closure)
                    
                    # Recalculamos loss para logging sin afectar los gradientes
                    with torch.no_grad():
                        pred_y = model(x_coords_data)
                        l_data = mse_loss(pred_y, batch_y).item()
                        # El physics loss requiere gradientes temporalmente
                    with torch.enable_grad():
                        l_phys = physics.compute_physics_loss(model, x_coords_phys, u_velocities_phys, bathymetry=bathy_phys).item()
                        
                    epoch_data_loss += l_data
                    epoch_physics_loss += l_phys
                    pbar.set_postfix({"L_data": f"{l_data:.4f}", "L_phys": f"{l_phys:.4f}"})
                
                avg_data_loss = epoch_data_loss / len(dataloader)
                avg_phys_loss = epoch_physics_loss / len(dataloader)
                
                mlflow.log_metrics({
                    "Data_Loss": avg_data_loss,
                    "Physics_Loss": avg_phys_loss,
                    "Total_Loss": avg_data_loss + lambda_phys_final * avg_phys_loss,
                    "lambda_phys": lambda_phys_final
                }, step=epoch)
                
                history_data_loss.append(avg_data_loss)
                history_phys_loss.append(avg_phys_loss)
                history_steps.append(epoch)
                
        print("Entrenamiento completado. Guardando modelo...")
        model_path = os.path.join(os.path.dirname(__file__), "pinn_model_final.pth")
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        
        # Generar y guardar gráfico de métricas
        plt.figure(figsize=(10, 6))
        plt.plot(history_steps, history_data_loss, label='Data Loss', color='blue')
        plt.plot(history_steps, history_phys_loss, label='Physics Loss', color='red')
        plt.yscale('log')
        plt.title('Convergencia del Entrenamiento PINN 4D')
        plt.xlabel('Epochs')
        plt.ylabel('Pérdida (Log Scale)')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        metrics_file = os.path.join(os.path.dirname(__file__), "training_metrics_convergence.png")
        plt.savefig(metrics_file, dpi=300, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(metrics_file)
        
        # Generar inferencia espacial y guardar el mapa
        print("Generando mapa de inferencia final para MLflow...")
        lat_bnds = [23.82, 32.75]
        lon_bnds = [-119.85, -111.92]
        inference_file = plot_continuous_field(model_path, lat_bnds, lon_bnds, depth=0.0, time_day=100.0, resolution=200)
        mlflow.log_artifact(inference_file)
        
        print(f"Artefactos y métricas registradas en {tracking_uri if tracking_uri else mlruns_dir}")

if __name__ == "__main__":
    # Entrenamiento Completo en Servidor (Fase Gold)
    # Incluye fase Adam + fase L-BFGS
    train_pinn(epochs=3000, batch_size=1024, lr=1e-3, curriculum_epochs=2000, colloc_ratio=4, lbfgs_epochs=500)
