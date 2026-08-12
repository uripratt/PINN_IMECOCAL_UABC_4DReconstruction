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

from src.data_ingestion.dataloader import get_dataloaders
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
    
    # u, v, y w son 0 en tierra
    batch_u = np.zeros(batch_size)
    batch_v = np.zeros(batch_size)
    batch_w = np.zeros(batch_size)
    batch_temp = np.full(batch_size, 15.0) # Dummy temperature (15C) until real data is ingested
    batch_chl_sat = np.zeros(batch_size)   # Dummy CHL sat
    
    # Ensamblar tensor X_full: (Lat, Lon, Prof, Tiempo, u, v, w, bathy, temp, chl_sat)
    X_numpy = np.column_stack((batch_lats, batch_lons, batch_depths, batch_times, batch_u, batch_v, batch_w, batch_bathy, batch_temp, batch_chl_sat))
    return torch.tensor(X_numpy, dtype=torch.float32).to(device)

# satellite batch is handled directly in the loop now
def train_pinn(epochs=10, batch_size=256, lr=1e-3, curriculum_epochs=5, colloc_ratio=4, lambda_sat=0.5, lbfgs_epochs=0, run_name="PINN_Training"):
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
        db_path = os.path.join(os.path.dirname(__file__), 'mlflow.db')
        print(f"Usando MLflow local (SQLite): {db_path}")
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        
    mlflow.set_experiment("PINNs_BajaCalifornia")
    
    # 2. Cargar DataLoader
    print("Preparando DataLoaders (Train/Test Split)...")
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)
    
    max_time_days = train_loader.dataset.df['time_days'].max()
    max_depth = train_loader.dataset.df['Profundidad'].max()
    
    print("Cargando malla de tierra para Puntos de Colocación...")
    land_data = load_land_points()
    
    # 3. Inicializar Arquitectura (Agent 3) y Física (Agent 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Extraer estadísticas para normalización (solo del train set para evitar data leakage)
    dataset_x = train_loader.dataset.X[:, 0:4]
    mean_x = dataset_x.mean(dim=0).numpy()
    std_x = dataset_x.std(dim=0).numpy()
    print(f"Normalizando entradas con Mean: {mean_x} y Std: {std_x}")
    
    model = CoastalPINNModel(num_layers=6, hidden_dim=128, input_mean=mean_x, input_std=std_x).to(device)
    # Pasamos el std_x a la física para corregir la dimensionalidad de las derivadas
    physics = CoastalPhysicsPINN(diff_coef=0.1, std_x=torch.tensor(std_x, dtype=torch.float32, device=device)).to(device)
    
    # El optimizador ahora entrena tanto la red neuronal como los parámetros biológicos (Física Inversa)
    optimizer = optim.Adam(list(model.parameters()) + list(physics.parameters()), lr=lr, weight_decay=1e-4)
    mse_loss = torch.nn.MSELoss()
    
    with mlflow.start_run(run_name=run_name):
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
        history_test_loss = []
        history_steps = []
        best_test_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            epoch_data_loss = 0.0
            epoch_physics_loss = 0.0
            epoch_sat_loss = 0.0
            epoch_sat_loss = 0.0
            epoch_sat_loss = 0.0
            
            # Curriculum Learning: El peso de la física aumenta gradualmente hasta 500.0 (para balancear magnitudes)
            lambda_phys = (epoch / curriculum_epochs) * 500.0 if epoch < curriculum_epochs else 500.0
                
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
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
                u_velocities_phys = physics_x_full[:, 4:7] # u, v, w
                bathy_phys = physics_x_full[:, 7:8]
                temp_phys = physics_x_full[:, 8:9]
                
                optimizer.zero_grad()
                
                # --- DATA LOSS (Solo empíricos) ---
                x_coords_data = batch_x_full[:, 0:4].to(device)
                pred_y = model(x_coords_data)
                loss_data = mse_loss(pred_y, batch_y)
                
                # --- PHYSICS LOSS (Empíricos + Tierra firme) ---
                loss_physics = physics.compute_physics_loss(model, x_coords_phys, u_velocities_phys, temp_phys, bathy_phys)
                
                # --- SATELLITE LOSS (Condición de frontera z=0) ---
                chl_sat = batch_x_full[:, 9:10].to(device)
                mask_sat = (chl_sat > 0.01).squeeze() # Filtramos nubes/sin datos
                if mask_sat.any():
                    x_coords_sat = batch_x_full[mask_sat, 0:4].clone().to(device)
                    x_coords_sat[:, 2] = 0.0 # Forzar profundidad z=0
                    pred_sat = model(x_coords_sat)
                    loss_satelite = mse_loss(pred_sat, chl_sat[mask_sat])
                else:
                    loss_satelite = torch.tensor(0.0, device=device)
                
                # --- TOTAL LOSS ---
                loss_total = loss_data + lambda_phys * loss_physics + lambda_sat * loss_satelite
                
                loss_total.backward()
                optimizer.step()
                
                epoch_data_loss += loss_data.item()
                epoch_physics_loss += loss_physics.item()
                epoch_sat_loss += loss_satelite.item()
                
                pbar.set_postfix({"L_data": f"{loss_data.item():.4f}", "L_phys": f"{loss_physics.item():.4f}", "L_sat": f"{loss_satelite.item():.4f}"})
            
            # Promedios del epoch (Train)
            avg_data_loss = epoch_data_loss / len(train_loader)
            avg_phys_loss = epoch_physics_loss / len(train_loader)
            avg_sat_loss = epoch_sat_loss / len(train_loader)
            
            # --- EVALUACIÓN (TEST SET) ---
            model.eval()
            epoch_test_loss = 0.0
            with torch.no_grad():
                for test_x, test_y in test_loader:
                    test_x = test_x.to(device)
                    test_y = test_y.to(device)
                    x_coords_test = test_x[:, 0:4]
                    pred_test = model(x_coords_test)
                    loss_test = mse_loss(pred_test, test_y)
                    epoch_test_loss += loss_test.item()
            avg_test_loss = epoch_test_loss / len(test_loader)
            
            # Early Stopping Check
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            mlflow.log_metrics({
                "Data_Loss": avg_data_loss,
                "Physics_Loss": avg_phys_loss,
                "Sat_Loss": avg_sat_loss,
                "Test_Loss": avg_test_loss,
                "Total_Loss": avg_data_loss + lambda_phys * avg_phys_loss + lambda_sat * avg_sat_loss,
                "lambda_phys": lambda_phys
            }, step=epoch)
            
            history_data_loss.append(avg_data_loss)
            history_phys_loss.append(avg_phys_loss)
            history_test_loss.append(avg_test_loss)
            history_steps.append(epoch)
            
        # ==========================================
        # FASE 2: Refinamiento con L-BFGS
        # ==========================================
        if lbfgs_epochs > 0:
            print(f"\nIniciando refinamiento de {lbfgs_epochs} epochs con optimizador L-BFGS...")
            
            # Usamos el peso final del curriculum
            lambda_phys_final = 500.0 
            
            for epoch in range(epochs, epochs + lbfgs_epochs):
                model.train()
                epoch_data_loss = 0.0
                epoch_physics_loss = 0.0
                
                pbar = tqdm(train_loader, desc=f"L-BFGS Epoch {epoch+1}/{epochs + lbfgs_epochs}")
                nan_detected = False
                for batch_x_full, batch_y in pbar:
                    # Re-inicializar L-BFGS por cada mini-batch para no arrastrar
                    # historial del Hessiano inválido de batches anteriores.
                    optimizer_lbfgs = optim.LBFGS(list(model.parameters()) + list(physics.parameters()), 
                                                  lr=0.01, 
                                                  max_iter=20, 
                                                  tolerance_grad=1e-7, 
                                                  tolerance_change=1e-9, 
                                                  history_size=50,
                                                  line_search_fn="strong_wolfe")
                    
                    batch_x_full = batch_x_full.to(device)
                    batch_y = batch_y.to(device)
                    
                    if land_data is not None:
                        num_colloc = batch_x_full.shape[0] * colloc_ratio
                        colloc_x_full = get_collocation_batch(land_data, num_colloc, max_time_days, max_depth, device)
                        physics_x_full = torch.cat([batch_x_full, colloc_x_full], dim=0)
                    else:
                        physics_x_full = batch_x_full
                    
                    x_coords_phys = physics_x_full[:, 0:4].requires_grad_(True)
                    u_velocities_phys = physics_x_full[:, 4:7] # u, v, w
                    bathy_phys = physics_x_full[:, 7:8]
                    temp_phys = physics_x_full[:, 8:9]
                    x_coords_data = batch_x_full[:, 0:4].to(device)
                    
                    def closure():
                        optimizer_lbfgs.zero_grad()
                        pred_y = model(x_coords_data)
                        loss_d = mse_loss(pred_y, batch_y)
                        loss_p = physics.compute_physics_loss(model, x_coords_phys, u_velocities_phys, temp_phys, bathy_phys)
                        loss_t = loss_d + lambda_phys_final * loss_p
                        loss_t.backward()
                        return loss_t
                    
                    # Guardamos el estado antes del step por si da NaN
                    prev_state = {k: v.clone() for k, v in model.state_dict().items()}
                    
                    # L-BFGS ejecuta el closure múltiples veces internamente
                    optimizer_lbfgs.step(closure)
                    
                    # Recalculamos loss para logging sin afectar los gradientes
                    with torch.no_grad():
                        pred_y = model(x_coords_data)
                        l_data = mse_loss(pred_y, batch_y).item()
                    with torch.enable_grad():
                        l_phys = physics.compute_physics_loss(model, x_coords_phys, u_velocities_phys, temp_phys, bathy_phys).item()
                        
                    if np.isnan(l_data) or np.isnan(l_phys):
                        print("\n[Advertencia] NaN detectado en L-BFGS. Revirtiendo pesos y cancelando L-BFGS para esta run.")
                        model.load_state_dict(prev_state)
                        nan_detected = True
                        break
                        
                    epoch_data_loss += l_data
                    epoch_physics_loss += l_phys
                    pbar.set_postfix({"L_data": f"{l_data:.4f}", "L_phys": f"{l_phys:.4f}"})
                
                if nan_detected:
                    break
                
                avg_data_loss = epoch_data_loss / len(train_loader)
                avg_phys_loss = epoch_physics_loss / len(train_loader)
                
                # --- EVALUACIÓN (TEST SET) L-BFGS ---
                model.eval()
                epoch_test_loss = 0.0
                with torch.no_grad():
                    for test_x, test_y in test_loader:
                        test_x = test_x.to(device)
                        test_y = test_y.to(device)
                        x_coords_test = test_x[:, 0:4]
                        pred_test = model(x_coords_test)
                        loss_test = mse_loss(pred_test, test_y)
                        epoch_test_loss += loss_test.item()
                avg_test_loss = epoch_test_loss / len(test_loader)
                
                # Early Stopping Check (L-BFGS)
                if avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
                mlflow.log_metrics({
                    "Data_Loss": avg_data_loss,
                    "Physics_Loss": avg_phys_loss,
                    "Test_Loss": avg_test_loss,
                    "Total_Loss": avg_data_loss + lambda_phys_final * avg_phys_loss,
                    "lambda_phys": lambda_phys_final
                }, step=epoch)
                
                history_data_loss.append(avg_data_loss)
                history_phys_loss.append(avg_phys_loss)
                history_test_loss.append(avg_test_loss)
                history_steps.append(epoch)
                
        print("Entrenamiento completado. Restaurando el mejor modelo según Test Loss...")
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        print(f"Mejor Test Loss alcanzado: {best_test_loss:.4f}")
        model_path = os.path.join(os.path.dirname(__file__), f"pinn_model_{run_name}.pth")
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        
        # Guardar métricas sin procesar en un CSV local
        import pandas as pd
        metrics_df = pd.DataFrame({
            'Epoch': history_steps,
            'Data_Loss': history_data_loss,
            'Physics_Loss': history_phys_loss,
            'Test_Loss': history_test_loss
        })
        csv_path = os.path.join(os.path.dirname(__file__), f"training_metrics_{run_name}.csv")
        metrics_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)
        
        # Generar y guardar gráfico de métricas
        plt.figure(figsize=(10, 6))
        plt.plot(history_steps, history_data_loss, label='Train Data Loss', color='blue')
        plt.plot(history_steps, history_test_loss, label='Test Loss (Hold-out)', color='green')
        plt.plot(history_steps, history_phys_loss, label='Physics Loss', color='red')
        plt.yscale('log')
        plt.title('Convergencia del Entrenamiento PINN 4D')
        plt.xlabel('Epochs')
        plt.ylabel('Pérdida (Log Scale)')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        metrics_file = os.path.join(os.path.dirname(__file__), f"training_metrics_{run_name}.png")
        plt.savefig(metrics_file, dpi=300, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(metrics_file)
        
        # Generar inferencia espacial y guardar el mapa
        print("Generando mapa de inferencia final para MLflow...")
        lat_bnds = [23.82, 32.75]
        lon_bnds = [-119.85, -111.92]
        inference_file = plot_continuous_field(model_path, lat_bnds, lon_bnds, depth=0.0, time_day=100.0, resolution=200, run_name=run_name)
        mlflow.log_artifact(inference_file)
        
        print(f"Artefactos y métricas registradas en {tracking_uri if tracking_uri else mlruns_dir}")

if __name__ == "__main__":
    # Entrenamiento Completo en Servidor (Fase Gold)
    # Incluye fase Adam + fase L-BFGS. lambda_sat se sube a 10.0.
    train_pinn(epochs=3000, batch_size=1024, lr=1e-3, curriculum_epochs=2000, colloc_ratio=20, lambda_sat=10.0, lbfgs_epochs=500)
