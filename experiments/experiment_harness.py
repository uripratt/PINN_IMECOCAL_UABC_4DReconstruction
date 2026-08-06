import os
import sys
import torch
import torch.optim as optim
import mlflow
from tqdm import tqdm

# Asegurar que se puede importar src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_ingestion.dataloader import get_dataloader
from src.models.pinn_model import CoastalPINNModel
from src.physics.physics_loss import CoastalPhysicsPINN

def train_pinn(epochs=10, batch_size=256, lr=1e-3, curriculum_epochs=5):
    """
    Experiment Harness (Agentes 3 y 4): Entrena la PINN usando Curriculum Learning 
    y registra experimentos y métricas en MLflow.
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
    
    # 3. Inicializar Arquitectura (Agent 3) y Física (Agent 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    model = CoastalPINNModel(num_layers=6, hidden_dim=128).to(device)
    physics = CoastalPhysicsPINN(diff_coef=0.1, decay_rate=0.01)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = torch.nn.MSELoss()
    
    with mlflow.start_run(run_name="PINN_Curriculum_Learning"):
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "curriculum_epochs": curriculum_epochs,
            "model_layers": 6,
            "hidden_dim": 128
        })
        
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
                
                # Desacoplar tensores: (Lat, Lon, Prof, Tiempo, u, v, bathy)
                # El modelo neuronal solo necesita las 4 coordenadas espaciotemporales
                x_coords = batch_x_full[:, 0:4].requires_grad_(True)
                u_velocities = batch_x_full[:, 4:6]
                bathy = batch_x_full[:, 6:7]
                
                optimizer.zero_grad()
                
                # --- DATA LOSS ---
                pred_y = model(x_coords)
                loss_data = mse_loss(pred_y, batch_y)
                
                # --- PHYSICS LOSS ---
                loss_physics = physics.compute_physics_loss(model, x_coords, u_velocities, bathymetry=bathy)
                
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
            
        print("Entrenamiento completado. Guardando modelo...")
        model_path = os.path.join(os.path.dirname(__file__), "pinn_model_final.pth")
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        print(f"Artefactos y métricas registradas en {mlruns_dir}")

if __name__ == "__main__":
    # Entrenamiento Completo en Servidor (Fase Gold)
    # - 5000 epochs para permitir que la red minimice el error físico y de datos.
    # - Batch size más grande (1024) para aprovechar la VRAM de la GPU del servidor.
    # - El peso de la física subirá lentamente durante los primeros 2000 epochs.
    train_pinn(epochs=5000, batch_size=1024, lr=1e-3, curriculum_epochs=2000)
