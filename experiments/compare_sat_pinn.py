import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import sys

# Agregar path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.models.pinn_model import CoastalPINNModel

def evaluate_pinn_vs_sat():
    print("Cargando modelo óptimo (LogPINN_Sat_Fuerte)...")
    model_path = "/home/uripratt/Documents/PhD/Estades_Investigació/UABC_Ensenada/experiments/logs_Server/4runs_1000epochs/pinn_model_LogPINN_Sat_Fuerte.pth"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        _sd = torch.load(model_path, map_location=device)
        # Auto-detecta si el checkpoint usa el embedding de Fourier (2026-09-14, pinn_model.py)
        _use_fourier = 'fourier.B' in _sd
        _fourier_size = _sd['fourier.B'].shape[1] if _use_fourier else 64
        model = CoastalPINNModel(num_layers=6, hidden_dim=128, input_mean=np.zeros(4), input_std=np.ones(4),
                                  use_fourier_features=_use_fourier, fourier_mapping_size=_fourier_size).to(device)
        model.load_state_dict(_sd)
        model.eval()
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return

    print("Cargando dataset usando dataloader...")
    from src.data_ingestion.dataloader import CoastalPINNDataset
    dataset = CoastalPINNDataset(split='train')
    
    # We want ALL data, so let's just use the dataset's underlying dataframe
    df = dataset.df
    
    # Only keep valid satellite data points
    mask = (df['CHL_sat'] > 0) & (df['CHL_sat'].notna())
    df_sat = df[mask].copy()
    
    # Re-extract X for these points
    X_numpy = np.column_stack((
        df_sat['Latitud'].values,
        df_sat['Longitud'].values,
        np.zeros(len(df_sat)), # Superficie (z=0)
        df_sat['time_days'].values
    ))
    
    X_tensor = torch.tensor(X_numpy, dtype=torch.float32).to(device)
    
    # Inferencia
    print("Ejecutando inferencia histórica...")
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy().flatten()
    
    # Como el modelo es LogPINN, la salida es Softplus, ¿qué target usamos?
    # Wait, in the dataloader: y_numpy = np.log1p(self.df[['Clorofila']].values)
    # But does the network output log1p? Softplus gives positive. Let's revert log1p.
    preds_real = np.expm1(preds)
    
    # Métricas
    y_true = df_sat['CHL_sat'].values
    y_pred = preds_real
    
    r2 = r2_score(np.log1p(y_true), np.log1p(y_pred))
    pearson, _ = pearsonr(np.log1p(y_true), np.log1p(y_pred))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    
    print("\n--- Resultados de Comparación Global (PINN Surface vs Satellite) ---")
    print(f"Puntos evaluados: {len(y_true)}")
    print(f"R^2 (Log-Log): {r2:.4f}")
    print(f"Pearson r (Log-Log): {pearson:.4f}")
    print(f"RMSE Lineal: {rmse:.4f} mg/m3")
    print(f"Sesgo Medio (PINN - Sat): {np.mean(y_pred - y_true):.4f} mg/m3")
    
if __name__ == "__main__":
    evaluate_pinn_vs_sat()
