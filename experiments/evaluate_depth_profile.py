import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_ingestion.dataloader import get_dataloaders
from src.models.pinn_model import CoastalPINNModel

def evaluate_vertical_profile():
    print("Calculando Perfil Vertical de Error (Validation Set)...")
    
    # 1. Cargar Dataloader (solo el set de validación, unseen profiles)
    _, val_loader = get_dataloaders(batch_size=8192) # Batch grande para ir rápido
    
    # 2. Configurar Modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # IMPORTANTE: Reemplazar por la ruta final de la run que se quiera validar
    # Para la batería, se guardan como pinn_model_LogPINN_Sat_Fuerte.pth, etc.
    model_path = "./experiments/logs_Server/log_error_17/pinn_model_LogPINN_Sat_Medio.pth" 
    if not os.path.exists(model_path):
        print(f"Buscando modelo en: {model_path} pero no existe. Usa un archivo válido.")
        return
        
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    input_mean = state_dict['input_mean'].cpu().numpy()
    input_std = state_dict['input_std'].cpu().numpy()
    
    model = CoastalPINNModel(num_layers=6, hidden_dim=128, input_mean=input_mean, input_std=input_std)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 3. Predicciones vs Realidad
    depths = []
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            # El dataloader nos da log1p.
            y_pred_log = model(batch_x[:, 0:4])
            
            # Extraer profundidad cruda. En batch_x la profundidad está en la columna 2.
            # Como batch_x devuelve inputs (z-scored? No, Dataloader da x_tensor raw antes de z-score en la red)
            # Hay que asegurar si batch_x ya viene normalizado o no. 
            # El CoastalPINNDataset actual retorna el tensor X *sin normalizar* (la normalización ocurre dentro de la PINN).
            # Por tanto, batch_x[:, 2] es la Profundidad en metros.
            z_batch = batch_x[:, 2].cpu().numpy()
            
            # Volvemos a espacio físico
            y_true_real = np.expm1(batch_y.cpu().numpy().flatten())
            y_pred_real = np.expm1(y_pred_log.cpu().numpy().flatten())
            
            depths.extend(z_batch)
            y_true_list.extend(y_true_real)
            y_pred_list.extend(y_pred_real)
            
    depths = np.array(depths)
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    
    # 4. Agrupar por las profundidades ESTÁNDAR reales del IMECOCAL
    # Hemos descubierto que casi todo el dataset está en 0-1, 10, 20, 50, 100, 150 y 200.
    bins = [0, 5, 15, 35, 75, 125, 175, 250]
    labels = ['Superficie (0-1m)', '10m', '20m', '50m', '100m', '150m', '200m']
    
    df_eval = pd.DataFrame({'Depth': depths, 'True_Chl': y_true, 'Pred_Chl': y_pred})
    df_eval['Depth_Bin'] = pd.cut(df_eval['Depth'], bins=bins, labels=labels, right=False)
    
    rmse_per_bin = []
    mean_depth_per_bin = []
    
    for label in labels:
        subset = df_eval[df_eval['Depth_Bin'] == label]
        if len(subset) > 0:
            rmse = np.sqrt(np.mean((subset['True_Chl'] - subset['Pred_Chl'])**2))
            mean_depth = subset['Depth'].median() # Usar mediana para que coincida con el nivel estándar
            rmse_per_bin.append(rmse)
            mean_depth_per_bin.append(mean_depth)
            print(f"Nivel {label:15s} | N={len(subset):4d} | RMSE = {rmse:.4f} mg/m3")
            
    # 5. Graficar Perfil Vertical
    plt.figure(figsize=(6, 8))
    plt.plot(rmse_per_bin, mean_depth_per_bin, 'bo-', linewidth=2, markersize=8)
    plt.gca().invert_yaxis() # Invertir el eje Y (profundidad hacia abajo)
    
    plt.xlabel('RMSE Error Clorofila-a (mg/m³)')
    plt.ylabel('Profundidad (m)')
    plt.title('Perfil Vertical de Error (Validation Set)\nPropagación de la Física 3D')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Rellenar con color suave
    plt.fill_betweenx(mean_depth_per_bin, 0, rmse_per_bin, color='blue', alpha=0.1)
    plt.xlim(0, max(rmse_per_bin)*1.2)
    
    out_file = './experiments/vertical_error_profile.png'
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"\n¡Gráfica generada en {out_file}!")

if __name__ == "__main__":
    evaluate_vertical_profile()
