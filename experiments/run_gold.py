import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.experiment_harness import train_pinn

def run_golden_training():
    print("=========================================================")
    print(" THE GOLDEN RUN: Multi-Fidelity + LOCO + Sat Real")
    print("=========================================================")
    
    # 1. Configuración de Arquitectura
    num_layers = 6
    hidden_dim = 128
    
    # 2. Configuración de Entrenamiento Híbrido (Mejor que 10k Adam)
    # 3,000 épocas de Adam son suficientes para posicionar la red cerca del mínimo.
    # 500 épocas de L-BFGS (2º orden) colapsarán el error físico y de datos mucho 
    # mejor y más rápido que 7,000 épocas extra de Adam.
    epochs_adam = 3000
    epochs_lbfgs = 500
    
    # 2. Batería de 4 Configuraciones
    sweep_config = [
        {"run_name": "Gold_Sat_Fuerte", "lambda_sat": 15.0, "lr": 5e-4},
        {"run_name": "Gold_Sat_Medio", "lambda_sat": 5.0, "lr": 5e-4},
        {"run_name": "Gold_Sat_Fuerte_LRLento", "lambda_sat": 15.0, "lr": 1e-4},
        {"run_name": "Gold_Sat_Medio_LRLento", "lambda_sat": 5.0, "lr": 1e-4}
    ]
    
    total_runs = len(sweep_config)
    
    for i, config in enumerate(sweep_config):
        print(f"\n[{i+1}/{total_runs}] 🚀 Lanzando Experimento: {config['run_name']}")
        print(f"   -> Adam Epochs: {epochs_adam} | L-BFGS Epochs: {epochs_lbfgs}")
        print(f"   -> lambda_sat: {config['lambda_sat']} | lr: {config['lr']}")
        
        try:
            train_pinn(
                epochs=epochs_adam,
                batch_size=2048,
                lr=config['lr'],
                curriculum_epochs=2000,
                colloc_ratio=4,
                lambda_sat=config['lambda_sat'],
                lbfgs_epochs=epochs_lbfgs,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                run_name=config['run_name']
            )
            print(f"✅ Experimento {config['run_name']} finalizado con éxito.")
        except Exception as e:
            print(f"❌ Error en {config['run_name']}: {str(e)}")

if __name__ == "__main__":
    run_golden_training()
