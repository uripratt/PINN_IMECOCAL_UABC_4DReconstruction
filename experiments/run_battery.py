import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.experiment_harness import train_pinn

def run_experiment_battery():
    print("=========================================================")
    print(" BATERÍA DE EXPERIMENTOS: PINN V2 (Log-Transformed)")
    print("=========================================================")
    
    # Hemos devuelto la red a 6 capas x 128 neuronas (para aprovechar el control del logaritmo)
    # y ahora vamos a buscar el equilibrio perfecto entre la Física y el Satélite.
    
    # 1. Configuración Fija
    epochs = 10000
    lbfgs_epochs = 0 # Apagamos L-BFGS de momento para iterar rápido con Adam
    num_layers = 6
    hidden_dim = 128
    
    # 2. Grid de Parámetros (Hyperparameter Sweep)
    # Vamos a probar distintos pesos para el satélite (lambda_sat) y tasas de aprendizaje.
    sweep_config = [
        {"run_name": "LogPINN_Sat_Fuerte", "lambda_sat": 15.0, "lr": 5e-4},
        {"run_name": "LogPINN_Sat_Medio", "lambda_sat": 5.0, "lr": 5e-4},
        {"run_name": "LogPINN_Sat_Fuerte_LRLento", "lambda_sat": 15.0, "lr": 1e-4},
        {"run_name": "LogPINN_Sat_Medio_LRLento", "lambda_sat": 5.0, "lr": 1e-4}
    ]
    
    total_runs = len(sweep_config)
    
    for i, config in enumerate(sweep_config):
        print(f"\n[{i+1}/{total_runs}] Lanzando Experimento: {config['run_name']}")
        print(f"   -> lambda_sat: {config['lambda_sat']} | lr: {config['lr']}")
        
        try:
            train_pinn(
                epochs=epochs,
                batch_size=2048, # Batch grande para estabilidad
                lr=config['lr'],
                curriculum_epochs=1000,
                colloc_ratio=2,
                lambda_sat=config['lambda_sat'],
                lbfgs_epochs=lbfgs_epochs,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                run_name=config['run_name']
            )
            print(f"✅ Experimento {config['run_name']} finalizado con éxito.")
        except Exception as e:
            print(f"❌ Error en {config['run_name']}: {str(e)}")

if __name__ == "__main__":
    run_experiment_battery()
