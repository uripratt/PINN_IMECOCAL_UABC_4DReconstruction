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
    
    run_name = "Golden_PINN_MF_LOCO"
    
    print(f"\n🚀 Lanzando Experimento Definitivo: {run_name}")
    print(f"   -> Adam Epochs: {epochs_adam} | L-BFGS Epochs: {epochs_lbfgs}")
    print(f"   -> lambda_sat: 15.0 (Sat_Fuerte ha demostrado ser el mejor regulador)")
    
    try:
        train_pinn(
            epochs=epochs_adam,
            batch_size=2048,
            lr=5e-4,
            curriculum_epochs=2000,
            colloc_ratio=4,
            lambda_sat=15.0, # Según nuestro análisis, Sat_Fuerte ganó
            lbfgs_epochs=epochs_lbfgs,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            run_name=run_name
        )
        print(f"✅ Experimento {run_name} finalizado con éxito.")
    except Exception as e:
        print(f"❌ Error en {run_name}: {str(e)}")

if __name__ == "__main__":
    run_golden_training()
