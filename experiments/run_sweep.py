import os
import sys
import tarfile
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.experiment_harness import train_pinn

def run_battery():
    # Definimos la grilla de hiperparámetros a explorar.
    # 3000 epochs es suficiente para ver la tendencia de estabilización post-curriculum (paso 1500)
    configs = [
        # 1. Baseline conservador
        {"name": "LR_Bajo_Mucha_Fisica", "lr": 5e-4, "colloc_ratio": 6, "batch_size": 512, "epochs": 3000, "curriculum": 1500, "lbfgs": 200},
        
        # 2. Batch Gigante: Estabiliza el gradiente al promediar más puntos a la vez (si la VRAM lo permite).
        {"name": "Batch_Gigante_Rapido", "lr": 1e-3, "colloc_ratio": 2, "batch_size": 2048, "epochs": 3000, "curriculum": 1500, "lbfgs": 200},
        
        # 3. Cirugía Fina: LR muy bajo, ratio estándar. Super suave pero lento.
        {"name": "LR_Micro_Suave", "lr": 1e-4, "colloc_ratio": 4, "batch_size": 1024, "epochs": 3000, "curriculum": 1500, "lbfgs": 200},
        
        # 4. Física Agresiva Temprana: Curriculum rápido para ver si fuerza la estabilización antes.
        {"name": "Fisica_Rapida", "lr": 5e-4, "colloc_ratio": 4, "batch_size": 1024, "epochs": 3000, "curriculum": 500, "lbfgs": 200},
    ]
    
    print(f"Iniciando batería de {len(configs)} entrenamientos exploratorios...")
    
    for i, cfg in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Experimento {i+1}/{len(configs)}: {cfg['name']}")
        print(f"Parámetros: LR={cfg['lr']}, Colloc={cfg['colloc_ratio']}x, Batch={cfg['batch_size']}, Curriculum={cfg['curriculum']}, L-BFGS={cfg['lbfgs']}")
        print(f"{'='*60}")
        
        # Limpiar memoria de la GPU entre entrenamientos para evitar OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        train_pinn(
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            curriculum_epochs=cfg["curriculum"],
            colloc_ratio=cfg["colloc_ratio"],
            lambda_sat=10.0,
            lbfgs_epochs=cfg["lbfgs"],
            run_name=cfg["name"]
        )
        time.sleep(3) # Pausa para asegurar escritura de MLflow
        
    print("\nEmpaquetando resultados de MLflow en un archivo comprimido...")
    mlruns_dir = os.path.join(os.path.dirname(__file__), 'mlruns')
    tar_path = os.path.join(os.path.dirname(__file__), 'sweep_mlruns.tar.gz')
    
    if os.path.exists(mlruns_dir):
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(mlruns_dir, arcname="mlruns")
        print(f"\n¡Batería completada con éxito!")
        print(f"Por favor, descarga el archivo: {tar_path} y compártelo para el análisis.")
    else:
        print("\n¡Batería completada con éxito!")
        print("MLflow logging fue remoto o no se guardó localmente. No hay archivo para comprimir.")

if __name__ == "__main__":
    run_battery()
