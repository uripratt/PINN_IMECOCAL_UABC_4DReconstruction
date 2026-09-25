import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.experiment_harness import train_pinn

"""
Batería DIAGNÓSTICA (no de producción) -- 2026-09-15.

Objetivo: con el pipeline ya corregido en la Fase 0 (dataset calibrado
chl_unified_v1.csv, peso_entrenamiento real, Dirichlet real, lambda_sat
corregido -- commit 12ebd94), aislar en una GPU local (RTX 3050 Ti, 4GB) y
en un presupuesto de épocas reducido (~80-100, no las 3.000-10.000 de un
run de producción) dos preguntas concretas que quedaron abiertas en
propuesta_integracion_datos_pinn.md, Sección 10:

1. ¿El esquema híbrido Adam+L-BFGS ("Golden Run") sigue sobreajustando
   ahora que los datos/pesos están corregidos, o el sobreajuste severo
   observado en agosto (Val_Loss nunca bajaba de ~0.12-0.13 en los runs
   Gold_Sat_*) era principalmente un problema de los datos sin calibrar?
2. ¿El embedding de Fourier anisotrópico (nuevo, src/models/pinn_model.py)
   cambia algo perceptible en la curva de pérdida incluso a esta escala
   reducida?

Esto NO es el entrenamiento de referencia final -- es una señal rápida
para decidir con qué configuración lanzar la batería real en el servidor
A100 (ver run_battery_a100.py). Arquitectura completa (6x128) para que la
comparación sea representativa; batch_size más pequeño que en producción
(8192, no 65536) por límite de VRAM local.
"""

BATCH_SIZE = 8192
COLLOC_RATIO = 4
NUM_LAYERS = 6
HIDDEN_DIM = 128
LR = 2e-3
LAMBDA_SAT = 1.0
LAMBDA_DIRICHLET = 1.0
DATE_TAG = "20260915"

configs = [
    {
        "run_name": f"Diag_AdamOnly_{DATE_TAG}",
        "epochs": 90, "curriculum_epochs": 45, "lbfgs_epochs": 0,
        "use_fourier_features": False,
    },
    {
        "run_name": f"Diag_AdamLBFGS_{DATE_TAG}",
        "epochs": 75, "curriculum_epochs": 45, "lbfgs_epochs": 15,
        "use_fourier_features": False,
    },
    {
        "run_name": f"Diag_Fourier_AdamOnly_{DATE_TAG}",
        "epochs": 90, "curriculum_epochs": 45, "lbfgs_epochs": 0,
        "use_fourier_features": True,
    },
]

if __name__ == "__main__":
    print("=" * 60)
    print(" BATERÍA DIAGNÓSTICA LOCAL (reducida, no es el run final)")
    print("=" * 60)
    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Lanzando: {cfg['run_name']}")
        t0 = time.time()
        try:
            train_pinn(
                epochs=cfg["epochs"],
                batch_size=BATCH_SIZE,
                lr=LR,
                curriculum_epochs=cfg["curriculum_epochs"],
                colloc_ratio=COLLOC_RATIO,
                lambda_sat=LAMBDA_SAT,
                lambda_dirichlet=LAMBDA_DIRICHLET,
                lbfgs_epochs=cfg["lbfgs_epochs"],
                num_layers=NUM_LAYERS,
                hidden_dim=HIDDEN_DIM,
                run_name=cfg["run_name"],
                use_fourier_features=cfg["use_fourier_features"],
            )
            print(f"✅ {cfg['run_name']} completado en {(time.time()-t0)/60:.1f} min")
        except Exception as e:
            print(f"❌ Error en {cfg['run_name']}: {e}")
    print("\nBatería diagnóstica local completada.")
