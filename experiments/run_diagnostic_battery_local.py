import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.experiment_harness import train_pinn

"""
Mini-cribado LOCAL (2026-09-25): arquitectura reducida (4x64) y pocas épocas, en la GPU del
portátil, para orientar las ablaciones antes de gastar horas de servidor (run_battery_a100.py).
NO es un resultado final: modelo pequeño y entrenamiento corto. Mide, con el protocolo
completo (val≠test, baselines, skill), si el prior climatológico, la estacionalidad y la
física (PDE) aportan algo.

Uso (enchufado a la corriente; usa MLflow local si no hay red):
    env -u MLFLOW_TRACKING_URI systemd-inhibit --what=sleep:idle -- python3 run_diagnostic_battery_local.py
"""
BASE = dict(epochs=30, batch_size=32768, lr=2e-3, curriculum_epochs=10, colloc_ratio=0.25, ocean_colloc_ratio=0.5,
            lambda_sat=1.0, lambda_dirichlet=1.0, num_layers=4, hidden_dim=64, lr_schedule="cosine",
            patience=None, fold="A")
TAG = "20260925"
CONFIGS = {
    "M0_base":                    dict(),
    "M1_prior_seasonal":          dict(use_climatology_prior=True, use_seasonal=True),
    "M2_prior_seasonal_sinfis":   dict(use_climatology_prior=True, use_seasonal=True, lambda_phys_max=0.0),
    "M3_prior_seasonal_fis10":    dict(use_climatology_prior=True, use_seasonal=True, lambda_phys_max=10.0),
}

if __name__ == "__main__":
    for name, extra in CONFIGS.items():
        t0 = time.time()
        try:
            s = train_pinn(run_name=f"{name}_{TAG}", **{**BASE, **extra})
            print(f"OK {name} en {(time.time()-t0)/60:.1f} min | val skill={s['val']['skill_clim']:+.3f} "
                  f"test skill={s['test']['skill_clim']:+.3f}")
        except Exception as e:
            print(f"ERROR en {name}: {type(e).__name__}: {e}")
    print("Mini-cribado completado. Resumen: python summarize_results.py 'results_M*_20260925.json'")
