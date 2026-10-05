import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.experiment_harness import train_pinn

"""
Batería para el servidor (GPU grande) -- rediseñada el 2026-09-25 tras la revisión
de resultados (propuesta_integracion_datos_pinn.md, Secciones 10 y 11).

Antes de lanzar (en el servidor):
    git pull origin main
    python src/data_ingestion/compute_vertical_velocity_v2.py data/raw/cmems_yearly   # w corregida (v2)
    python src/data_ingestion/build_dataset.py     # regenera parquet + ocean_colloc.parquet + tabla de cobertura
  -> MIRAR la tabla "Cobertura REAL de forzantes por año": u,v,w,T deben estar ~1.0 en 1998-2012.

Uso:
    python run_battery_a100.py screen            # cribado: 7 configuraciones en el fold A
    python run_battery_a100.py folds S3_prior_seasonal S5_prior_seasonal_sinfisica   # confirmar en folds B y C

Protocolo: val elige checkpoint/para (patience); test se evalúa UNA vez con el modelo elegido; cada run
registra baselines (media y climatología de train) y skill frente a ellos. Un modelo solo "funciona" si su
skill_clim en TEST es > 0 de forma consistente en varios folds; elegir por val, reportar test.
"""

BASE = dict(epochs=400, batch_size=65536, lr=2e-3, curriculum_epochs=100, colloc_ratio=0.5,
            ocean_colloc_ratio=1.0, lambda_sat=1.0, lambda_dirichlet=1.0, lbfgs_epochs=0,
            num_layers=6, hidden_dim=128, lr_schedule="cosine", patience=60)

CONFIGS = {
    # arquitectura histórica, pero con física corregida + protocolo nuevo (referencia)
    "S0_base":                     dict(),
    "S1_seasonal":                 dict(use_seasonal=True),
    "S2_prior":                    dict(use_climatology_prior=True),
    "S3_prior_seasonal":           dict(use_climatology_prior=True, use_seasonal=True),
    "S4_prior_seasonal_fourier":   dict(use_climatology_prior=True, use_seasonal=True, use_fourier_features=True),
    # ablaciones de la física: ¿aporta algo la PDE al val/test?
    "S5_prior_seasonal_sinfisica": dict(use_climatology_prior=True, use_seasonal=True, lambda_phys_max=0.0),
    "S6_prior_seasonal_fisica10":  dict(use_climatology_prior=True, use_seasonal=True, lambda_phys_max=10.0),
}
DATE_TAG = "20260925"


def run(name, fold):
    kw = {**BASE, **CONFIGS[name]}
    run_name = f"{name}_fold{fold}_{DATE_TAG}"
    print(f"\n{'='*70}\n {run_name}\n{'='*70}")
    t0 = time.time()
    try:
        s = train_pinn(run_name=run_name, fold=fold, **kw)
        print(f"OK {run_name} en {(time.time()-t0)/60:.1f} min | test skill_clim={s['test']['skill_clim']:+.3f}")
    except Exception as e:
        print(f"ERROR en {run_name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if mode == "screen":
        for name in CONFIGS:
            run(name, "A")
    elif mode == "folds":
        names = sys.argv[2:] or ["S3_prior_seasonal"]
        for name in names:
            for fold in ("B", "C"):
                run(name, fold)
    else:
        raise SystemExit("modo desconocido: usa 'screen' o 'folds <configs...>'")
    print("\nHecho. Resumen: python summarize_results.py")
