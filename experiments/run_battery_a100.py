import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.experiment_harness import train_pinn

"""
Batería de PRODUCCIÓN para el servidor A100 -- 2026-09-15.

Primera batería que entrena con el pipeline ya corregido en la Fase 0
(commit 12ebd94: chl_unified_v1.csv calibrado, peso_entrenamiento real,
Dirichlet real, lambda_sat corregido). Ninguno de los 41 runs anteriores
en MLflow (incluidos los 8 "Gold_Sat_*"/"_A100" de agosto) usó este
pipeline -- ver propuesta_integracion_datos_pinn.md, Sección 10.

Diseño de las 3 configuraciones, justificado por el diagnóstico de esa
misma sección (no elegidas a ciegas):

- A100_AdamOnly_10k: reproduce el esquema que mejor generalizó en el
  histórico (LogPINN_Sat_Medio_LRLento, Val_Loss mínimo 0.0894, Adam puro
  10.000 épocas) -- ahora sobre datos calibrados. Es la referencia.
- A100_AdamLBFGS_Golden: reproduce el esquema "Golden Run" (Adam 3.000 +
  L-BFGS 500) que en agosto sobreajustó severamente (Val_Loss nunca bajó
  de ~0.12-0.13 incluso en su mejor época) -- para aislar si ese
  sobreajuste era por los datos sin calibrar (en cuyo caso debería
  mejorar mucho aquí) o por el propio esquema de optimización (en cuyo
  caso seguirá sobreajustando incluso con datos buenos).
- A100_AdamOnly_Fourier_10k: A100_AdamOnly_10k + el nuevo embedding de
  Fourier anisotrópico (src/models/pinn_model.py, use_fourier_features),
  para probar si ataca el problema, ya diagnosticado por separado, de
  pérdida de estructura de mesoescala en las reconstrucciones (Sección
  "Resultados" del reporte, Gap 2).

Antes de correr esto: regenerar imecocal_augmented.parquet con
`python src/data_ingestion/build_dataset.py` (ya corregido en la Fase 0,
pero el .parquet en disco del servidor puede seguir siendo el antiguo).

Tiempo esperado: del orden de las ~3.6h que tomó cada run de 10.000
épocas en el histórico (LogPINN_Sat_Medio_LRLento) -- unas ~11h para las
3 configuraciones en serie. Ajustar batch_size/lr si la VRAM del servidor
difiere del run que fijó estos valores (a2b9f28, "Optimizar batch size y
LR para GPU A100").
"""

BATCH_SIZE = 65536
COLLOC_RATIO = 4
NUM_LAYERS = 6
HIDDEN_DIM = 128
LR = 2e-3
LAMBDA_SAT = 1.0
LAMBDA_DIRICHLET = 1.0
CURRICULUM_EPOCHS = 2000
DATE_TAG = "20260915"

configs = [
    {
        "run_name": f"A100_AdamOnly_10k_{DATE_TAG}",
        "epochs": 10000, "lbfgs_epochs": 0,
        "use_fourier_features": False,
    },
    {
        "run_name": f"A100_AdamLBFGS_Golden_{DATE_TAG}",
        "epochs": 3000, "lbfgs_epochs": 500,
        "use_fourier_features": False,
    },
    {
        "run_name": f"A100_AdamOnly_Fourier_10k_{DATE_TAG}",
        "epochs": 10000, "lbfgs_epochs": 0,
        "use_fourier_features": True,
    },
]

if __name__ == "__main__":
    print("=" * 60)
    print(" BATERÍA DE PRODUCCIÓN A100 (pipeline Fase 0, primera vez)")
    print("=" * 60)
    total = len(configs)
    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{total}] 🚀 Lanzando: {cfg['run_name']}")
        try:
            train_pinn(
                epochs=cfg["epochs"],
                batch_size=BATCH_SIZE,
                lr=LR,
                curriculum_epochs=CURRICULUM_EPOCHS,
                colloc_ratio=COLLOC_RATIO,
                lambda_sat=LAMBDA_SAT,
                lambda_dirichlet=LAMBDA_DIRICHLET,
                lbfgs_epochs=cfg["lbfgs_epochs"],
                num_layers=NUM_LAYERS,
                hidden_dim=HIDDEN_DIM,
                run_name=cfg["run_name"],
                use_fourier_features=cfg["use_fourier_features"],
            )
            print(f"✅ {cfg['run_name']} finalizado con éxito.")
        except Exception as e:
            print(f"❌ Error en {cfg['run_name']}: {str(e)}")

    print("\nBatería A100 completada. Comparar Val_Loss (mínimo por época,")
    print("no solo el valor final) entre las 3 configuraciones en MLflow")
    print("antes de elegir cuál usar como referencia de producción.")
