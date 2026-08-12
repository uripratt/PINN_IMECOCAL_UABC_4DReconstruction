#!/bin/bash

# Este script automatiza todo el pipeline de Data Engineering y Entrenamiento
# para ejecutarse de forma desatendida en el servidor.

# 1. Limpieza de archivos corruptos previos
echo "============================================================"
echo "Limpiando cachés y velocidades verticales corruptas antiguas..."
echo "============================================================"
rm -f data/raw/cmems_yearly/*_with_w.nc

# 2. Re-calcular Upwelling (Bucle Día a Día a prueba de RAM)
echo "============================================================"
echo "PASO 1: Calculando campos de Velocidad Vertical (wo)..."
echo "============================================================"
python3 compute_vertical_velocity.py
if [ $? -ne 0 ]; then
    echo "Error en compute_vertical_velocity.py. Abortando pipeline."
    exit 1
fi

# 3. Ensamblar Dataset (Fusión de CMEMS y Satellite CHL)
echo "============================================================"
echo "PASO 2: Construyendo imecocal_augmented.csv..."
echo "============================================================"
python3 src/data_ingestion/build_dataset.py
if [ $? -ne 0 ]; then
    echo "Error en build_dataset.py. Abortando pipeline."
    exit 1
fi

# 4. Entrenamiento de la Arquitectura SOTA (PINN 4D)
echo "============================================================"
echo "PASO 3: Iniciando Entrenamiento PINN + Física Inversa..."
echo "============================================================"
python3 experiments/experiment_harness.py

echo "============================================================"
echo "¡PIPELINE COMPLETADO CON ÉXITO!"
echo "Revisa la interfaz de MLflow para ver las gráficas y métricas."
echo "============================================================"
