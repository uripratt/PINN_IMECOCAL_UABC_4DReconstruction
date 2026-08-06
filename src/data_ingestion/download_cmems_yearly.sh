#!/bin/bash

# Script robusto para descargar covariables de Copernicus (CMEMS) año por año
# Esto evita que el sistema aborte (Killed/OOM) por intentar procesar 15 años de datos diarios de golpe.

OUT_DIR="data/raw/cmems_yearly"
mkdir -p $OUT_DIR

echo "Iniciando descarga particionada por años de las corrientes uo, vo (1998-2012)..."

for year in {1998..2012}; do
    FILE_NAME="cmems_currents_${year}.nc"
    if [ -f "${OUT_DIR}/${FILE_NAME}" ]; then
        echo "=> El año $year ya está descargado. Saltando..."
        continue
    fi
    
    echo "----------------------------------------"
    echo "Descargando año $year..."
    
    copernicusmarine subset -i cmems_mod_glo_phy_my_0.083deg_P1D-m \
      -x -119.85 -X -111.92 -y 23.82 -Y 32.75 \
      -t ${year}-01-01 -T ${year}-12-31 \
      -v uo -v vo \
      -o $OUT_DIR -f $FILE_NAME \
      --force-download
      
    if [ $? -eq 0 ]; then
        echo "✅ Año $year completado con éxito."
    else
        echo "❌ Error descargando el año $year. Reintentando en la próxima ejecución."
    fi
done

echo "----------------------------------------"
echo "¡Proceso de descarga multianual finalizado!"
