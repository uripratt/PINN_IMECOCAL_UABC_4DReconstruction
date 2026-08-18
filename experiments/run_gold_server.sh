#!/bin/bash
# Script para lanzar la corrida final (Golden Run) en el servidor de forma segura

echo "Iniciando Golden Run en el servidor..."
echo "Los logs se guardarán en: experiments/logs_Server/golden_run.log"

# Crear directorio de logs si no existe
mkdir -p experiments/logs_Server/

# Lanzar con nohup para que sobreviva a desconexiones SSH
nohup python3 experiments/run_gold.py > experiments/logs_Server/golden_run.log 2>&1 &

# Obtener y mostrar el PID
PID=$!
echo "✅ Proceso lanzado exitosamente en segundo plano."
echo "PID del proceso: $PID"
echo "Para ver el progreso en tiempo real, ejecuta:"
echo "tail -f experiments/logs_Server/golden_run.log"
