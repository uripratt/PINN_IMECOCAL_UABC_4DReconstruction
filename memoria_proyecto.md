# Memoria de Investigación: Reconstrucción 4D de Clorofila-a (IMECOCAL)
*Documento de registro de decisiones, hallazgos y evolución científica del proyecto.*

## Fase 0: Definición Arquitectónica y Revisión del Estado del Arte
**Fecha:** 05 de Agosto, 2026
**Agente Responsable:** Agente 0 (Literature & Architecture Researcher)

### Hallazgos
*   El análisis descriptivo exploratorio de los perfiles de IMECOCAL (1998-2012) mostró un fuerte gradiente costa-océano y variabilidad temporal modulada por ciclos ENOS. Sin embargo, los datos presentan grandes vacíos ("gaps") espaciales y temporales que los métodos tradicionales de interpolación (ej. Kriging) no pueden resolver de manera físicamente consistente debido a la naturaleza dinámica de las surgencias costeras y remolinos.
*   La literatura actual en Oceanografía de Machine Learning apunta a dos arquitecturas de vanguardia para datos *mesh-free* (sin cuadrícula fija):
    1.  **PINNs (Physics-Informed Neural Networks)**: Ideales para campos continuos y diferenciación exacta.
    2.  **PI-GNNs (Physics-Informed Graph Neural Networks)**: Estado del arte (2024-2025) para capturar teleconexiones a larga distancia y topologías de muestreo irregulares.

### Decisión Arquitectónica
*   **Estrategia Secuencial Acordada:**
    *   **Paso 1:** Comenzar desarrollando una **PINN Clásica** acoplada a la Ecuación de Advección-Difusión-Reacción. Esto permitirá establecer el *Test Harness* físico (los operadores de PyTorch Autograd) y validar la viabilidad del campo de velocidades sin añadir complejidad topológica.
    *   **Paso 2:** Una vez validada la PINN 4D y obtenidos los primeros resultados, pivotar y elevar la arquitectura hacia una **PI-GNN (Opción B)**, utilizando el grafo de estaciones para modelar de manera explícita el transporte entre las mismas.

---

## Fase 1: Ingeniería de Datos (Infraestructura Espaciotemporal)
**Estado:** En curso
**Agente Responsable:** Agente 1 (Data Engineer)

### Objetivos Inmediatos
1.  **Formatear Datos IMECOCAL:** Estructurar los datos interpolados de clorofila-$a$ (0, 10, 20, 50, 100 m) en tensores de entrada $(x, y, z, t)$.
2.  **Infraestructura ERDDAP:** Implementar las rutinas de descarga automatizada para obtener las covariables físicas de forzamiento continuo correspondientes al dominio y fechas de IMECOCAL:
    *   **Velocidades oceánicas 3D ($u, v, w$):** Provenientes de reanálisis (ej. Copernicus Marine CMEMS / GLORYS12).
    *   **SST y Anomalías (SLA):** Para forzamientos en superficie.
    *   **Batimetría (ETOPO1):** Para imponer condiciones de frontera de fondo físico en la PINN.

### Hallazgos y Ejecución
*   **05 de Agosto, 2026:**
    *   Se creó la estructura de directorios del *Harness* (`src/`, `data/`, `tests/`, `evaluation/`, `experiments/`).
    *   Se desarrolló `src/data_ingestion/erddap_downloader.py`. Este script detecta automáticamente la caja delimitadora (Bounding Box) del dominio IMECOCAL leyendo el archivo Excel original.
    *   **Dominio Detectado:** Latitud [23.82, 32.76], Longitud [-119.85, -111.92], Periodo [1998 - 2012].
    *   Se descargó exitosamente la **batimetría ETOPO1** de NOAA CoastWatch ERDDAP y se guardó en `data/raw/etopo_bathymetry.nc` (Tamaño: 537x477 píxeles, ~1MB).
    *   *Nota sobre las Corrientes 3D:* Para el dataset de reanálisis CMEMS (GLORYS12V1), se definió que la mejor ruta será usar el paquete `copernicusmarine` con el comando CLI generado dinámicamente en el script (debido a la necesidad de tokens de autenticación).
    *   Se implementó y verificó el script `src/data_ingestion/dataloader.py` (Clase `CoastalPINNDataset`). Este script ingesta exitosamente el dataset bruto en formato Excel (`Cl_Imec98_12.xlsx`), consolida el formato temporal en días continuos, y estructura `16,239` muestras espaciotemporales en tensores PyTorch `X (lat, lon, prof, time)` y `y (clorofila)`, listos para alimentar la función de pérdida empírica (Data Loss) de la PINN.

---

## Fase 3: Desarrollo Físico y Neuronal
**Estado:** Completado
**Agente Responsable:** Agente 2 (Physical Modeler) & Agente 3 (NN Architect) & Agente 4 (Validator)

### Hallazgos y Ejecución
*   **05 de Agosto, 2026:**
    *   **Agente 2 (Physical Modeler):** Se codificó `src/physics/physics_loss.py`, implementando el cálculo de la Ecuación en Derivadas Parciales (PDE) de Advección-Difusión-Reacción oceánica utilizando `torch.autograd`. El operador de pérdida física penaliza las violaciones al balance entre el transporte por corrientes continuas de CMEMS ($u, v$), la difusión turbulenta, y el decaimiento biogeoquímico.
    *   **Agente 3 (NN Architect):** Se implementó la arquitectura fundacional `src/models/pinn_model.py` (Multi-Layer Perceptron), con 4 entradas espaciotemporales ($lat, lon, prof, t$) y 1 salida (Clorofila). Se añadió una función de activación `Softplus` en la capa de salida para garantizar físicamente que las predicciones de clorofila siempre sean estrictamente positivas.
    *   **Agente 4 (Validator):** Se configuró la suite de validación `tests/test_harness.py`. Se comprobó exitosamente, mediante pruebas unitarias (`unittest`), la integridad matemática del flujo de gradientes (Graph Retention para Backpropagation), las restricciones de positividad, y el formato de los tensores de salida.

---

## Fase 4: Experimentación y Reconstrucción 4D
**Estado:** En curso
**Agente Responsable:** Agente 3 (NN Architect) & Agente 4 (Evaluation & Validator)

### Hallazgos y Ejecución
*   **05 de Agosto, 2026:**
    *   Se implementó el `Experiment Harness` (`experiments/experiment_harness.py`).
    *   Se automatizó el registro de métricas y artefactos mediante **MLflow**.
    *   Se incorporó con éxito **Curriculum Learning**, comenzando el entrenamiento puramente sobre observaciones empíricas (`Data Loss`) e incrementando el peso de la pérdida física (`Physics Loss`) progresivamente de `lambda = 0` a `0.1`.
    *   El modelo fundacional de PINN ha sido entrenado de manera base y exportado como `pinn_model_final.pth`, cerrando el bucle de datos desde ERDDAP hasta la red neuronal.

### Actualización de Cierre de Jornada (05 de Agosto)
*   **Finalización CMEMS:** Se completó la descarga del histórico completo de corrientes (1998-2012, 15 archivos NetCDF, ~15 GB totales).
*   **OOM-Safe Ingestion:** Para evitar el colapso de memoria RAM de `open_mfdataset`, se construyó el preprocesador `src/data_ingestion/build_dataset.py`, iterando los CMEMS anualmente mediante Dask (`sel(method='nearest')`) y exportando el híbrido continuo a `imecocal_augmented.csv`.
*   **Validación de Entrenamiento:** El `dataloader.py` reformado lee la matriz en < 1 segundo. El `experiment_harness.py` probó con éxito la evaluación autograd sobre el dataset de 16,239 observaciones combinadas.
*   **Manual Teórico:** Se culminó la versión final de 17 páginas (`Plan_Estudio_Oceanografia.pdf`), desarrollando analíticamente el problema inverso (4D-Var), aproximación de Boussinesq, Autograd y Grafos Dinámicos.

### Próximos Pasos (Hoja de Ruta)
1. **Migración a Servidor (GitHub):** Crear repositorio, configurar `.gitignore` para omitir datos pesados (NetCDF/Excel) y trasladar el código base a una instancia con GPU > 24GB VRAM (para L-BFGS) y +64GB RAM.
2. **Ajustes Físicos (Dirichlet):** Implementar condiciones de frontera en `physics_loss.py` para anular la clorofila sobre tierra firme (batimetría > 0).
3. **Entrenamiento Continuo Gold:** Entrenar el modelo final aplicando la rampa de Curriculum Learning (Adam -> L-BFGS) para las variables hidrodinámicas de CMEMS.
4. **Sinergia Satelital (Hito Futuro):** Integrar datos superficiales de satélite (Ocean Color / SST) a $z=0$ en el Dataloader para anclar la superficie y usar la física de la PINN para propagar esa alta resolución hacia la columna de agua profunda de IMECOCAL.
5. **Inferencia 4D:** Ejecutar `plot_inference.py` para materializar las visualizaciones del campo reconstruido final.
