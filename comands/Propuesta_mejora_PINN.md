Esta es una propuesta para mejorar el proyecto (o crear uno nuevo). 

El enfoque que s está planteando se encuentra en la vanguardia de la investigación en oceanografía física y biogeoquímica. Pasar de perfiles verticales discretos e irregulares (obtenidos por CTD, rosetas, gliders o BGC-Argo) a campos continuos 4D (x,y,z,t) preservando la coherencia física es uno de los mayores desafíos actuales del Big Data marino.

Para mi doctorado, este tema es excelente: combina una necesidad científica real con una oportunidad metodológica clara.

## 1. Estado del Arte: ¿Qué se ha hecho hasta ahora?
A. PINNs en Oceanografía Física (Temperatura, Salinidad, Velocidades)

    Reconstrucción 2D/3D de dinámica marina: Se han aplicado PINNs para resolver ecuaciones de advección-difusión y fluidos (Navier-Stokes/Ecuaciones Primitivas) para interpolar velocidades subsuperficiales a partir de drifters o reconstruir campos continuos de temperatura y salinidad (T,S) combinando observaciones in-situ y datos satelitales (ej. MDRF-Net, 2024; PINN-SR, 2025).  

    Difusión Térmica Vertical: Existen trabajos recientes que usan PINNs 1D/2D para resolver la ecuación de difusión de calor en la columna de agua con datos de flotadores Argo, aprendiendo la difusividad térmica vertical (κ).  

B. Reconstrucción 3D/4D de Biogeoquímica (Clorofila, O2​, PAR)

    Modelos puramente Data-Driven (MLP, Random Forest, Transformers): La inmensa mayoría de la literatura sobre reconstrucción 3D de Clorofila, Oxígeno o Nutrientes a partir de satélite y perfiles in-situ (como el modelo de Sammartino et al., 2020 para el Mediterráneo o algoritmos como CANYON-B / SOCA) se basan en machine learning empírico clásico (MLPs o redes convolucionales).

    El problema de las redes puramente data-driven: Generan mapas suaves, pero no respetan leyes de conservación (ej. balance de oxígeno, conservación de masa) y suelen fallar en eventos extremos o en capas profundas donde el satélite no tiene visión directa.  

## 2. Los Gaps de Conocimiento para tu Tesis Doctoral

Aquí es donde tu investigación puede hacer aportaciones novedosas de alto impacto:
Gap 1: Formulación de la Loss Física Acoplada Bio-Física (Coupled Bio-Physical PDEs)

    El problema actual: Las PINNs en oceanografía se limitan casi exclusivamente a la física pura (T,S,u,v,w). Variables como la Clorofila, el PAR y el Oxígeno Disuelto no se rigen solo por transporte fluido, sino por reacciones acopladas (Advección-Difusión-Reacción + Óptica):  

        PAR (Radiación Fotosintéticamente Activa): Obedece la Ley de Beer-Lambert acoplada a la atenuación por clorofila:
        ∂z∂PAR​=−Kd​(C)⋅PAR

        Clorofila/Biomasa (C): Sigue una EAE (Ecuación de Advección-Difusión) con término fuente biológico guiado por el PAR y la temperatura:
        ∂t∂C​+u⋅∇C=Kv​∂z2∂2C​+μ(PAR,T,N)C−mC

        Oxígeno (O2​): Depende de la tasa de fotosíntesis (producción), respiración y solubilidad en función de T y S.

    Tu aporte: Diseñar una PINN multivariable con pérdidas acopladas. Casi no hay literatura que combine en la función de pérdida (loss function) las ecuaciones físicas junto con las ecuaciones de luz (PAR) y de balance biogeoquímico.  

Gap 2: Asimetría Espaciotemporal en Representaciones Implícitas (Sparse 1D Profiles → Continuous 4D Field)

    El problema actual: Los perfiles verticales de roseta o gliders tienen una risolución muy alta en el eje z (centímetros/metros), pero son sumamente dispersos en x,y,t (puntos discontinuos en el espacio-tiempo). Las PINNs convencionales sufren de desequilibrio de gradientes cuando la densidad de datos varía drásticamente entre dimensiones.

    Tu aporte: Proponer una arquitectura de Representación Neural Implícita (INR / Coordinada) optimizada para grillas asimétricas. Puedes abordar cómo ponderar dinámicamente las pérdidas de datos en la vertical frente a la regularización física en el plano horizontal y temporal.

Gap 3: Corrección del "Ciego Subsuperficial" (DCM y Oxiclina)

    El problema actual: La superficie de la clorofila (observada por satélite) rara vez coincide en magnitud ni ubicación con el Máximo Profundo de Clorofila (DCM) o la Zona de Mínimo de Oxígeno (OMZ).

    Tu aporte: Demostrar cuantitativamente que una PINN (guiada por restricciones de atenuación de luz y estratificación térmica) logra reconstruir la profundidad y forma del DCM y la oxiclina mucho mejor que los métodos de interpolación estocástica (Kriging, OI) y los modelos de deep learning empíricos.

## 3. Hoja de Ruta Sugerida para la Metodología

[Perfiles Discretos (CTD, BGC)] + [Campos Superficiales (Satélite: SST, Chl)]
                                │
                                ▼
         ┌─────────────────────────────────────────────┐
         │       PINN Multivariable (u,v,w,T,S,Chl,O2) │
         ├─────────────────────────────────────────────┤
         │  Loss_data: MSE en datos puntuales          │
         │  Loss_physics:                              │
         │   • PDE Térmica / Salina (Advección/Difusión)│
         │   • PDE Óptica (Atenuación PAR / Beer-Lambert)│
         │   • PDE Oxígeno (Producción / Respiración)  │
         └─────────────────────────────────────────────┘
                                │
                                ▼
               [Campo Continuo 4D (x, y, z, t)]

    Fase 1 (Sintética / Twin Experiment): Valida la red con salidas de un modelo numérico acoplado (como NEMO-PISCES o ROMS-FENN3L). Esto te permitirá medir el error real del campo 4D reconstruido frente a un ground truth continuo.

    Fase 2 (Datos Reales): Entrena la PINN usando tus perfiles in-situ (T,S,Chl,PAR,O2​) utilizando los datos de satélite (SST, Chl superficial, altimetría) como condiciones de borde en la superficie (z=0).

    Fase 3 (Validación Espaciotemporal Estricta - Post Baseline):
    Una vez que el modelo logre minimizar la función de pérdida con un split aleatorio (15% random point hold-out), se debe transicionar a métricas de generalización oceánica real para evitar el *Data Leakage* por interpolación vertical:
    - **Leave-One-Cruise-Out (LOCO):** Ocultar campañas oceanográficas enteras (ej. ocultar Primavera 2005). Demuestra la capacidad predictiva temporal (Gemelo Digital) propagando el estado del océano a través del tiempo.

    Fase 4 (Iteraciones Futuras Post-Estabilización):
    - **Escalado de Capacidad:** Una vez controlados los gradientes espurios (overfitting) mediante la transformación logarítmica, revertir la arquitectura a `6 capas x 128 neuronas` para comprobar si una mayor capacidad permite aprender texturas sub-mesoscalares y frentes más precisos (similares a los del satélite).
    - **Optimizadores Avanzados (PCGrad):** Dado que la Loss de Satélite, la Loss de Datos in-situ y la Physics Loss a menudo compiten en direcciones opuestas, sustituir Adam por **PCGrad** (Projected Conflicting Gradients). PCGrad es el nuevo estado del arte en PINNs oceánicas, proyectando gradientes conflictivos ortogonalmente para evitar que la física destruya el ajuste a los datos (y viceversa).
