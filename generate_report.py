import pandas as pd
import numpy as np

# Load data
df = pd.read_excel('Cl_Imec98_12.xlsx')

# Calculate advanced stats
chl = df['Clorofila']
prof = df['Profundidad']

stats = {
    'total_samples': len(df),
    'years': len(df['Año'].unique()),
    'stations': len(df.groupby(['Latitud', 'Longitud'])),
    'chl_mean': chl.mean(),
    'chl_median': chl.median(),
    'chl_std': chl.std(),
    'chl_var': chl.var(),
    'chl_max': chl.max(),
    'chl_p90': chl.quantile(0.90),
    'chl_p95': chl.quantile(0.95),
    'chl_p99': chl.quantile(0.99),
    'prof_mean': prof.mean(),
    'prof_median': prof.median(),
    'prof_std': prof.std(),
    'prof_max': prof.max(),
    'corr_chl_prof': chl.corr(prof)
}

html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Reporte Técnico: Datos IMECOCAL (1998 - 2012)</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th, td {{
            padding: 10px;
            border: 1px solid #ddd;
            text-align: left;
        }}
        th {{
            background-color: #f4f6f7;
            font-weight: bold;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .image-container img {{
            max-width: 100%;
            border: 1px solid #ddd;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }}
        .highlight {{
            background-color: #f9f9f9;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 20px 0;
        }}
    </style>
</head>
<body>

    <h1>Reporte Técnico y Estadístico: Proyecto IMECOCAL (1998 - 2012)</h1>

    <h2>1. Resumen Espacio-Temporal del Muestreo</h2>
    <p>El conjunto de datos comprende información oceanográfica recopilada a lo largo de la península de Baja California. La base de datos contiene:</p>
    <ul>
        <li><strong>Total de Observaciones:</strong> {stats['total_samples']:,} perfiles discretos</li>
        <li><strong>Periodo:</strong> {stats['years']} años continuos (1998 - 2012)</li>
        <li><strong>Estaciones Únicas Muestreadas:</strong> {stats['stations']}</li>
    </ul>

    <h2>2. Distribución Espacial y Batimétrica</h2>
    <div class="image-container">
        <!-- Using absolute path or relative path, weasyprint handles it -->
        <img src="mapa_estaciones.png" alt="Mapa de Estaciones IMECOCAL" />
        <p><em>Figura 1: Mapa espacial de estaciones superpuesto a la batimetría ETOPO1 (isobáticas cada 500m).</em></p>
    </div>

    <h2>3. Análisis Estadístico Detallado</h2>
    
    <h3>3.1. Variable Objetivo: Clorofila (mg/m³)</h3>
    <table>
        <tr><th>Métrica Estadística</th><th>Valor Obtenido</th></tr>
        <tr><td>Media Global</td><td>{stats['chl_mean']:.4f}</td></tr>
        <tr><td>Mediana (P50)</td><td>{stats['chl_median']:.4f}</td></tr>
        <tr><td>Desviación Estándar (σ)</td><td>{stats['chl_std']:.4f}</td></tr>
        <tr><td>Varianza (σ²)</td><td>{stats['chl_var']:.4f}</td></tr>
        <tr><td>Percentil 90 (P90)</td><td>{stats['chl_p90']:.4f}</td></tr>
        <tr><td>Percentil 95 (P95)</td><td>{stats['chl_p95']:.4f}</td></tr>
        <tr><td>Percentil 99 (P99)</td><td>{stats['chl_p99']:.4f}</td></tr>
        <tr><td>Máximo Absoluto</td><td>{stats['chl_max']:.4f}</td></tr>
    </table>
    
    <div class="highlight">
        <strong>Nota Analítica:</strong> La elevada varianza espacial y la asimetría positiva fuerte (la media es el doble de la mediana y el P99 está en {stats['chl_p99']:.2f} mientras el máximo es {stats['chl_max']:.2f}) son consistentes con la dinámica de afloramientos (upwelling) costeros impulsados por el viento en el Sistema de la Corriente de California.
    </div>

    <h3>3.2. Variable Descriptiva: Profundidad (m)</h3>
    <table>
        <tr><th>Métrica Estadística</th><th>Valor Obtenido</th></tr>
        <tr><td>Media de Muestreo</td><td>{stats['prof_mean']:.2f} m</td></tr>
        <tr><td>Mediana de Muestreo</td><td>{stats['prof_median']:.2f} m</td></tr>
        <tr><td>Profundidad Máxima Evaluada</td><td>{stats['prof_max']:.2f} m</td></tr>
        <tr><td>Correlación de Pearson (Profundidad vs Clorofila)</td><td>{stats['corr_chl_prof']:.4f}</td></tr>
    </table>
    <p>La correlación negativa ({stats['corr_chl_prof']:.4f}) indica, como es esperado ecológicamente, un decaimiento exponencial generalizado de la biomasa fitoplanctónica a medida que se desciende en la columna de agua (debido a la atenuación de la radiación PAR).</p>

    <h2>4. Metodología para la Generación de un Mapa Continuo (Interpolación)</h2>
    <p>Dado que los datos actuales provienen de estaciones discretas (puntos euclidianos fijos), para generar un campo o malla continua 2D/3D representativo del océano, es necesario aplicar métodos estadísticos o determinísticos antes de implementar arquitecturas avanzadas como las <strong>Physics-Informed Neural Networks (PINNs)</strong>. Las técnicas convencionales recomendadas para esta etapa son:</p>
    
    <ul>
        <li><strong>Inverse Distance Weighting (IDW):</strong> Un método determinístico rápido. Asume que los puntos más cercanos tienen mayor similitud que los lejanos. Su desventaja en oceanografía es que genera patrones de "ojo de buey" alrededor de los puntos de muestreo y no considera las barreras físicas como la costa.</li>
        <li><strong>Kriging (Ordinario y Universal):</strong> Un método geoestadístico avanzado. Se basa en el modelado del variograma espacial, el cual captura la autocorrelación de la clorofila a diferentes distancias (anisotropía). Es la técnica <em>Gold Standard</em> en oceanografía estadística porque proporciona mapas de varianza del error de interpolación. Para este caso costero, Kriging Universal con la profundidad batimétrica como co-variable (Cokriging) sería ideal.</li>
        <li><strong>Optimal Interpolation (OI):</strong> Muy utilizada en asimilación de datos marinos. Modela la covarianza del error de fondo (background) y de las observaciones, integrando con facilidad las correlaciones espacio-temporales del campo de clorofila y atenuándose cerca del litoral (isobáticas).</li>
        <li><strong>Splines con Barreras:</strong> Métodos matemáticos (ej. Thin-plate splines) que permiten ajustar una superficie suave por los puntos minimizando la curvatura. Implementar la barrera de costa (land mask) evitaría la propagación irreal de la señal de clorofila hacia el continente.</li>
    </ul>

    <div class="highlight">
        <strong>Conclusión para Transición a PINNs:</strong> Para modelar este ecosistema con PINNs, generar un campo continuo inicial mediante Kriging u OI como "Ground Truth" o malla base puede ser útil. Sin embargo, las PINNs (como arquitecturas Mesh-free) destacan precisamente por su capacidad de asimilar datos en puntos dispersos (las estaciones discretas) y acoplarlos a ecuaciones de advección-difusión biológica, generando la malla continua internamente de manera físicamente consistente.
    </div>

</body>
</html>
"""

with open("Reporte_Tecnico_IMECOCAL.html", "w", encoding='utf-8') as f:
    f.write(html_content)

print("HTML Report generated.")
