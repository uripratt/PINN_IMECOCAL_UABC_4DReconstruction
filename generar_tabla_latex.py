import pandas as pd
import numpy as np

# Cargar los datos completos
df = pd.read_csv('Cobertura_Completa_IMECOCAL.csv')

# Identificar el top 3 de clorofila por cada año
top3_pairs = set()
for año in df['Año'].unique():
    año_df = df[df['Año'] == año]
    # Dropna para Max_Clorofila y ordenar descendente
    top3 = año_df.dropna(subset=['Max_Clorofila_mg_m3']).sort_values('Max_Clorofila_mg_m3', ascending=False).head(3)
    for st in top3['Estacion']:
        top3_pairs.add((st, año))

# Crear matriz pivot de esfuerzo (Num_Perfiles)
esfuerzo = df.pivot(index='Estacion', columns='Año', values='Num_Perfiles').fillna(0).astype(int)
años = esfuerzo.columns.tolist()
estaciones = esfuerzo.index.tolist()

# Escribir código LaTeX
latex_lines = []
latex_lines.append(r"% ==========================================")
latex_lines.append(r"% Tabla de Esfuerzo de Muestreo (Auto-Generada)")
latex_lines.append(r"% ==========================================")
latex_lines.append(r"{\scriptsize")
latex_lines.append(r"\setlength{\tabcolsep}{3pt}")
latex_lines.append(r"\begin{longtable}{l" + "c" * len(años) + "}")
latex_lines.append(r"\caption{Matriz de esfuerzo de muestreo espaciotemporal (123 estaciones). Los valores indican el número de perfiles de clorofila procesados por estación y año. Las celdas sombreadas en verde pastel indican las tres estaciones con los picos máximos absolutos de biomasa registrados en ese año.} \label{tab:esfuerzo} \\")
latex_lines.append(r"\toprule")
latex_lines.append(r"Estación & " + " & ".join([str(a) for a in años]) + r" \\")
latex_lines.append(r"\midrule")
latex_lines.append(r"\endfirsthead")

# Header simplificado
latex_lines.append(r"\toprule")
latex_lines.append(r"Estación & " + " & ".join([str(a) for a in años]) + r" \\")
latex_lines.append(r"\midrule")
latex_lines.append(r"\endhead")
latex_lines.append(r"\bottomrule")
latex_lines.append(r"\endfoot")
latex_lines.append(r"\bottomrule")
latex_lines.append(r"\endlastfoot")

# Rellenar filas
for st in estaciones:
    row_str = [str(st).replace('_', r'\_')]
    for año in años:
        val = esfuerzo.loc[st, año]
        val_str = str(val) if val > 0 else "-"
        
        if (st, año) in top3_pairs:
            row_str.append(r"\cellcolor{green!20}\textbf{" + val_str + "}")
        else:
            row_str.append(val_str)
            
    latex_lines.append(" & ".join(row_str) + r" \\")

# Fila final de totales
latex_lines.append(r"\midrule")
totales = [str(esfuerzo[a].sum()) for a in años]
latex_lines.append(r"\textbf{Total Perfiles} & \textbf{" + r"} & \textbf{".join(totales) + r"} \\")

latex_lines.append(r"\end{longtable}")
latex_lines.append(r"}")

with open('tabla_esfuerzo_latex.tex', 'w') as f:
    f.write("\n".join(latex_lines))

print("Archivo tabla_esfuerzo_latex.tex generado con éxito.")
