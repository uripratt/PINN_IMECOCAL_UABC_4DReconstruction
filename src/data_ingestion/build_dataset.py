import os
import numpy as np
import pandas as pd
import xarray as xr

def build_augmented_dataset():
    """
    Construye 'imecocal_augmented.parquet': el dataset de entrada de la PINN,
    aumentado con covariables físicas de CMEMS y ETOPO1.

    Fase 0 (2026-09-03): reescrito para leer directamente
    `chl_unified_v1.csv` (botella + CTD ya fusionados y calibrados con el
    modelo jerárquico log-log + término de profundidad, r=0.81 held-out) en
    vez de re-parsear las fuentes crudas con un clip simple [-5,100] y sin
    calibración. Esto también arregla dos bugs documentados en
    `data/raw/imecocal/propuesta_integracion_datos_pinn.md` (Sección 8.2):

    - Bug B: el `date_map` anterior solo se construía desde la botella
      antigua (`Cl_Imec98_12.xlsx`, hasta 2012), así que cualquier lance de
      CTD de un (año, mes) fuera de ese archivo se descartaba por Fecha=NaT
      — probablemente todo 2013-2019. `chl_unified_v1.csv` ya trae fecha
      exacta por fila, así que no hace falta ese merge en absoluto.
    - Lista negra: la anterior excluía `0101` y `0207` enteros; la
      calibración jerárquica los rescató con coeficiente propio (ver
      `informe_calibracion_jerarquica.pdf`, Sección 3.4/discusión) y ya
      vienen con valor válido en `chl_unified_v1.csv`.

    Los ~1.6M puntos de CTD sin clorofila calibrada (sensor apagado,
    cruceros excluidos, etc.) se dejan fuera de esta build a propósito: son
    candidatos a colocación física adicional (checklist, no aplicado
    todavía), no observaciones para la pérdida de datos.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    unified_path = os.path.join(
        project_root,
        'data/raw/imecocal/analysis_output/chl_unified_v1.csv',
    )
    bathy_path = os.path.join(project_root, 'data/raw/etopo_bathymetry.nc')
    output_path = os.path.join(project_root, 'data/processed/imecocal_augmented.parquet')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("1. Cargando dataset unificado y calibrado (chl_unified_v1.csv)...")
    if not os.path.exists(unified_path):
        raise FileNotFoundError(
            f"No se encontró {unified_path}. Este es el dataset producido por el "
            "pipeline de calibración (analysis_output/); revisa el DATA_CARD si falta."
        )

    df = pd.read_csv(unified_path, low_memory=False)
    # OJO (bug encontrado y arreglado 2026-09-03): 'datetime' mezcla formatos
    # entre fuentes -- botella es 'YYYY-MM-DD', CTD es 'YYYY-MM-DD HH:MM:SS'.
    # Sin format='mixed', pandas infiere el formato de las primeras filas
    # (botella) y descarta en SILENCIO cualquier fila que no calce ese
    # formato exacto (errors='coerce' -> NaT), lo que en la práctica
    # convertía TODAS las filas de CTD en NaT y las tiraba más abajo.
    # Verificado: sin 'mixed', 4.266.642/4.266.642 filas de CTD daban NaT.
    df['Fecha'] = pd.to_datetime(df['datetime'], format='mixed', errors='coerce')
    df = df.rename(columns={
        'lat': 'Latitud',
        'lon': 'Longitud',
        'profundidad_m': 'Depth',
    })

    # Chequeo defensivo: si una fuente entera pierde su fecha, algo se rompió
    # en el parseo -- que falle ruidosamente en vez de silenciosamente.
    nat_por_fuente = df.groupby('fuente')['Fecha'].apply(lambda s: s.isna().mean())
    for fuente, frac_nat in nat_por_fuente.items():
        if frac_nat > 0.5:
            raise ValueError(
                f"{frac_nat:.0%} de las fechas de fuente='{fuente}' no se pudieron "
                "parsear (NaT). Esto probablemente indica un problema de formato "
                "en la columna 'datetime' de chl_unified_v1.csv -- revisar antes "
                "de continuar, no seguir con un dataset silenciosamente incompleto."
            )

    # Solo observaciones con clorofila calibrada válida. Los registros con
    # chl_mgm3=NaN (sensor apagado, cruceros excluidos, techos físicos) se
    # excluyen aquí -- ver docstring.
    df = df.dropna(subset=['chl_mgm3', 'Latitud', 'Longitud', 'Depth', 'Fecha']).copy()

    df['Año'] = df['Fecha'].dt.year
    df['Chl_CTD'] = np.where(df['fuente'] == 'ctd', df['chl_mgm3'], np.nan)
    df['Chl_Bottle'] = np.where(df['fuente'] == 'botella', df['chl_mgm3'], np.nan)

    print(f"Total registros con clorofila calibrada válida: {len(df)}")
    print(df['fuente'].value_counts().to_string())
    print(f"Rango de años: {int(df['Año'].min())}-{int(df['Año'].max())}")

    keep_cols = [
        'Latitud', 'Longitud', 'Depth', 'Fecha', 'Año',
        'Chl_CTD', 'Chl_Bottle', 'peso_entrenamiento', 'fuente', 'crucero_code',
    ]
    df_final = df[keep_cols].reset_index(drop=True)

    print("2. Extrayendo CMEMS (Corrientes, Temperatura, Satélite) de forma Vectorizada...")
    df_final['uo'] = 0.0
    df_final['vo'] = 0.0
    df_final['wo'] = 0.0
    df_final['thetao'] = 15.0
    df_final['CHL_sat'] = 0.0

    years = sorted(df_final['Año'].dropna().unique())
    for y in years:
        cmems_file = os.path.join(project_root, f'data/raw/cmems_yearly/cmems_currents_{int(y)}_with_w.nc')
        thetao_file = os.path.join(project_root, f'data/raw/cmems_yearly_thetao/cmems_thetao_{int(y)}.nc')
        chl_sat_file = os.path.join(project_root, f'data/raw/satellite_chl_yearly/satellite_chl_{int(y)}.nc')

        mask = df_final['Año'] == y
        if not mask.any():
            continue

        print(f"  Vectorizando extracciones CMEMS para Año {y}...")
        df_y = df_final[mask]

        x = xr.DataArray(df_y['Longitud'].values, dims='points')
        y_lat = xr.DataArray(df_y['Latitud'].values, dims='points')
        z = xr.DataArray(df_y['Depth'].values, dims='points')
        t = xr.DataArray(df_y['Fecha'].values, dims='points')

        if os.path.exists(cmems_file):
            try:
                ds_year = xr.open_dataset(cmems_file)
            except Exception as e:
                print(f"  [Advertencia] No se pudo leer {cmems_file}: {e}")
                ds_year = None
            if ds_year is not None:
                try:
                    val = ds_year.sel(longitude=x, latitude=y_lat, depth=z, time=t, method='nearest')
                    df_final.loc[mask, 'uo'] = val['uo'].values
                    df_final.loc[mask, 'vo'] = val['vo'].values
                    df_final.loc[mask, 'wo'] = val['wo'].values
                except Exception as e:
                    print(f"  Error en corrientes {y}: {e}")
                ds_year.close()
        else:
            print(f"  [Info] Sin corrientes CMEMS para {y} (fuera de cobertura 1998-2012); uo/vo/wo quedan en 0.0.")

        if os.path.exists(thetao_file):
            try:
                ds_thetao = xr.open_dataset(thetao_file)
            except Exception as e:
                print(f"  [Advertencia] No se pudo leer {thetao_file}: {e}")
                ds_thetao = None
            if ds_thetao is not None:
                try:
                    val_t = ds_thetao.sel(longitude=x, latitude=y_lat, depth=z, time=t, method='nearest')
                    df_final.loc[mask, 'thetao'] = val_t['thetao'].values
                except Exception:
                    pass
                ds_thetao.close()

        if os.path.exists(chl_sat_file):
            try:
                ds_chl = xr.open_dataset(chl_sat_file)
            except Exception as e:
                print(f"  [Advertencia] No se pudo leer {chl_sat_file}: {e}")
                ds_chl = None
            if ds_chl is not None:
                try:
                    # Satélite no tiene profundidad
                    val_c = ds_chl.sel(longitude=x, latitude=y_lat, time=t, method='nearest')
                    df_final.loc[mask, 'CHL_sat'] = val_c['CHL'].values
                except Exception:
                    pass
                ds_chl.close()

    print("3. Extrayendo Batimetría Vectorizada...")
    if os.path.exists(bathy_path):
        try:
            ds_bathy = xr.open_dataset(bathy_path)
            var_name = 'altitude' if 'altitude' in ds_bathy else 'elevation'
        except Exception as e:
            print(f"Error bathy: {e}")
            ds_bathy = None

        if ds_bathy is not None:
            unique_coords = df_final[['Latitud', 'Longitud']].drop_duplicates()
            x = xr.DataArray(unique_coords['Longitud'].values, dims='points')
            y_lat = xr.DataArray(unique_coords['Latitud'].values, dims='points')

            try:
                val = ds_bathy.sel(longitude=x, latitude=y_lat, method='nearest')
                unique_coords['bathy'] = val[var_name].values
            except Exception:
                unique_coords['bathy'] = 0.0

            df_final = df_final.merge(unique_coords, on=['Latitud', 'Longitud'], how='left')
            ds_bathy.close()

    print(f"Guardando dataset consolidado ({len(df_final)} filas) en Parquet...")
    df_final.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"\n¡Dataset Multi-Fidelidad guardado exitosamente en: {output_path}!")

if __name__ == "__main__":
    build_augmented_dataset()
