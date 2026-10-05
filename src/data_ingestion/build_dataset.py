import os
import numpy as np
import pandas as pd
import xarray as xr

def _extract_by_month(sub, nc_path, varnames, with_depth):
    """Extracción puntual (vecino más cercano) troceada por mes para acotar la RAM.
    sub: DataFrame con Latitud, Longitud, Depth, Fecha. Devuelve {var: array} alineado con sub.
    Lanza excepción si algo falla (no se silencia)."""
    out = {v: np.full(len(sub), np.nan) for v in varnames}
    months = sub['Fecha'].dt.month.values
    with xr.open_dataset(nc_path) as ds:
        for m in np.unique(months):
            sel_m = months == m
            d = sub[sel_m]
            t0 = d['Fecha'].min().normalize() - pd.Timedelta(days=1)
            t1 = d['Fecha'].max().normalize() + pd.Timedelta(days=1)
            # .load(): la indexación puntual sobre el netCDF perezoso (int16 + scale_factor) tarda
            # minutos por año; con el mes en memoria (~65 MB) es instantánea.
            dsm = ds.sel(time=slice(t0, t1)).load()
            kw = dict(longitude=xr.DataArray(d['Longitud'].values, dims='points'),
                      latitude=xr.DataArray(d['Latitud'].values, dims='points'),
                      time=xr.DataArray(d['Fecha'].values, dims='points'))
            if with_depth:
                kw['depth'] = xr.DataArray(d['Depth'].values, dims='points')
            val = dsm.sel(**kw, method='nearest')
            for v in varnames:
                out[v][sel_m] = val[v].values
    return out


def build_ocean_collocation(project_root, n_points=300_000, seed=0):
    """Pool de puntos de colocación OCEÁNICOS con forzante real (2026-09-25).

    Hasta ahora los únicos puntos de colocación eran de tierra (elev>0), con residuo
    de PDE anulado, así que la PDE solo se evaluaba en los puntos con dato y nunca en
    los huecos, que es donde un PINN aporta algo. Aquí se muestrean (lat, lon, prof,
    día) uniformes en el dominio CMEMS (1998-2012), en océano (ETOPO<0) y con prof
    <= fondo, y se extrae u,v,w,T reales. Se descartan puntos con forzante NaN."""
    bathy_path = os.path.join(project_root, 'data/raw/etopo_bathymetry.nc')
    rng = np.random.default_rng(seed)
    lat_rng, lon_rng = (23.9, 32.6), (-119.6, -112.1)   # interior de la malla CMEMS
    with xr.open_dataset(bathy_path) as ds_b:
        var_name = 'altitude' if 'altitude' in ds_b else 'elevation'
        lat_b = ds_b.latitude.values
        lon_b = ds_b.longitude.values
        elev = ds_b[var_name].values
    frames = []
    n_target = int(n_points * 1.6)   # margen para lo que se descarte
    lat = rng.uniform(*lat_rng, n_target)
    lon = rng.uniform(*lon_rng, n_target)
    ilat = np.clip(np.searchsorted(lat_b, lat), 0, len(lat_b) - 1)
    ilon = np.clip(np.searchsorted(lon_b, lon), 0, len(lon_b) - 1)
    el = elev[ilat, ilon]
    ok = el < -30.0                                   # océano con al menos 30 m de columna
    lat, lon, el = lat[ok], lon[ok], el[ok]
    dmax = np.minimum(-el, 1000.0)
    depth = dmax * rng.uniform(0, 1, len(lat)) ** 2   # más densidad en la capa superior
    days = rng.integers(0, (pd.Timestamp('2012-12-31') - pd.Timestamp('1998-01-01')).days + 1, len(lat))
    fecha = pd.Timestamp('1998-01-01') + pd.to_timedelta(days, unit='D')
    col = pd.DataFrame({'Latitud': lat, 'Longitud': lon, 'Depth': depth, 'Fecha': fecha, 'bathy': el})
    col = col.iloc[:int(n_points * 1.3)].reset_index(drop=True)
    for c in ('uo', 'vo', 'wo', 'thetao'):
        col[c] = np.nan
    for y in sorted(col['Fecha'].dt.year.unique()):
        idx = col.index[col['Fecha'].dt.year == y]
        sub = col.loc[idx]
        r = _extract_by_month(sub, os.path.join(project_root, f'data/raw/cmems_yearly/cmems_currents_{y}.nc'), ['uo', 'vo'], True)
        col.loc[idx, 'uo'], col.loc[idx, 'vo'] = r['uo'], r['vo']
        r = _extract_by_month(sub, os.path.join(project_root, f'data/raw/cmems_yearly/cmems_w_v2_{y}.nc'), ['wo'], True)
        col.loc[idx, 'wo'] = r['wo']
        r = _extract_by_month(sub, os.path.join(project_root, f'data/raw/cmems_yearly_thetao/cmems_thetao_{y}.nc'), ['thetao'], True)
        col.loc[idx, 'thetao'] = r['thetao']
    n0 = len(col)
    col = col.dropna(subset=['uo', 'vo', 'wo', 'thetao']).reset_index(drop=True).iloc[:n_points]
    print(f"   Colocación oceánica: {len(col)} puntos con forzante real (descartados por NaN: {n0 - len(col)})")
    return col


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

    # Lectura por trozos, solo columnas necesarias y solo filas con clorofila válida:
    # el CSV pesa ~526 MB y completo puede agotar la RAM en equipos justos.
    usecols = ['lon', 'lat', 'profundidad_m', 'datetime', 'chl_mgm3', 'fuente',
               'peso_entrenamiento', 'crucero_code']
    partes = []
    for chunk in pd.read_csv(unified_path, usecols=usecols, chunksize=400_000, low_memory=False):
        partes.append(chunk[chunk['chl_mgm3'].notna()])
    df = pd.concat(partes, ignore_index=True)
    del partes
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

    print("2. Extrayendo forzantes (corrientes u,v; w v2; temperatura) y satélite, con flags de validez...")
    # Los flags *_ok distinguen forzante REAL de valor imputado: el dataloader/harness
    # los usan para NO evaluar la PDE donde la forzante es inventada (0 m/s, 15 C).
    # (Bug 2026-09-25: el 100% de filas tenía u=v=w=0 y nadie lo notó; la razón era
    # que se buscaba `_with_w.nc`, que no existe en todos los entornos, y el mensaje
    # decía "fuera de cobertura" aunque el año SÍ estuviera cubierto.)
    df_final['uo'] = 0.0
    df_final['vo'] = 0.0
    df_final['wo'] = 0.0
    df_final['thetao'] = 15.0
    df_final['CHL_sat'] = 0.0
    df_final['uv_ok'] = False
    df_final['w_ok'] = False
    df_final['T_ok'] = False

    coverage = []
    years = sorted(int(v) for v in df_final['Año'].dropna().unique())
    for y in years:
        cur_file = os.path.join(project_root, f'data/raw/cmems_yearly/cmems_currents_{y}.nc')
        w_file = os.path.join(project_root, f'data/raw/cmems_yearly/cmems_w_v2_{y}.nc')
        thetao_file = os.path.join(project_root, f'data/raw/cmems_yearly_thetao/cmems_thetao_{y}.nc')
        chl_sat_file = os.path.join(project_root, f'data/raw/satellite_chl_yearly/satellite_chl_{y}.nc')

        idx = df_final.index[df_final['Año'] == y]
        if len(idx) == 0:
            continue
        print(f"  Año {y}: {len(idx)} filas")

        # --- corrientes horizontales (archivo crudo, sin w) ---
        if os.path.exists(cur_file):
            r = _extract_by_month(df_final.loc[idx], cur_file, ['uo', 'vo'], with_depth=True)
            ok = np.isfinite(r['uo']) & np.isfinite(r['vo'])
            df_final.loc[idx, 'uo'] = np.where(ok, r['uo'], 0.0)
            df_final.loc[idx, 'vo'] = np.where(ok, r['vo'], 0.0)
            df_final.loc[idx, 'uv_ok'] = ok
        else:
            print(f"    [Info] No hay archivo de corrientes para {y} ({cur_file}); u,v quedan sin forzante (uv_ok=False).")

        # --- velocidad vertical v2 (compute_vertical_velocity_v2.py) ---
        if os.path.exists(w_file):
            r = _extract_by_month(df_final.loc[idx], w_file, ['wo'], with_depth=True)
            ok = np.isfinite(r['wo'])
            df_final.loc[idx, 'wo'] = np.where(ok, r['wo'], 0.0)
            df_final.loc[idx, 'w_ok'] = ok
        elif os.path.exists(cur_file):
            print(f"    [Advertencia] Falta {os.path.basename(w_file)}: ejecuta compute_vertical_velocity_v2.py. w queda sin forzante (w_ok=False).")

        # --- temperatura ---
        if os.path.exists(thetao_file):
            r = _extract_by_month(df_final.loc[idx], thetao_file, ['thetao'], with_depth=True)
            ok = np.isfinite(r['thetao'])
            df_final.loc[idx, 'thetao'] = np.where(ok, r['thetao'], 15.0)
            df_final.loc[idx, 'T_ok'] = ok
        else:
            print(f"    [Info] No hay archivo de thetao para {y}; T_ok=False.")

        # --- satélite (sin profundidad) ---
        if os.path.exists(chl_sat_file):
            r = _extract_by_month(df_final.loc[idx], chl_sat_file, ['CHL'], with_depth=False)
            df_final.loc[idx, 'CHL_sat'] = np.where(np.isfinite(r['CHL']), r['CHL'], 0.0)

        sub = df_final.loc[idx]
        coverage.append((y, len(idx), sub['uv_ok'].mean(), sub['w_ok'].mean(), sub['T_ok'].mean(), (sub['CHL_sat'] > 0).mean(),
                         os.path.exists(cur_file)))

    df_final['forcing_ok'] = df_final['uv_ok'] & df_final['w_ok'] & df_final['T_ok']

    print("\n   Cobertura REAL de forzantes por año (fracción de filas con dato, no imputado):")
    print("   año   filas    u,v     w      T    sat   forcing_ok")
    for (y, n_y, uv, wk, tk, sk, has_file) in coverage:
        fo = df_final.loc[df_final['Año'] == y, 'forcing_ok'].mean()
        print(f"   {y}  {n_y:7d}  {uv:5.2f}  {wk:5.2f}  {tk:5.2f}  {sk:5.2f}   {fo:5.2f}")
    print(f"   TOTAL forcing_ok: {df_final['forcing_ok'].mean():.1%} de las filas")

    # Chequeo defensivo: si el archivo de corrientes EXISTE para un año, la mayoría de
    # sus filas deben salir con u,v reales. Si no, algo se rompió en silencio.
    for (y, n_y, uv, wk, tk, sk, has_file) in coverage:
        if has_file and uv < 0.5:
            raise RuntimeError(f"Año {y}: existe el archivo de corrientes pero solo {uv:.0%} de las filas "
                               "tiene u,v reales. Revisar la extracción antes de entrenar.")


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

    print("4. Construyendo pool de puntos de colocación oceánicos con forzante real...")
    col = build_ocean_collocation(project_root)
    col_path = os.path.join(project_root, 'data/processed/ocean_colloc.parquet')
    col.to_parquet(col_path, engine='pyarrow', index=False)
    print(f"   Guardado {col_path}")

if __name__ == "__main__":
    build_augmented_dataset()
