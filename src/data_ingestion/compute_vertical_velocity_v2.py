"""
Velocidad vertical w a partir de la continuidad (incompresibilidad), v2.

Sustituye a compute_vertical_velocity.py (raíz del repo), que tenía un error
de unidades: dividía d(u)/d(lon en GRADOS) entre los metros de UN PASO de
malla (0.0833 deg), en vez de entre los metros de UN GRADO, lo que sobreestima
la divergencia horizontal exactamente ~12x (=1/paso de malla). Verificado
numéricamente el 2026-09-25 sobre 2005-04-11: |w| p99 (0-200 m) = 895 m/día
con el script antiguo vs 75 m/día con la fórmula correcta.

Cambios respecto al script antiguo:
- Divergencia con la métrica correcta: du/dx = (du/dlon)/(m_por_grado*cos(lat)).
- Integración con tapa rígida desde la superficie: w(z) = -int_0^z div dz'
  (regla del trapecio sobre la malla vertical no uniforme).
- Escritura día a día con netCDF4 (memoria constante, ~2 MB por día), en
  un archivo APARTE cmems_w_v2_{año}.nc con solo `wo`, para no duplicar u/v
  y para que NO colisione con los `_with_w.nc` antiguos (que no deben usarse).

Limitación conocida: NaN de tierra/fondo se sustituyen por 0 en la divergencia,
lo que infraestima |w| en columnas pegadas a la costa (justo donde hay
surgencia). Lo ideal sería descargar `wo` directamente de GLORYS12.
"""
import os
import sys
import glob
import numpy as np
import xarray as xr
import netCDF4

R_EARTH = 6371000.0
M_PER_DEG = np.pi / 180.0 * R_EARTH


def w_day(uo, vo, lat, lon, depth):
    """uo, vo: arrays (depth, lat, lon) en m/s. Devuelve w (depth, lat, lon) en m/s, float32."""
    coslat = np.cos(np.deg2rad(lat))[None, :, None]
    du_dx = np.gradient(uo, lon, axis=2) / (M_PER_DEG * coslat)
    dv_dy = np.gradient(vo, lat, axis=1) / M_PER_DEG
    div = np.nan_to_num(du_dx + dv_dy, nan=0.0)
    dz = np.diff(depth)[:, None, None]
    incr = 0.5 * (div[1:] + div[:-1]) * dz
    w = np.zeros_like(div)
    w[1:] = -np.cumsum(incr, axis=0)
    return w.astype(np.float32)


def process_year(path_in, path_out):
    ds = xr.open_dataset(path_in)
    lat = ds.latitude.values.astype(np.float64)
    lon = ds.longitude.values.astype(np.float64)
    depth = ds.depth.values.astype(np.float64)
    nt = ds.sizes["time"]
    tmp = path_out + ".partial"
    with netCDF4.Dataset(tmp, "w", format="NETCDF4") as nc:
        nc.createDimension("time", None)
        nc.createDimension("depth", len(depth))
        nc.createDimension("latitude", len(lat))
        nc.createDimension("longitude", len(lon))
        for name, vals in (("depth", depth), ("latitude", lat), ("longitude", lon)):
            v = nc.createVariable(name, "f8", (name,))
            v[:] = vals
        tv = nc.createVariable("time", "f8", ("time",))
        tv.units = "hours since 1950-01-01 00:00:00"
        tv.calendar = "gregorian"
        wv = nc.createVariable("wo", "f4", ("time", "depth", "latitude", "longitude"),
                               zlib=True, complevel=4, chunksizes=(1, len(depth), len(lat), len(lon)))
        wv.units = "m s-1"
        wv.long_name = "vertical velocity from continuity (rigid lid), v2"
        times = ds.time.values
        for i in range(nt):
            d = ds.isel(time=i)
            wv[i] = w_day(d.uo.values, d.vo.values, lat, lon, depth)
            tv[i] = (times[i] - np.datetime64("1950-01-01T00:00:00")) / np.timedelta64(1, "h")
    ds.close()
    os.replace(tmp, path_out)


if __name__ == "__main__":
    in_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/cmems_yearly"
    files = sorted(f for f in glob.glob(os.path.join(in_dir, "cmems_currents_[0-9][0-9][0-9][0-9].nc")))
    print(f"{len(files)} archivos de corrientes crudos encontrados")
    for f in files:
        year = os.path.basename(f).split("_")[2].split(".")[0]
        out = os.path.join(in_dir, f"cmems_w_v2_{year}.nc")
        if os.path.exists(out):
            print(f"[{year}] ya existe, salto")
            continue
        print(f"[{year}] calculando w ...", flush=True)
        process_year(f, out)
        print(f"[{year}] guardado {out} ({os.path.getsize(out)/1e6:.0f} MB)", flush=True)
    print("Listo.")
