"""
Climatología suave de log1p(Chl) para usar como PRIOR del modelo (2026-09-25).

Motivación (revisión de resultados del 2026-09-25): sobre el crucero LOCO 2005-4
predecir la media del train da MSE=0.0336 y una climatología profundidad x celda
1 grado da 0.0242, mientras que la PINN, pasadas ~100 épocas, daba 0.04-0.05:
peor que predecir la media. Modelar como  L = clima(z,lat,lon) + corrección_NN
hace que la red parta de la climatología y solo se aleje de ella donde los datos
lo justifican (en huecos sin datos vuelve a la climatología en vez de
"inventar").

La climatología se construye SOLO con datos de entrenamiento (nunca val/test)
y se evalúa con interpolación trilineal de una rejilla (log-profundidad, lat,
lon), que es diferenciable respecto a las coordenadas (necesario para que el
residuo de la PDE pueda usar dL/dx del campo completo).
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter

Z_MAX = 3000.0


def _fill_nan_by_smoothing(grid, sigmas=((1, 1, 1), (2, 3, 3), (4, 8, 8))):
    """Rellena NaN con versiones cada vez más suavizadas (normalizadas por máscara)."""
    out = grid.copy()
    for sg in sigmas:
        if not np.isnan(out).any():
            break
        m = np.isfinite(out).astype(float)
        num = gaussian_filter(np.where(m > 0, out, 0.0), sg, mode="nearest")
        den = gaussian_filter(m, sg, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            sm = num / den
        fill = np.isnan(out) & (den > 1e-6)
        out[fill] = sm[fill]
    if np.isnan(out).any():
        out[np.isnan(out)] = np.nanmean(out)
    return out


def build_climatology_grid(lat, lon, depth, y, w, lat_range, lon_range, n_lat=44, n_lon=48, n_z=20):
    """Media ponderada de y por celda (log-prof, lat, lon), suavizada. Devuelve (grid, logz_range).

    lat, lon, depth, y, w: arrays 1D de TRAIN. La rejilla tiene n_z nodos uniformes en log1p(prof)."""
    lz = np.log1p(np.clip(depth, 0, Z_MAX))
    lz_range = (0.0, float(np.log1p(Z_MAX)))
    iz = np.clip(np.rint((lz - lz_range[0]) / (lz_range[1] - lz_range[0]) * (n_z - 1)).astype(int), 0, n_z - 1)
    ia = np.clip(np.rint((lat - lat_range[0]) / (lat_range[1] - lat_range[0]) * (n_lat - 1)).astype(int), 0, n_lat - 1)
    io = np.clip(np.rint((lon - lon_range[0]) / (lon_range[1] - lon_range[0]) * (n_lon - 1)).astype(int), 0, n_lon - 1)
    flat = (iz * n_lat + ia) * n_lon + io
    size = n_z * n_lat * n_lon
    num = np.bincount(flat, weights=w * y, minlength=size).reshape(n_z, n_lat, n_lon)
    den = np.bincount(flat, weights=w, minlength=size).reshape(n_z, n_lat, n_lon)
    sg = (0.8, 1.0, 1.0)
    num_s = gaussian_filter(num, sg, mode="nearest")
    den_s = gaussian_filter(den, sg, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.where(den_s > 1e-3 * max(den_s.max(), 1e-12), num_s / den_s, np.nan)
    grid = _fill_nan_by_smoothing(grid)
    return grid.astype(np.float32), lz_range


class ClimatologyPrior(nn.Module):
    """Interpolación trilineal diferenciable de la rejilla climatológica."""

    def __init__(self, grid, lat_range, lon_range, lz_range):
        super().__init__()
        self.register_buffer("grid", torch.as_tensor(grid, dtype=torch.float32)[None, None])  # (1,1,D,H,W)
        self.register_buffer("ranges", torch.tensor([lat_range[0], lat_range[1], lon_range[0], lon_range[1],
                                                     lz_range[0], lz_range[1]], dtype=torch.float32))

    def forward(self, lat, lon, depth):
        """Interpolación trilineal MANUAL (no F.grid_sample): la PDE necesita derivar dos veces
        (dL/dx dentro de la pérdida, que luego se retropropaga) y grid_sampler_3d no implementa
        la doble derivada. Los índices son enteros (sin gradiente); los pesos son lineales en las
        coordenadas, así que los gradientes de primer orden sí fluyen."""
        g = self.grid[0, 0]                      # (D, H, W)
        D, H, W = g.shape
        r = self.ranges
        fx = ((lon - r[2]) / (r[3] - r[2])).clamp(0.0, 1.0) * (W - 1)
        fy = ((lat - r[0]) / (r[1] - r[0])).clamp(0.0, 1.0) * (H - 1)
        lz = torch.log1p(depth.clamp(min=0.0, max=Z_MAX))
        fz = ((lz - r[4]) / (r[5] - r[4])).clamp(0.0, 1.0) * (D - 1)
        x0 = fx.detach().floor().long().clamp(0, W - 2); tx = fx - x0
        y0 = fy.detach().floor().long().clamp(0, H - 2); ty = fy - y0
        z0 = fz.detach().floor().long().clamp(0, D - 2); tz = fz - z0
        x0, y0, z0 = x0.squeeze(1), y0.squeeze(1), z0.squeeze(1)

        def c(dz, dy, dx):
            return g[z0 + dz, y0 + dy, x0 + dx].unsqueeze(1)

        c00 = c(0, 0, 0) * (1 - tx) + c(0, 0, 1) * tx
        c01 = c(0, 1, 0) * (1 - tx) + c(0, 1, 1) * tx
        c10 = c(1, 0, 0) * (1 - tx) + c(1, 0, 1) * tx
        c11 = c(1, 1, 0) * (1 - tx) + c(1, 1, 1) * tx
        c0 = c00 * (1 - ty) + c01 * ty
        c1 = c10 * (1 - ty) + c11 * ty
        return c0 * (1 - tz) + c1 * tz

    @classmethod
    def empty_like_shape(cls, grid_shape):
        return cls(np.zeros(grid_shape, dtype=np.float32), (0, 1), (0, 1), (0, 1))
