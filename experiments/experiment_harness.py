import os
import sys
import json
import time
import urllib.request
import urllib.error
import torch
import torch.optim as optim
import mlflow
from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_ingestion.dataloader import get_dataloaders
from src.models.pinn_model import CoastalPINNModel
from src.models.climatology import build_climatology_grid, ClimatologyPrior
from src.physics.physics_loss import CoastalPhysicsPINN
from experiments.plot_inference import plot_continuous_field
import matplotlib.pyplot as plt

LOG10_EPS = 0.01  # mg/m3: suelo para log10(Chl+eps) en las métricas en unidades físicas


# ----------------------------------------------------------------------------
# Puntos de colocación
# ----------------------------------------------------------------------------
def load_land_points():
    """Carga las coordenadas de tierra firme desde el archivo ETOPO."""
    bathy_file = os.path.join(os.path.dirname(__file__), '../data/raw/etopo_bathymetry.nc')
    if not os.path.exists(bathy_file):
        print("Advertencia: No se encontró etopo_bathymetry.nc. No se usarán puntos de colocación terrestres.")
        return None
    ds = xr.open_dataset(bathy_file)
    var_name = 'altitude' if 'altitude' in ds else 'elevation'
    Lon, Lat = np.meshgrid(ds.longitude.values, ds.latitude.values)
    elev = ds[var_name].values
    mask = elev > 0
    out = Lat[mask], Lon[mask], elev[mask]
    ds.close()
    return out


def get_collocation_batch(land_data, batch_size, max_time_days, max_depth, device):
    """Puntos aleatorios sobre tierra firme (solo para la condición de Dirichlet C=0)."""
    land_lats, land_lons, land_elevs = land_data
    indices = np.random.choice(len(land_lats), batch_size, replace=True)
    z = np.zeros(batch_size)
    X_numpy = np.column_stack((
        land_lats[indices], land_lons[indices],
        np.random.uniform(0, max_depth, batch_size), np.random.uniform(0, max_time_days, batch_size),
        z, z, z, land_elevs[indices], np.full(batch_size, 15.0), z, z,   # ..., thetao, chl_sat, forcing_ok=0
    ))
    return torch.tensor(X_numpy, dtype=torch.float32).to(device)


def load_ocean_collocation_pool(t0, device):
    """Pool de colocación OCEÁNICA con forzante real (build_dataset.build_ocean_collocation).
    Devuelve tensor (N, 11) con las mismas columnas que X, o None si no existe."""
    path = os.path.join(os.path.dirname(__file__), '../data/processed/ocean_colloc.parquet')
    if not os.path.exists(path):
        print("[Advertencia] No existe ocean_colloc.parquet (regenera con build_dataset.py): "
              "la PDE solo se evaluará en los puntos con dato.")
        return None
    col = pd.read_parquet(path)
    col['Fecha'] = pd.to_datetime(col['Fecha'])
    t_days = (col['Fecha'] - t0).dt.total_seconds().values / 86400.0
    z = np.zeros(len(col))
    X = np.column_stack((col['Latitud'], col['Longitud'], col['Depth'], t_days,
                         col['uo'], col['vo'], col['wo'], col['bathy'], col['thetao'], z, np.ones(len(col))))
    print(f"Pool de colocación oceánica: {len(col)} puntos con forzante real")
    return torch.tensor(X, dtype=torch.float32).to(device)


def sample_collocation(n_data, land_data, ocean_pool, land_ratio, ocean_ratio, max_time_days, max_depth, device):
    ocean = land = None
    if ocean_pool is not None and ocean_ratio > 0:
        idx = torch.randint(0, ocean_pool.shape[0], (max(1, int(n_data * ocean_ratio)),), device=device)
        ocean = ocean_pool[idx]
    if land_data is not None and land_ratio > 0:
        land = get_collocation_batch(land_data, max(1, int(n_data * land_ratio)), max_time_days, max_depth, device)
    return ocean, land


# ----------------------------------------------------------------------------
# Pérdidas y evaluación
# ----------------------------------------------------------------------------
def compute_weighted_data_loss(pred_y, batch_y, source_share=None):
    """Pérdida de datos multi-fidelidad ponderada por muestra (peso_entrenamiento, col. 4).
    batch_y: [y_ctd, y_bottle, mask_ctd, mask_bottle, peso].

    source_share=None: comportamiento histórico (suma ponderada de todas las muestras).
    source_share=s (p. ej. 0.5): cada fuente aporta una cuota fija de la pérdida, así las
    ~20.800 botellas no quedan diluidas por los ~2,2 M puntos de CTD (98,8 % del peso antes).
    La cuota se aplica por lote y solo si la fuente tiene muestras en ese lote."""
    y_ctd, y_bottle = batch_y[:, 0:1], batch_y[:, 1:2]
    mask_ctd, mask_bottle, weight = batch_y[:, 2:3], batch_y[:, 3:4], batch_y[:, 4:5]
    target = y_ctd * mask_ctd + y_bottle * mask_bottle
    mask_any = torch.clamp(mask_ctd + mask_bottle, max=1.0)
    if source_share is None:
        return torch.sum(weight * mask_any * (pred_y - target) ** 2) / (torch.sum(weight * mask_any) + 1e-8)
    err2 = weight * (pred_y - target) ** 2
    w_c, w_b = weight * mask_ctd, weight * mask_bottle
    l_c = torch.sum(w_c * (pred_y - target) ** 2) / (torch.sum(w_c) + 1e-8)
    l_b = torch.sum(w_b * (pred_y - target) ** 2) / (torch.sum(w_b) + 1e-8)
    has_c = torch.sum(mask_ctd) > 0
    has_b = torch.sum(mask_bottle) > 0
    if has_c and has_b:
        return (1 - source_share) * l_c + source_share * l_b
    if has_b:
        return l_b
    if has_c:
        return l_c
    return torch.sum(err2) * 0.0


def compute_step_losses(model, physics, batch_x, batch_y, ocean, land, mse_loss, source_share=None):
    """Las 4 pérdidas de un paso (compartido por Adam y L-BFGS)."""
    pred = model(batch_x[:, 0:4])
    l_data = compute_weighted_data_loss(pred, batch_y, source_share)

    px = torch.cat([batch_x, ocean], dim=0) if ocean is not None else batch_x
    coords = px[:, 0:4].clone().requires_grad_(True)
    l_phys = physics.compute_physics_loss(model, coords, px[:, 4:7], px[:, 8:9], px[:, 7:8], valid_mask=px[:, 10:11])

    l_dir = physics.compute_dirichlet_loss(model, land[:, 0:4]) if land is not None else torch.zeros((), device=batch_x.device)

    chl_sat = batch_x[:, 9:10]
    mask_sat = (chl_sat > 0.01).squeeze(1)
    if mask_sat.any():
        xs = batch_x[mask_sat, 0:4].clone()
        xs[:, 2] = 0.0
        l_sat = mse_loss(model(xs), chl_sat[mask_sat])
    else:
        l_sat = torch.zeros((), device=batch_x.device)
    return l_data, l_phys, l_sat, l_dir


@torch.no_grad()
def evaluate_loader(model, loader, device, clim_prior=None, const_value=None):
    """Métricas sobre un loader (acumulando numeradores/denominadores, no medias de medias).
    - mse_w / skill_*: sobre TODAS las muestras (CTD + botella).
    - *_bottle: solo sobre botellas (referencia de laboratorio, métrica principal desde 2026-10-06).
    - rmse_log10 / rmse_log10_bottle: RMSE en log10(Chl+0.01).
    """
    model.eval()
    keys = ['se', 'w', 'se_c', 'se_k', 'e2', 'n', 'e2b', 'nb', 'se_b', 'w_b', 'se_cb', 'se_kb']
    a = {k: 0.0 for k in keys}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x[:, 0:4])
        m_ctd, m_bot, w = y[:, 2:3], y[:, 3:4], y[:, 4:5]
        target = y[:, 0:1] * m_ctd + y[:, 1:2] * m_bot
        m = torch.clamp(m_ctd + m_bot, max=1.0)
        se_all = (pred - target) ** 2
        a['se'] += float(torch.sum(w * m * se_all)); a['w'] += float(torch.sum(w * m))
        a['se_b'] += float(torch.sum(w * m_bot * se_all)); a['w_b'] += float(torch.sum(w * m_bot))
        if clim_prior is not None:
            c = clim_prior(x[:, 0:1], x[:, 1:2], x[:, 2:3])
            se_c = (c - target) ** 2
            a['se_c'] += float(torch.sum(w * m * se_c))
            a['se_cb'] += float(torch.sum(w * m_bot * se_c))
        if const_value is not None:
            se_k = (const_value - target) ** 2
            a['se_k'] += float(torch.sum(w * m * se_k))
            a['se_kb'] += float(torch.sum(w * m_bot * se_k))
        lp = torch.log10(torch.expm1(pred).clamp(min=0) + LOG10_EPS)
        lt = torch.log10(torch.expm1(target).clamp(min=0) + LOG10_EPS)
        e2 = (lp - lt) ** 2
        a['e2'] += float(torch.sum(e2 * m)); a['n'] += int(torch.sum(m))
        a['e2b'] += float(torch.sum(e2 * m_bot)); a['nb'] += int(torch.sum(m_bot))
    out = {'mse_w': a['se'] / max(a['w'], 1e-8), 'n': a['n'],
           'rmse_log10': (a['e2'] / max(a['n'], 1)) ** 0.5,
           'rmse_log10_bottle': (a['e2b'] / a['nb']) ** 0.5 if a['nb'] > 0 else float('nan'),
           'n_bottle': a['nb'],
           'mse_w_bottle': a['se_b'] / max(a['w_b'], 1e-8)}
    if clim_prior is not None:
        out['mse_clim'] = a['se_c'] / max(a['w'], 1e-8)
        out['skill_clim'] = 1.0 - out['mse_w'] / out['mse_clim']
        out['mse_clim_bottle'] = a['se_cb'] / max(a['w_b'], 1e-8)
        out['skill_clim_bottle'] = 1.0 - out['mse_w_bottle'] / out['mse_clim_bottle'] if a['w_b'] > 0 else float('nan')
    if const_value is not None:
        out['mse_const'] = a['se_k'] / max(a['w'], 1e-8)
        out['skill_const'] = 1.0 - out['mse_w'] / out['mse_const']
        out['mse_const_bottle'] = a['se_kb'] / max(a['w_b'], 1e-8)
        out['skill_const_bottle'] = 1.0 - out['mse_w_bottle'] / out['mse_const_bottle'] if a['w_b'] > 0 else float('nan')
    return out


# ----------------------------------------------------------------------------
# MLflow tolerante a caídas de red
# ----------------------------------------------------------------------------
def setup_mlflow():
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        try:
            urllib.request.urlopen(uri, timeout=8)
            reachable = True
        except urllib.error.HTTPError:
            reachable = True   # el servidor responde (p. ej. 401/404): está alcanzable
        except Exception as e:
            print(f"[Advertencia] MLflow remoto {uri} no alcanzable ({type(e).__name__}); uso SQLite local.")
            reachable = False
        if reachable:
            print(f"Conectando al servidor MLflow remoto: {uri}")
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment("PINNs_BajaCalifornia")
            return uri
    db_path = os.path.join(os.path.dirname(__file__), 'mlflow.db')
    print(f"Usando MLflow local (SQLite): {db_path}")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("PINNs_BajaCalifornia")
    return db_path


class SafeMlflow:
    """Un fallo de red al registrar métricas no debe matar un entrenamiento de horas."""
    def __init__(self):
        self.fails = 0

    def metrics(self, d, step):
        if self.fails >= 5:
            return
        try:
            mlflow.log_metrics(d, step=step)
            self.fails = 0
        except Exception as e:
            self.fails += 1
            print(f"[Advertencia] mlflow.log_metrics falló ({type(e).__name__}); {'se desactiva el registro remoto' if self.fails >= 5 else 'sigo entrenando'}.")

    def artifact(self, path):
        try:
            mlflow.log_artifact(path)
        except Exception as e:
            print(f"[Advertencia] No se pudo subir {os.path.basename(path)} a MLflow ({type(e).__name__}); queda en disco.")


# ----------------------------------------------------------------------------
# Entrenamiento
# ----------------------------------------------------------------------------
def train_pinn(epochs=10, batch_size=256, lr=1e-3, curriculum_epochs=5, colloc_ratio=0.5, lambda_sat=1.0,
               lambda_dirichlet=1.0, lbfgs_epochs=0, num_layers=6, hidden_dim=128, run_name="PINN_Training",
               use_fourier_features=False, fourier_mapping_size=64, fourier_scales=(3.0, 3.0, 1.0, 1.0),
               use_seasonal=False, use_climatology_prior=False,
               lambda_phys_max=1.0, ocean_colloc_ratio=1.0,
               lr_schedule="cosine", min_lr_frac=0.02, patience=None, smooth_window=5,
               fold="A", seed=0, source_share=None):
    """
    Experiment Harness. Protocolo de evaluación (2026-09-25):
    - val (elige checkpoint y para) y test (se evalúa UNA vez al final) son cruceros distintos
      (LOCO_FOLDS[fold]).
    - Se registran baselines (media y climatología de train) y el skill del modelo frente a ellos.
    - El mejor checkpoint se elige sobre Val_Loss SUAVIZADO (media móvil de `smooth_window`
      épocas) para no premiar un mínimo aislado de una serie ruidosa.
    - `patience`: épocas sin mejora del val suavizado tras las que se detiene (None = no parar).
    - `colloc_ratio`: puntos de tierra (Dirichlet) por punto de dato; `ocean_colloc_ratio`: puntos de
      colocación oceánica (PDE en huecos) por punto de dato.
    - `lambda_phys_max`: peso final de la PDE (antes 500 fijo, ajustado a una pérdida física
      subestimada por un bug de doble normalización; con el operador corregido hay que reajustarlo).
    """
    print("Iniciando Experiment Harness (PINN Training)...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    tracking_uri = setup_mlflow()
    safe = SafeMlflow()

    print(f"Preparando DataLoaders (fold {fold}: val≠test)...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size, fold=fold, return_test=True)
    train_ds = train_loader.dataset
    max_time_days = float(train_ds.df['time_days'].max())
    max_depth = float(train_ds.df['Depth'].max())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    land_data = load_land_points()
    ocean_pool = load_ocean_collocation_pool(train_ds.t0, device)

    # Estadísticas de normalización y climatología: SOLO del train
    dataset_x = train_ds.X[:, 0:4]
    mean_x = dataset_x.mean(dim=0).numpy()
    std_x = dataset_x.std(dim=0).numpy()
    y_tr, w_tr, has_tr = train_ds.target_arrays()
    lat_rng = (float(train_ds.df['Latitud'].min()) - 0.1, float(train_ds.df['Latitud'].max()) + 0.1)
    lon_rng = (float(train_ds.df['Longitud'].min()) - 0.1, float(train_ds.df['Longitud'].max()) + 0.1)
    grid, lz_rng = build_climatology_grid(
        train_ds.df['Latitud'].values[has_tr], train_ds.df['Longitud'].values[has_tr],
        train_ds.df['Depth'].values[has_tr], y_tr[has_tr], w_tr[has_tr], lat_rng, lon_rng)
    clim_prior = ClimatologyPrior(grid, lat_rng, lon_rng, lz_rng).to(device)
    const_value = float(np.sum(w_tr[has_tr] * y_tr[has_tr]) / np.sum(w_tr[has_tr]))

    model = CoastalPINNModel(
        num_layers=num_layers, hidden_dim=hidden_dim, input_mean=mean_x, input_std=std_x,
        use_fourier_features=use_fourier_features, fourier_mapping_size=fourier_mapping_size,
        fourier_scales=fourier_scales, use_seasonal=use_seasonal,
        climatology_prior=ClimatologyPrior(grid, lat_rng, lon_rng, lz_rng) if use_climatology_prior else None,
    ).to(device)
    physics = CoastalPhysicsPINN(diff_coef=0.1, std_x=torch.tensor(std_x, dtype=torch.float32, device=device)).to(device)

    base_val = evaluate_loader(model, val_loader, device, clim_prior, const_value)
    base_test = evaluate_loader(model, test_loader, device, clim_prior, const_value)
    print(f"BASELINES en val:  media={base_val['mse_const']:.4f}  climatología={base_val['mse_clim']:.4f}")
    print(f"BASELINES en test: media={base_test['mse_const']:.4f}  climatología={base_test['mse_clim']:.4f}")

    optimizer = optim.Adam(list(model.parameters()) + list(physics.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = (optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=lr * min_lr_frac)
                 if lr_schedule == "cosine" else None)
    mse_loss = torch.nn.MSELoss()

    with mlflow.start_run(run_name=run_name):
        try:
            mlflow.log_params({
                "epochs": epochs, "batch_size": batch_size, "colloc_ratio": colloc_ratio,
                "ocean_colloc_ratio": ocean_colloc_ratio, "learning_rate": lr, "lr_schedule": lr_schedule,
                "curriculum_epochs": curriculum_epochs, "lambda_sat": lambda_sat,
                "lambda_dirichlet": lambda_dirichlet, "lambda_phys_max": lambda_phys_max,
                "model_layers": num_layers, "hidden_dim": hidden_dim, "lbfgs_epochs": lbfgs_epochs,
                "use_fourier_features": use_fourier_features, "use_seasonal": use_seasonal,
                "use_climatology_prior": use_climatology_prior, "patience": patience, "fold": fold, "seed": seed,
                "val_baseline_const": round(base_val['mse_const'], 5), "val_baseline_clim": round(base_val['mse_clim'], 5),
            })
        except Exception as e:
            print(f"[Advertencia] log_params falló ({type(e).__name__}); sigo.")

        hist = {k: [] for k in ('epoch', 'Data_Loss', 'Physics_Loss', 'Sat_Loss', 'Dirichlet_Loss', 'Val_Loss',
                                'Val_RMSE_log10', 'Val_Skill_clim', 'lr')}
        state = dict(best=float('inf'), best_state=None, best_epoch=-1, since=0)

        def end_of_epoch(epoch, avg, lam_phys, cur_lr):
            v = evaluate_loader(model, val_loader, device, clim_prior, const_value)
            hist['epoch'].append(epoch); hist['Val_Loss'].append(v['mse_w']); hist['lr'].append(cur_lr)
            for k in ('Data_Loss', 'Physics_Loss', 'Sat_Loss', 'Dirichlet_Loss'):
                hist[k].append(avg[k])
            hist['Val_RMSE_log10'].append(v['rmse_log10']); hist['Val_Skill_clim'].append(v['skill_clim'])
            smooth = float(np.mean(hist['Val_Loss'][-smooth_window:]))
            improved = smooth < state['best'] - 1e-6
            if improved:
                state.update(best=smooth, best_epoch=epoch, since=0,
                             best_state={k: t.detach().cpu().clone() for k, t in model.state_dict().items()})
            else:
                state['since'] += 1
            safe.metrics({
                **{k: avg[k] for k in ('Data_Loss', 'Physics_Loss', 'Sat_Loss', 'Dirichlet_Loss')},
                "Val_Loss": v['mse_w'], "Val_Loss_smooth": smooth, "Val_RMSE_log10": v['rmse_log10'],
                "Val_RMSE_log10_bottle": v['rmse_log10_bottle'], "Val_Skill_clim": v['skill_clim'],
                "Val_Skill_const": v['skill_const'], "lambda_phys": lam_phys, "lr": cur_lr,
                "param_mu_max": float(physics.mu_max.abs()), "param_k_e": float(physics.k_e.abs()), "param_m": float(physics.m.abs()),
            }, step=epoch)
            return patience is not None and state['since'] >= patience

        stopped_early = False
        for epoch in range(epochs):
            model.train()
            acc = dict(Data_Loss=0.0, Physics_Loss=0.0, Sat_Loss=0.0, Dirichlet_Loss=0.0)
            lam_phys = min(1.0, epoch / max(curriculum_epochs, 1)) * lambda_phys_max
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_x, batch_y in pbar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                ocean, land = sample_collocation(batch_x.shape[0], land_data, ocean_pool, colloc_ratio,
                                                 ocean_colloc_ratio, max_time_days, max_depth, device)
                optimizer.zero_grad()
                l_data, l_phys, l_sat, l_dir = compute_step_losses(model, physics, batch_x, batch_y, ocean, land, mse_loss, source_share)
                loss = l_data + lam_phys * l_phys + lambda_sat * l_sat + lambda_dirichlet * l_dir
                loss.backward()
                optimizer.step()
                acc['Data_Loss'] += l_data.item(); acc['Physics_Loss'] += l_phys.item()
                acc['Sat_Loss'] += l_sat.item(); acc['Dirichlet_Loss'] += l_dir.item()
                pbar.set_postfix({"L_data": f"{l_data.item():.4f}", "L_phys": f"{l_phys.item():.4f}",
                                  "L_sat": f"{l_sat.item():.4f}", "L_dir": f"{l_dir.item():.4f}"})
            cur_lr = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                scheduler.step()
            avg = {k: v / len(train_loader) for k, v in acc.items()}
            if end_of_epoch(epoch, avg, lam_phys, cur_lr):
                print(f"\nEarly stopping en la época {epoch}: sin mejora del val suavizado en {patience} épocas "
                      f"(mejor en la época {state['best_epoch']}).")
                stopped_early = True
                break

        # ---------------- Fase L-BFGS (opcional, esquema "Golden Run") ----------------
        if lbfgs_epochs > 0 and not stopped_early:
            print(f"\nIniciando refinamiento de {lbfgs_epochs} epochs con optimizador L-BFGS...")
            for epoch in range(epochs, epochs + lbfgs_epochs):
                model.train()
                acc = dict(Data_Loss=0.0, Physics_Loss=0.0, Sat_Loss=0.0, Dirichlet_Loss=0.0)
                nan_detected = False
                pbar = tqdm(train_loader, desc=f"L-BFGS Epoch {epoch+1}/{epochs + lbfgs_epochs}")
                for batch_x, batch_y in pbar:
                    opt_l = optim.LBFGS(list(model.parameters()) + list(physics.parameters()), lr=0.01, max_iter=20,
                                        tolerance_grad=1e-7, tolerance_change=1e-9, history_size=50,
                                        line_search_fn="strong_wolfe")
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    ocean, land = sample_collocation(batch_x.shape[0], land_data, ocean_pool, colloc_ratio,
                                                     ocean_colloc_ratio, max_time_days, max_depth, device)

                    def closure():
                        opt_l.zero_grad()
                        d, p, s, r = compute_step_losses(model, physics, batch_x, batch_y, ocean, land, mse_loss, source_share)
                        t = d + lambda_phys_max * p + lambda_sat * s + lambda_dirichlet * r
                        t.backward()
                        return t

                    prev_state = {k: v.clone() for k, v in model.state_dict().items()}
                    opt_l.step(closure)
                    d, p, s, r = compute_step_losses(model, physics, batch_x, batch_y, ocean, land, mse_loss, source_share)
                    if not all(np.isfinite(float(z)) for z in (d, p)):
                        print("\n[Advertencia] NaN detectado en L-BFGS. Revirtiendo pesos y cancelando L-BFGS.")
                        model.load_state_dict(prev_state)
                        nan_detected = True
                        break
                    acc['Data_Loss'] += d.item(); acc['Physics_Loss'] += p.item()
                    acc['Sat_Loss'] += s.item(); acc['Dirichlet_Loss'] += r.item()
                    pbar.set_postfix({"L_data": f"{d.item():.4f}", "L_phys": f"{p.item():.4f}"})
                if nan_detected:
                    break
                avg = {k: v / len(train_loader) for k, v in acc.items()}
                if end_of_epoch(epoch, avg, lambda_phys_max, 0.01):
                    print(f"\nEarly stopping (L-BFGS) en la época {epoch}.")
                    break

        # ---------------- Selección final y TEST (una sola vez) ----------------
        print("Entrenamiento completado. Restaurando el mejor modelo según Val Loss suavizado...")
        if state['best_state'] is not None:
            model.load_state_dict(state['best_state'])
        final_val = evaluate_loader(model, val_loader, device, clim_prior, const_value)
        final_test = evaluate_loader(model, test_loader, device, clim_prior, const_value)
        print(f"Mejor época (val suavizado): {state['best_epoch']}  (val suavizado={state['best']:.4f})")
        print(f"VAL  : MSE={final_val['mse_w']:.4f}  skill_clim={final_val['skill_clim']:+.3f}  skill_media={final_val['skill_const']:+.3f}  "
              f"RMSE_log10={final_val['rmse_log10']:.3f}  (botella: {final_val['rmse_log10_bottle']:.3f}, n={final_val['n_bottle']})")
        print(f"TEST : MSE={final_test['mse_w']:.4f}  skill_clim={final_test['skill_clim']:+.3f}  skill_media={final_test['skill_const']:+.3f}  "
              f"RMSE_log10={final_test['rmse_log10']:.3f}  (botella: {final_test['rmse_log10_bottle']:.3f}, n={final_test['n_bottle']})")
        try:
            mlflow.log_metrics({
                "Best_Epoch": state['best_epoch'], "Best_Val_Smooth": state['best'],
                "Final_Val_MSE": final_val['mse_w'], "Final_Val_Skill_clim": final_val['skill_clim'],
                "Test_MSE": final_test['mse_w'], "Test_Skill_clim": final_test['skill_clim'],
                "Test_Skill_const": final_test['skill_const'], "Test_RMSE_log10": final_test['rmse_log10'],
                "Test_RMSE_log10_bottle": final_test['rmse_log10_bottle'],
                "Test_Skill_clim_bottle": final_test.get('skill_clim_bottle', float('nan')),
                "Test_Skill_const_bottle": final_test.get('skill_const_bottle', float('nan')),
                "Test_Baseline_clim": final_test['mse_clim'], "Test_Baseline_const": final_test['mse_const'],
            })
        except Exception as e:
            print(f"[Advertencia] No se pudieron registrar las métricas finales en MLflow ({type(e).__name__}).")

        here = os.path.dirname(__file__)
        model_path = os.path.join(here, f"pinn_model_{run_name}.pth")
        torch.save(model.state_dict(), model_path)
        safe.artifact(model_path)

        summary = dict(run_name=run_name, fold=fold, best_epoch=state['best_epoch'], stopped_early=stopped_early,
                       epochs_run=len(hist['epoch']), val=final_val, test=final_test,
                       baselines_val={k: base_val[k] for k in ('mse_const', 'mse_clim')},
                       baselines_test={k: base_test[k] for k in ('mse_const', 'mse_clim')},
                       physics_params={k: float(getattr(physics, k).abs()) for k in ('mu_max', 'k_e', 'm')})
        json_path = os.path.join(here, f"results_{run_name}.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        safe.artifact(json_path)

        metrics_df = pd.DataFrame(hist)
        csv_path = os.path.join(here, f"training_metrics_{run_name}.csv")
        metrics_df.to_csv(csv_path, index=False)
        safe.artifact(csv_path)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(hist['epoch'], hist['Data_Loss'], label='Train Data Loss', color='blue')
        ax.plot(hist['epoch'], hist['Val_Loss'], label='Val Loss (cruceros de val)', color='green')
        ax.axhline(base_val['mse_const'], color='gray', ls=':', label='Baseline: media del train')
        ax.axhline(base_val['mse_clim'], color='black', ls='--', label='Baseline: climatología')
        ax.axvline(state['best_epoch'], color='orange', ls='-.', label=f"Mejor época ({state['best_epoch']})")
        ax.plot(hist['epoch'], hist['Physics_Loss'], label='Physics Loss', color='red', alpha=0.6)
        ax.set_yscale('log'); ax.grid(True, which="both", ls="--", alpha=0.4); ax.legend()
        ax.set_title(f'Convergencia PINN 4D — {run_name}'); ax.set_xlabel('Epochs'); ax.set_ylabel('Pérdida (log)')
        png_path = os.path.join(here, f"training_metrics_{run_name}.png")
        fig.savefig(png_path, dpi=200, bbox_inches='tight'); plt.close(fig)
        safe.artifact(png_path)

        try:
            print("Generando mapa de inferencia final...")
            inf = plot_continuous_field(model_path, [23.82, 32.75], [-119.85, -111.92], depth=0.0, time_day=100.0,
                                        resolution=200, run_name=run_name, num_layers=num_layers, hidden_dim=hidden_dim)
            safe.artifact(inf)
        except Exception as e:
            print(f"[Advertencia] No se pudo generar el mapa de inferencia ({type(e).__name__}: {e}).")
        print(f"Artefactos y métricas registradas en {tracking_uri}")
    return summary


if __name__ == "__main__":
    train_pinn(epochs=300, batch_size=65536, lr=2e-3, curriculum_epochs=100, patience=60,
               use_seasonal=True, use_climatology_prior=True, fold="A")
