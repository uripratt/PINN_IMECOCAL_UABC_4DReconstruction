import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Folds LOCO (Leave-One-Cruise-Out) con cruceros de VALIDACIÓN y de TEST distintos,
# en estaciones y años distintos, todos dentro del periodo con forzante CMEMS
# (2001-2012) y con botella propia (>=240 filas). Códigos = `crucero_code` (AAMM).
#  - val: se usa para elegir checkpoint / parar / comparar configuraciones.
#  - test: se evalúa UNA sola vez, con el modelo ya elegido; nunca interviene en la selección.
LOCO_FOLDS = {
    "A": {"val": (504,), "test": (807,)},    # val primavera 2005   | test verano 2008
    "B": {"val": (1101,), "test": (404,)},   # val invierno 2011    | test primavera 2004
    "C": {"val": (307,), "test": (910,)},    # val verano 2003      | test otoño 2009
}


class CoastalPINNDataset(Dataset):
    """
    Dataset Multi-Fidelidad para la red neuronal PINN ('imecocal_augmented.parquet').

    split='train' excluye los cruceros de val y test; 'val' y 'test' devuelven solo
    esos cruceros. t0 se calcula con TODAS las filas antes de partir, así que
    time_days es coherente entre splits.

    X = (lat, lon, depth, time_days, u, v, w, bathy, thetao, chl_sat_log, forcing_ok)
    y = (y_ctd, y_bottle, mask_ctd, mask_bottle, peso)   [y en log1p(Chl)]
    """

    def __init__(self, augmented_path=None, split='train', val_cruises=(504,), test_cruises=(807,),
                 test_cruise_year=None, test_cruise_month=None):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        self.data_path = augmented_path or os.path.join(project_root, 'data/processed/imecocal_augmented.parquet')

        print(f"Cargando dataset Multi-Fidelidad desde: {self.data_path}")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"No se encontró {self.data_path}. Ejecuta build_dataset.py primero.")

        self.df = pd.read_parquet(self.data_path)
        self.df = self.df.dropna(subset=['Fecha']).copy()
        self.df['Fecha'] = pd.to_datetime(self.df['Fecha'])
        self.t0 = self.df['Fecha'].min()

        if 'crucero_code' in self.df.columns and test_cruise_year is None:
            code = pd.to_numeric(self.df['crucero_code'], errors='coerce')
            is_val = code.isin(list(val_cruises)).values
            is_test = code.isin(list(test_cruises)).values
            desc = f"val={tuple(val_cruises)} test={tuple(test_cruises)}"
        else:
            # Compatibilidad con la interfaz antigua (año-mes): el único crucero retenido hacía de val.
            yy, mm = (test_cruise_year or 2005), (test_cruise_month or 4)
            is_val = ((self.df['Fecha'].dt.year == yy) & (self.df['Fecha'].dt.month == mm)).values
            is_test = np.zeros(len(self.df), dtype=bool)
            desc = f"val={yy}-{mm} (interfaz antigua)"

        if split == 'train':
            self.df = self.df[~(is_val | is_test)].copy()
        elif split == 'val':
            self.df = self.df[is_val].copy()
        elif split == 'test':
            self.df = self.df[is_test].copy()
        else:
            raise ValueError("split debe ser 'train', 'val' o 'test'.")
        print(f"Dataset {split.upper()} ({desc}): {len(self.df)} filas")

        self.df = self.df.sort_values('Fecha').reset_index(drop=True)
        self._prepare_tensors()

    def _prepare_tensors(self):
        self.df['time_days'] = (self.df['Fecha'] - self.t0).dt.total_seconds() / (24 * 3600)

        for col in ['uo', 'vo', 'wo', 'bathy', 'CHL_sat']:
            if col not in self.df.columns:
                self.df[col] = 0.0
            else:
                self.df[col] = self.df[col].fillna(0.0)
        if 'thetao' not in self.df.columns:
            self.df['thetao'] = 15.0
        else:
            self.df['thetao'] = self.df['thetao'].fillna(15.0)

        # forcing_ok: 1 donde u,v,w,T son forzante REAL (ver build_dataset.py). Con un
        # parquet antiguo sin la columna se estima con una heurística (y se avisa).
        if 'forcing_ok' in self.df.columns:
            forcing_ok = self.df['forcing_ok'].astype(bool).values
        else:
            print("[Advertencia] El parquet no tiene 'forcing_ok' (versión antigua): se estima como "
                  "(u!=0 o v!=0) y thetao!=15. Regenera con build_dataset.py actualizado.")
            forcing_ok = (((self.df['uo'] != 0) | (self.df['vo'] != 0)) & (self.df['thetao'] != 15.0)).values

        # Mapeo satélite -> in-situ (CORREGIDO 2026-09-25). La Fase 0 invertía la regresión
        # FORWARD log10(sat)=0.31*log10(insitu)-0.42; pero esa pendiente baja es dilución de
        # regresión (se condiciona por la variable ruidosa), y su inversa amplifica x3 en log10:
        # en cruceros no vistos daba RMSE_log10=0.93 (peor que el satélite crudo, 0.46, y que una
        # constante, 0.57) y valores de hasta 1.7e6 mg/m3. Lo correcto es la esperanza condicional
        # (regresión INVERSA, ajustada con el match-up N=9.189 y validada en cruceros no vistos):
        #     log10(insitu) = 1.0124*log10(sat) + 0.0900      RMSE_log10 = 0.44
        # (pendiente ~1: el satélite es casi insesgado). Se acota a 30 mg/m3 (techo físico del pipeline).
        SAT_SLOPE = 1.0124
        SAT_INTERCEPT = 0.0900
        raw_sat = self.df['CHL_sat'].values
        chl_sat_corrected = np.where(
            raw_sat > 0,
            np.clip(10 ** (SAT_SLOPE * np.log10(np.clip(raw_sat, 1e-6, None)) + SAT_INTERCEPT), 0.0, 30.0),
            0.0,
        )

        X_numpy = np.column_stack((
            self.df['Latitud'].values,
            self.df['Longitud'].values,
            self.df['Depth'].values,
            self.df['time_days'].values,
            self.df['uo'].values,
            self.df['vo'].values,
            self.df['wo'].values,
            self.df['bathy'].values,
            self.df['thetao'].values,
            np.log1p(chl_sat_corrected),
            forcing_ok.astype(np.float32),
        ))

        chl_ctd = self.df['Chl_CTD'].values
        mask_ctd = ~np.isnan(chl_ctd)
        y_ctd = np.zeros_like(chl_ctd)
        y_ctd[mask_ctd] = np.log1p(np.clip(chl_ctd[mask_ctd], 0, None))

        chl_bottle = self.df['Chl_Bottle'].values
        mask_bottle = ~np.isnan(chl_bottle)
        y_bottle = np.zeros_like(chl_bottle)
        y_bottle[mask_bottle] = np.log1p(np.clip(chl_bottle[mask_bottle], 0, None))

        if 'peso_entrenamiento' in self.df.columns:
            weight = self.df['peso_entrenamiento'].fillna(1.0).values
        else:
            print("[Advertencia] 'peso_entrenamiento' no está en el parquet -- usando peso=1.0.")
            weight = np.ones_like(chl_ctd)

        self.X = torch.tensor(X_numpy, dtype=torch.float32)
        y_numpy = np.column_stack((y_ctd, y_bottle, mask_ctd.astype(float), mask_bottle.astype(float), weight))
        self.y = torch.tensor(y_numpy, dtype=torch.float32)
        print(f"Tensores: X {tuple(self.X.shape)}, y {tuple(self.y.shape)}; forzante real en "
              f"{forcing_ok.mean():.1%} de las filas")

    def target_arrays(self):
        """(y_log1p, peso) unificados sobre las filas con dato, para climatología y baselines."""
        y = (self.y[:, 0] * self.y[:, 2] + self.y[:, 1] * self.y[:, 3]).numpy()
        has = ((self.y[:, 2] + self.y[:, 3]) > 0).numpy()
        return y, self.y[:, 4].numpy(), has

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(batch_size=1024, val_cruises=(504,), test_cruises=(807,), return_test=False,
                    test_cruise_year=None, test_cruise_month=None, fold=None):
    """Devuelve (train, val) o, con return_test=True, (train, val, test).
    `fold` ('A','B','C') selecciona los cruceros de LOCO_FOLDS."""
    if fold is not None:
        val_cruises, test_cruises = LOCO_FOLDS[fold]["val"], LOCO_FOLDS[fold]["test"]
    kw = dict(val_cruises=val_cruises, test_cruises=test_cruises,
              test_cruise_year=test_cruise_year, test_cruise_month=test_cruise_month)
    train_dataset = CoastalPINNDataset(split='train', **kw)
    val_dataset = CoastalPINNDataset(split='val', **kw)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    if not return_test:
        return train_loader, val_loader
    test_dataset = CoastalPINNDataset(split='test', **kw)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Probando DataLoader Multi-Fidelidad LOCO...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=10, return_test=True)
    for x, y in train_loader:
        print("X shape:", x.shape)
        print("Y shape:", y.shape)
        break
