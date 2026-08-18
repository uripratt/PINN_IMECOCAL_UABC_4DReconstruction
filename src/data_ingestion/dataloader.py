import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class CoastalPINNDataset(Dataset):
    """
    Dataset Multi-Fidelidad para la red neuronal PINN.
    Carga el dataset 'imecocal_augmented.parquet'.
    Implementa validación cruzada estricta oceánica: Leave-One-Cruise-Out (LOCO).
    """
    def __init__(self, augmented_path=None, split='train', test_cruise_year=2005, test_cruise_month=4):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        self.data_path = augmented_path or os.path.join(project_root, 'data/processed/imecocal_augmented.parquet')
        
        print(f"Cargando dataset Multi-Fidelidad desde: {self.data_path}")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"No se encontró {self.data_path}. Ejecuta build_dataset.py primero.")
            
        self.df = pd.read_parquet(self.data_path)
        
        # QC: Si hay Fecha = NaT, la tiramos
        self.df = self.df.dropna(subset=['Fecha']).copy()
        self.df['Fecha'] = pd.to_datetime(self.df['Fecha'])
        
        # Global t0 for exact temporal matching between train and test
        self.t0 = self.df['Fecha'].min()
        
        # STRICT LOCO SPLIT (Leave-One-Cruise-Out)
        # Ocultamos la campaña definida por test_cruise_year y test_cruise_month
        is_test_cruise = (self.df['Fecha'].dt.year == test_cruise_year) & (self.df['Fecha'].dt.month == test_cruise_month)
        
        if split == 'train':
            self.df = self.df[~is_test_cruise].copy()
            print(f"Dataset TRAIN: Ocultando el crucero {test_cruise_year}-{test_cruise_month} (Filas: {len(self.df)})")
        elif split == 'test':
            self.df = self.df[is_test_cruise].copy()
            print(f"Dataset TEST (LOCO): Usando SOLO el crucero {test_cruise_year}-{test_cruise_month} (Filas: {len(self.df)})")
        else:
            raise ValueError("split debe ser 'train' o 'test'.")
            
        self.df = self.df.sort_values('Fecha').reset_index(drop=True)
        self._prepare_tensors()
        
    def _prepare_tensors(self):
        self.df['time_days'] = (self.df['Fecha'] - self.t0).dt.total_seconds() / (24 * 3600)
        
        # Llenamos NaNs en covariables físicas
        for col in ['uo', 'vo', 'wo', 'bathy', 'CHL_sat']:
            if col not in self.df.columns:
                self.df[col] = 0.0
            else:
                self.df[col] = self.df[col].fillna(0.0)
                
        if 'thetao' not in self.df.columns: self.df['thetao'] = 15.0
        else: self.df['thetao'] = self.df['thetao'].fillna(15.0)
            
        # X: (Lat, Lon, Depth, Time_days, u, v, w, bathy, temp, chl_sat)
        X_numpy = np.column_stack((
            self.df['Latitud'].values,
            self.df['Longitud'].values,
            self.df['Depth'].values, # Now we use Depth instead of Profundidad
            self.df['time_days'].values,
            self.df['uo'].values,
            self.df['vo'].values,
            self.df['wo'].values,
            self.df['bathy'].values,
            self.df['thetao'].values,
            np.log1p(self.df['CHL_sat'].values) # Satélite en Log
        ))
        
        # Multi-Fidelity Targets
        # CTD (Low Fidelity) - Contínuo
        chl_ctd = self.df['Chl_CTD'].values
        mask_ctd = ~np.isnan(chl_ctd)
        y_ctd = np.zeros_like(chl_ctd)
        y_ctd[mask_ctd] = np.log1p(np.clip(chl_ctd[mask_ctd], 0, None))
        
        # Bottles (High Fidelity) - Discreto
        chl_bottle = self.df['Chl_Bottle'].values
        mask_bottle = ~np.isnan(chl_bottle)
        y_bottle = np.zeros_like(chl_bottle)
        y_bottle[mask_bottle] = np.log1p(np.clip(chl_bottle[mask_bottle], 0, None))
        
        self.X = torch.tensor(X_numpy, dtype=torch.float32)
        
        # y contains [CTD_value, Bottle_value, CTD_mask, Bottle_mask]
        y_numpy = np.column_stack((y_ctd, y_bottle, mask_ctd.astype(float), mask_bottle.astype(float)))
        self.y = torch.tensor(y_numpy, dtype=torch.float32)
        
        print(f"Tensores: X shape {self.X.shape}. y shape {self.y.shape} (CTD, Bottle, Mask_CTD, Mask_Bottle)")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(batch_size=1024, test_cruise_year=2005, test_cruise_month=4):
    train_dataset = CoastalPINNDataset(split='train', test_cruise_year=test_cruise_year, test_cruise_month=test_cruise_month)
    test_dataset = CoastalPINNDataset(split='test', test_cruise_year=test_cruise_year, test_cruise_month=test_cruise_month)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

if __name__ == "__main__":
    print("Probando DataLoader Multi-Fidelidad LOCO...")
    train_loader, test_loader = get_dataloaders(batch_size=10)
    for x, y in train_loader:
        print("X shape:", x.shape)
        print("Y shape:", y.shape)
        break
