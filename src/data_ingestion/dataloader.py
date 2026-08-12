import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class CoastalPINNDataset(Dataset):
    """
    Dataset mesh-free para la red neuronal PINN.
    Carga el dataset 'imecocal_augmented.csv' preprocesado que ya 
    fusionó de forma segura la batimetría ETOPO y las velocidades CMEMS.
    """
    def __init__(self, augmented_csv_path=None, split='train', test_size=0.15, random_state=42):
        from sklearn.model_selection import train_test_split
        
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        self.csv_path = augmented_csv_path or os.path.join(project_root, 'data/processed/imecocal_augmented.csv')
        
        print(f"Cargando dataset preprocesado desde: {self.csv_path}")
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"No se encontró {self.csv_path}. Por favor ejecuta 'python src/data_ingestion/build_dataset.py' primero.")
            
        self.df = pd.read_csv(self.csv_path)
        self.df['Fecha'] = pd.to_datetime(self.df['Fecha'])
        
        # Eliminar cualquier NaN restante por seguridad
        self.df = self.df.dropna(subset=['Clorofila']).copy()
        
        # Definir t0 global antes del split para evitar desfases de tiempo entre train y test
        self.t0 = self.df['Fecha'].min()
        
        # SOTA: Profile Hold-out (Estrategia C - Ciegos Verticales)
        # En vez de separar filas aleatorias (lo que causa data leakage), separamos perfiles CTD completos.
        # Un perfil se define por una misma coordenada y fecha.
        profiles = self.df[['Latitud', 'Longitud', 'Fecha']].drop_duplicates()
        
        train_profiles, test_profiles = train_test_split(profiles, test_size=test_size, random_state=random_state)
        
        if split == 'train':
            self.df = self.df.merge(train_profiles, on=['Latitud', 'Longitud', 'Fecha'])
        elif split == 'test':
            self.df = self.df.merge(test_profiles, on=['Latitud', 'Longitud', 'Fecha'])
        else:
            raise ValueError("El parámetro 'split' debe ser 'train' o 'test'.")
            
        # Ordenamos por fecha por si acaso
        self.df = self.df.sort_values('Fecha').reset_index(drop=True)
        
        self._prepare_tensors()
        
    def _prepare_tensors(self):
        """
        Transforma el DataFrame limpio en tensores PyTorch.
        """
        # Convertir tiempo a un formato numérico (días desde el inicio global t0)
        self.df['time_days'] = (self.df['Fecha'] - self.t0).dt.total_seconds() / (24 * 3600)
        
        # Si no existen nuevas variables, inicializamos dummy
        if 'wo' not in self.df.columns: self.df['wo'] = 0.0
        if 'thetao' not in self.df.columns: self.df['thetao'] = 15.0
        if 'CHL_sat' not in self.df.columns: self.df['CHL_sat'] = 0.0
            
        # X: (Lat, Lon, Depth, Time_days, u, v, w, bathy, temp, chl_sat)
        X_numpy = np.column_stack((
            self.df['Latitud'].values,
            self.df['Longitud'].values,
            self.df['Profundidad'].values,
            self.df['time_days'].values,
            self.df['uo'].fillna(0.0).values,
            self.df['vo'].fillna(0.0).values,
            self.df['wo'].fillna(0.0).values,
            self.df['bathy'].fillna(0.0).values,
            self.df['thetao'].fillna(15.0).values,
            self.df['CHL_sat'].fillna(0.0).values
        ))
        
        # y: (Clorofila)
        y_numpy = self.df[['Clorofila']].values
        
        self.X = torch.tensor(X_numpy, dtype=torch.float32)
        self.y = torch.tensor(y_numpy, dtype=torch.float32)
        
        print(f"Tensores preparados: X shape {self.X.shape}, y shape {self.y.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_x = self.X[idx]
        sample_y = self.y[idx]
        
        # Retornamos (X, y)
        return sample_x, sample_y

def get_dataloaders(batch_size=256, test_size=0.15, random_state=42):
    train_dataset = CoastalPINNDataset(split='train', test_size=test_size, random_state=random_state)
    test_dataset = CoastalPINNDataset(split='test', test_size=test_size, random_state=random_state)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

if __name__ == "__main__":
    # Test rápido del Dataset
    print("Probando instanciación del Dataset de la PINN...")
    ds = CoastalPINNDataset()
    
    if len(ds) > 0:
        x_sample, y_sample = ds[0]
        print("\nEjemplo de Muestra 0:")
        print(f"  Inputs (Lat, Lon, Prof, Tiempo_Dias, u, v, w, bathy, temp, chl_sat): {x_sample}")
        print(f"  Target (Clorofila): {y_sample}")
