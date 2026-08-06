import xarray as xr
import pandas as pd
import numpy as np
import os
import urllib.request
from datetime import datetime

class ERDDAPDownloader:
    """
    Clase para interactuar con servidores ERDDAP y descargar covariables físicas (SST, corrientes, batimetría)
    mediante OpenDAP (lazy loading con xarray).
    """
    def __init__(self, output_dir="../../data/raw"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Servidores públicos conocidos
        self.coastwatch_url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
        
    def get_bounding_box_from_csv(self, csv_path):
        """Extrae el dominio espacial y temporal de un dataset in-situ"""
        df = pd.read_excel(csv_path) if csv_path.endswith('.xlsx') else pd.read_csv(csv_path)
        
        # Añadir un pequeño buffer (0.5 grados) al dominio
        min_lat, max_lat = df['Latitud'].min() - 0.5, df['Latitud'].max() + 0.5
        min_lon, max_lon = df['Longitud'].min() - 0.5, df['Longitud'].max() + 0.5
        
        # Para el tiempo, tomar el min y max año/mes
        min_year, max_year = df['Año'].min(), df['Año'].max()
        
        print(f"Dominio detectado: Lat[{min_lat:.2f}, {max_lat:.2f}], Lon[{min_lon:.2f}, {max_lon:.2f}]")
        print(f"Periodo temporal: {min_year} - {max_year}")
        return (min_lat, max_lat, min_lon, max_lon, f"{min_year}-01-01", f"{max_year}-12-31")

    def download_bathymetry(self, bounds):
        """Descarga la batimetría ETOPO (resolución global) para el dominio"""
        min_lat, max_lat, min_lon, max_lon, _, _ = bounds
        print("\n--- Descargando Batimetría (ETOPO) ---")
        
        dataset_id = "etopo180"
        url = f"{self.coastwatch_url}/{dataset_id}.nc?altitude[({min_lat}):1:({max_lat})][({min_lon}):1:({max_lon})]"
        
        output_file = os.path.join(self.output_dir, "etopo_bathymetry.nc")
        try:
            urllib.request.urlretrieve(url, output_file)
            print(f"Éxito: Batimetría guardada en {output_file}")
            
            # Verificar con xarray
            ds = xr.open_dataset(output_file)
            print(ds)
            ds.close()
        except Exception as e:
            print(f"Error descargando batimetría: {e}")

    def download_sst_opendap(self, bounds):
        """
        Descarga Temperatura Superficial del Mar (OISST v2.1) usando OpenDAP vía xarray.
        Es mucho más eficiente porque solo descarga el slice requerido.
        """
        min_lat, max_lat, min_lon, max_lon, start_time, end_time = bounds
        print("\n--- Conectando a OISST (Temperatura Superficial) vía OpenDAP ---")
        
        dataset_id = "ncdcOisst21Agg_LonPM180"
        opendap_url = f"{self.coastwatch_url}/{dataset_id}"
        
        try:
            # Lazy loading
            ds = xr.open_dataset(opendap_url)
            
            # Slice espaciotemporal
            sliced_ds = ds.sel(
                time=slice(start_time, end_time),
                latitude=slice(min_lat, max_lat),
                longitude=slice(min_lon, max_lon)
            )
            
            print(f"Slice seleccionado: {sliced_ds.dims}")
            
            # Guardar a netCDF local
            output_file = os.path.join(self.output_dir, "oisst_1998_2012_slice.nc")
            print(f"Descargando datos (esto puede tomar unos minutos dependiendo del tamaño temporal)...")
            sliced_ds.to_netcdf(output_file)
            print(f"Éxito: SST histórico guardado en {output_file}")
            ds.close()
            
        except Exception as e:
            print(f"Error procesando SST con OpenDAP: {e}")

    def download_cmems_currents_placeholder(self, bounds):
        """
        Las corrientes 3D (u,v,w) históricas se obtienen mejor de CMEMS (GLORYS12V1).
        Copernicus requiere autenticación. Este es el placeholder arquitectónico.
        """
        print("\n--- Preparación para Corrientes 3D (CMEMS GLORYS12) ---")
        print("Para corrientes 3D históricas de 1998-2012, el dataset estándar es GLOBAL_MULTIYEAR_PHY_001_030.")
        print("Sugerencia: Usar el paquete 'copernicusmarine' en Python para la descarga autenticada:")
        print(f"copernicusmarine subset -i cmems_mod_glo_phy_my_0.083_P1D-m -x {bounds[2]} -X {bounds[3]} -y {bounds[0]} -Y {bounds[1]} -t {bounds[4]} -T {bounds[5]} -v uo -v vo")

if __name__ == "__main__":
    # La ruta al dataset in-situ es relativa a donde ejecutemos el script
    # Asumiremos que ejecutamos desde la raíz o pasamos el path absoluto
    dataset_path = "../../Cl_Imec98_12.xlsx"
    
    if not os.path.exists(dataset_path):
        # Fallback si se ejecuta desde el directorio de UABC_Ensenada
        dataset_path = "Cl_Imec98_12.xlsx"
        downloader = ERDDAPDownloader(output_dir="data/raw")
    else:
        downloader = ERDDAPDownloader()
        
    bounds = downloader.get_bounding_box_from_csv(dataset_path)
    
    downloader.download_bathymetry(bounds)
    # downloader.download_sst_opendap(bounds) # Comentado inicialmente para probar rápido
    downloader.download_cmems_currents_placeholder(bounds)
    
    print("\nFase 1 (ERDDAP) Inicializada.")
