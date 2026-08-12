import os
import subprocess

def download_in_batches():
    # Coordenadas y dominio
    lon_min, lon_max = -119.83, -112.00
    lat_min, lat_max = 23.83, 32.75
    
    # Directorios de salida
    thetao_dir = "data/raw/cmems_yearly_thetao"
    chl_dir = "data/raw/satellite_chl_yearly"
    os.makedirs(thetao_dir, exist_ok=True)
    os.makedirs(chl_dir, exist_ok=True)

    print("Iniciando descarga por lotes (año a año) para evitar OOM (Killed)...")

    for year in range(1998, 2013):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        thetao_file = os.path.join(thetao_dir, f"cmems_thetao_{year}.nc")
        if os.path.exists(thetao_file) and os.path.getsize(thetao_file) > 1000000:
            print(f"[{year}] Temperatura ya existe, saltando...")
        else:
            print(f"\n--- Descargando Temperatura (thetao) para {year} ---")
            thetao_cmd = [
                "copernicusmarine", "subset",
                "-i", "cmems_mod_glo_phy_my_0.083deg_P1D-m",
                "-x", str(lon_min), "-X", str(lon_max),
                "-y", str(lat_min), "-Y", str(lat_max),
                "-t", start_date, "-T", end_date,
                "-v", "thetao",
                "--output-directory", thetao_dir,
                "--output-filename", f"cmems_thetao_{year}.nc",
                "--force-download"
            ]
            subprocess.run(thetao_cmd)
        
        chl_file = os.path.join(chl_dir, f"satellite_chl_{year}.nc")
        if os.path.exists(chl_file) and os.path.getsize(chl_file) > 100000:
            print(f"[{year}] Clorofila Satelital ya existe, saltando...")
        else:
            print(f"\n--- Descargando Clorofila Satelital (CHL) para {year} ---")
            chl_cmd = [
                "copernicusmarine", "subset",
                "-i", "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D",
                "-x", str(lon_min), "-X", str(lon_max),
                "-y", str(lat_min), "-Y", str(lat_max),
                "-t", start_date, "-T", end_date,
                "-v", "CHL",
                "--output-directory", chl_dir,
                "--output-filename", f"satellite_chl_{year}.nc",
                "--force-download"
            ]
            subprocess.run(chl_cmd)

if __name__ == "__main__":
    download_in_batches()
