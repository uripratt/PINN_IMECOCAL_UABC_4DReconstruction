import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_csv_metrics():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_csv = os.path.join(base_dir, "mlruns/mlruns_server/from_mlflow/Data_Loss(1).csv")
    phys_csv = os.path.join(base_dir, "mlruns/mlruns_server/from_mlflow/Physics_Loss(1).csv")
    
    df_data = pd.read_csv(data_csv)
    df_phys = pd.read_csv(phys_csv)
    
    df_data = df_data.sort_values(by='step')
    df_phys = df_phys.sort_values(by='step')

    plt.figure(figsize=(10, 6))
    plt.plot(df_data['step'], df_data['value'], label='Data Loss (MSE Observaciones)', color='blue', alpha=0.8, linewidth=1.5)
    plt.plot(df_phys['step'], df_phys['value'], label='Physics Loss (Residuo PDEs)', color='red', alpha=0.8, linewidth=1.5)
    
    plt.yscale('log')
    plt.title('Convergencia del Entrenamiento de la PINN 4D (5000 Epochs)', fontsize=14, pad=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Pérdida (Log Scale)', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    
    out_file = os.path.join(base_dir, "training_metrics_convergence_5000.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico generado en: {out_file}")

if __name__ == "__main__":
    plot_csv_metrics()
