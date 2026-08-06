import os
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient

def plot_training_metrics(mlruns_dir="mlruns"):
    if not os.path.exists(mlruns_dir):
        print(f"Directorio {mlruns_dir} no encontrado. Asegúrate de copiarlo del servidor.")
        return

    mlflow.set_tracking_uri(f"file://{os.path.abspath(mlruns_dir)}")
    client = MlflowClient()
    
    # Obtener el experimento
    experiment = client.get_experiment_by_name("PINNs_BajaCalifornia")
    if experiment is None:
        print("No se encontró el experimento 'PINNs_BajaCalifornia'.")
        return
        
    # Obtener el último run (el más reciente)
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
    if not runs:
        print("No hay runs registrados.")
        return
        
    latest_run = runs[0]
    run_id = latest_run.info.run_id
    print(f"Analizando el Run ID: {run_id}")
    
    # Extraer métricas
    data_loss = client.get_metric_history(run_id, "Data_Loss")
    phys_loss = client.get_metric_history(run_id, "Physics_Loss")
    
    steps_data = [m.step for m in data_loss]
    steps_phys = [m.step for m in phys_loss]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps_data, [m.value for m in data_loss], label='Data Loss (MSE Observaciones)', color='blue', alpha=0.8, linewidth=1.5)
    plt.plot(steps_phys, [m.value for m in phys_loss], label='Physics Loss (Residuo PDEs)', color='red', alpha=0.8, linewidth=1.5)
    
    plt.yscale('log') # Escala logarítmica crucial para ver convergencia profunda
    plt.title('Convergencia del Entrenamiento de la PINN 4D', fontsize=14, pad=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Pérdida (Log Scale)', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    
    out_file = os.path.join(os.path.dirname(__file__), "training_metrics_convergence.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Gráfico de métricas de entrenamiento guardado en: {out_file}")

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    plot_training_metrics(os.path.join(current_dir, "mlruns"))
