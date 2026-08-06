import torch
import torch.nn as nn

class CoastalPINNModel(nn.Module):
    """
    Agente 3 (NN Architect): Red Neuronal Informada por la Física (PINN).
    
    Esta red toma las coordenadas espaciotemporales (x, y, z, t) y predice 
    la concentración de Clorofila-a.
    """
    def __init__(self, num_layers=6, hidden_dim=128):
        super(CoastalPINNModel, self).__init__()
        
        # Entrada: (Latitud, Longitud, Profundidad, Tiempo) = 4 variables
        layers = [nn.Linear(4, hidden_dim), nn.Tanh()]
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        # Salida: (Clorofila-a) = 1 variable
        # Usamos Softplus al final para asegurar que la clorofila predicha sea siempre positiva
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Softplus())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        x shape: [batch_size, 4]
        return shape: [batch_size, 1]
        """
        return self.network(x)

if __name__ == "__main__":
    # Prueba rápida de forward pass
    model = CoastalPINNModel()
    dummy_input = torch.randn(10, 4)  # 10 puntos de prueba
    output = model(dummy_input)
    print("Dummy input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Muestra de predicción:", output[:3].detach().numpy())
