import unittest
import torch
import sys
import os

# Asegurar que se puede importar src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.pinn_model import CoastalPINNModel
from src.physics.physics_loss import CoastalPhysicsPINN

class TestCoastalPINN(unittest.TestCase):
    """
    Test Harness (Agente 4): Verificación Matemática Local.
    """
    def setUp(self):
        # Crear modelo e instanciar la clase de pérdida física
        self.model = CoastalPINNModel(num_layers=3, hidden_dim=32)
        self.physics = CoastalPhysicsPINN(diff_coef=0.1, decay_rate=0.01)

    def test_model_output_shape(self):
        """Verifica que el modelo devuelva el tensor de forma correcta."""
        # 10 muestras x 4 variables (lat, lon, depth, time)
        dummy_x = torch.randn(10, 4)
        out = self.model(dummy_x)
        self.assertEqual(out.shape, (10, 1), "El output debe ser [batch_size, 1]")

    def test_physics_loss_gradient_flow(self):
        """
        Verifica que el cálculo de la pérdida física devuelva un escalar 
        que preserve el grafo computacional (requiere gradientes para backprop).
        """
        dummy_x = torch.rand(10, 4, requires_grad=True)
        # Velocidades u, v simuladas
        dummy_u = torch.rand(10, 2)
        
        loss_p = self.physics.compute_physics_loss(self.model, dummy_x, dummy_u)
        
        self.assertTrue(torch.is_tensor(loss_p), "La pérdida física debe ser un tensor")
        self.assertEqual(loss_p.dim(), 0, "La pérdida física debe ser un escalar (0 dim)")
        self.assertTrue(loss_p.requires_grad, "La pérdida física debe mantener el grafo para backpropagation")

    def test_positivity_constraint(self):
        """Verifica que la concentración de clorofila nunca sea negativa."""
        dummy_x = torch.randn(100, 4) * 100 # Valores extremos
        out = self.model(dummy_x)
        self.assertTrue(torch.all(out >= 0), "La red no debe predecir valores de clorofila negativos")

if __name__ == '__main__':
    unittest.main()
