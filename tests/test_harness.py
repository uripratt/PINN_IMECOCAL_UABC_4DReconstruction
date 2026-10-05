import unittest
import numpy as np
import torch
import sys
import os

# Asegurar que se puede importar src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.pinn_model import CoastalPINNModel
from src.physics.physics_loss import CoastalPhysicsPINN
from src.models.climatology import build_climatology_grid, ClimatologyPrior

class TestCoastalPINN(unittest.TestCase):
    """
    Test Harness (Agente 4): Verificación Matemática Local.
    """
    def setUp(self):
        # Crear modelo e instanciar la clase de pérdida física
        self.model = CoastalPINNModel(num_layers=3, hidden_dim=32)
        # decay_rate no es un argumento del constructor actual (la mortalidad
        # `m` es un nn.Parameter interno, no se pasa desde fuera) -- eliminado.
        self.physics = CoastalPhysicsPINN(diff_coef=0.1)

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
        # Velocidades u, v, w simuladas (compute_physics_loss indexa las 3
        # componentes: u_velocities[:, 0:1], [:, 1:2], [:, 2:3])
        dummy_u = torch.rand(10, 3)
        # Temperatura simulada -- argumento obligatorio de compute_physics_loss
        dummy_temp = torch.rand(10, 1)

        loss_p = self.physics.compute_physics_loss(self.model, dummy_x, dummy_u, dummy_temp)
        
        self.assertTrue(torch.is_tensor(loss_p), "La pérdida física debe ser un tensor")
        self.assertEqual(loss_p.dim(), 0, "La pérdida física debe ser un escalar (0 dim)")
        self.assertTrue(loss_p.requires_grad, "La pérdida física debe mantener el grafo para backpropagation")

    def test_positivity_constraint(self):
        """Verifica que la concentración de clorofila nunca sea negativa."""
        dummy_x = torch.randn(100, 4) * 100 # Valores extremos
        out = self.model(dummy_x)
        self.assertTrue(torch.all(out >= 0), "La red no debe predecir valores de clorofila negativos")

class TestNuevasPiezas(unittest.TestCase):
    """Regresiones de la revisión del 2026-09-25."""
    MEAN = np.array([28.3, -116.2, 508.0, 3748.0])
    STD = np.array([2.11, 1.54, 410.3, 1562.6])

    def test_autograd_da_derivada_cruda_sin_doble_normalizacion(self):
        """El modelo normaliza internamente: dL/dX_crudo del autograd debe coincidir con
        diferencias finitas en unidades crudas (bug: la física dividía otra vez por std)."""
        torch.manual_seed(0)
        m = CoastalPINNModel(num_layers=3, hidden_dim=32, input_mean=self.MEAN, input_std=self.STD)
        X = torch.tensor([[29.0, -115.0, 30.0, 4000.0]], requires_grad=True)
        g = torch.autograd.grad(m(X), X)[0][0].detach().numpy()
        fd = []
        for i, h in enumerate([1e-2, 1e-2, 1.0, 5.0]):
            xp, xm = X.detach().clone(), X.detach().clone()
            xp[0, i] += h; xm[0, i] -= h
            fd.append(float((m(xp) - m(xm)) / (2 * h)))
        np.testing.assert_allclose(g, np.array(fd), rtol=5e-2, atol=1e-7)

    def test_valid_mask_anula_y_normaliza(self):
        torch.manual_seed(0)
        m = CoastalPINNModel(num_layers=2, hidden_dim=16, input_mean=self.MEAN, input_std=self.STD)
        ph = CoastalPhysicsPINN()
        X = torch.tensor([[29.0, -115.0, 30.0, 4000.0]] * 8, requires_grad=True)
        uvw, T, bathy = torch.rand(8, 3) * 0.1, torch.full((8, 1), 15.0), torch.full((8, 1), -500.0)
        l_all = ph.compute_physics_loss(m, X.clone(), uvw, T, bathy, valid_mask=torch.ones(8, 1))
        l_none = ph.compute_physics_loss(m, X.clone(), uvw, T, bathy, valid_mask=torch.zeros(8, 1))
        self.assertEqual(float(l_none), 0.0)
        # duplicar los puntos con máscara 0 no debe diluir la pérdida (antes se dividía entre todos)
        X2 = torch.cat([X, X]).detach().requires_grad_(True)
        l_dil = ph.compute_physics_loss(m, X2, torch.cat([uvw, uvw]), torch.cat([T, T]), torch.cat([bathy, bathy]),
                                        valid_mask=torch.cat([torch.ones(8, 1), torch.zeros(8, 1)]))
        self.assertAlmostEqual(float(l_all), float(l_dil), places=6)

    def test_prior_climatologico_arranca_en_la_climatologia(self):
        rng = np.random.default_rng(0)
        n = 5000
        lat, lon, dep = rng.uniform(24, 32, n), rng.uniform(-120, -112, n), rng.uniform(0, 300, n)
        y = 0.5 * np.exp(-dep / 60.0)          # perfil sintético
        grid, lz = build_climatology_grid(lat, lon, dep, y, np.ones(n), (24, 32), (-120, -112))
        prior = ClimatologyPrior(grid, (24, 32), (-120, -112), lz)
        m = CoastalPINNModel(num_layers=2, hidden_dim=16, input_mean=self.MEAN, input_std=self.STD, climatology_prior=prior)
        X = torch.tensor([[28.0, -116.0, 10.0, 3000.0]])
        clim = float(prior(X[:, 0:1], X[:, 1:2], X[:, 2:3]))
        self.assertAlmostEqual(float(m(X)), clim, delta=0.02)   # capa final a cero => sale la climatología
        self.assertTrue(float(m(X)) > 0)

    def test_from_state_dict_reconstruye_todas_las_variantes(self):
        grid = np.random.rand(20, 44, 48).astype(np.float32)
        prior = ClimatologyPrior(grid, (24, 32), (-120, -112), (0.0, 8.0))
        for kw in (dict(), dict(use_fourier_features=True, fourier_mapping_size=8), dict(use_seasonal=True),
                   dict(use_seasonal=True, climatology_prior=prior, use_fourier_features=True, fourier_mapping_size=8)):
            m = CoastalPINNModel(num_layers=2, hidden_dim=16, input_mean=self.MEAN, input_std=self.STD, **kw)
            m2 = CoastalPINNModel.from_state_dict(m.state_dict(), num_layers=2, hidden_dim=16)
            X = torch.rand(6, 4) * torch.tensor([8, 8, 300, 5000.0]) + torch.tensor([24, -120, 0, 0.0])
            torch.testing.assert_close(m(X), m2(X))

if __name__ == '__main__':
    unittest.main()
