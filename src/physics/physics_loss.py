import torch
import torch.nn as nn

class CoastalPhysicsPINN(nn.Module):
    """
    Agente 4 (Physics Validator): Representa las leyes físicas y biogeoquímicas
    que gobiernan la dinámica oceánica.
    """
    def __init__(self, diff_coef=0.1, std_x=None):
        super().__init__()
        # Coeficiente de difusión Turbulenta (K) (Fijo por ahora)
        self.K = diff_coef
        self.std_x = std_x if std_x is not None else torch.ones(4)
        
        # --- PARÁMETROS BIOLÓGICOS ENTRENABLES (Física Inversa) ---
        # La red descubrirá estos valores durante el entrenamiento
        self.mu_max = nn.Parameter(torch.tensor([0.5]))  # Tasa máxima de crecimiento (1/día)
        self.k_e = nn.Parameter(torch.tensor([0.05]))    # Coef. atenuación luz (1/m)
        self.m = nn.Parameter(torch.tensor([0.1]))       # Tasa de mortalidad (1/día)

    def compute_physics_loss(self, model, X, u_velocities, temperature, bathymetry=None):
        """
        Calcula el residuo de la Ecuación Diferencial (Physics Loss) usando autograd.
        
        Parámetros:
        - model: La red neuronal (PINN) que predice la Clorofila.
        - X: Tensor de entrada [N, 4] -> (lat, lon, depth, time)
        - u_velocities: Tensor de velocidades [N, 3] -> (u, v, w)
        - temperature: Tensor de temperatura [N, 1]
        - bathymetry: Tensor de batimetría [N, 1] -> elevación (valores > 0 indican tierra)
        """
        if not X.requires_grad:
            X.requires_grad_(True)
            
        C_log = model(X)
        # CLIPPING DE SEGURIDAD: Evitar que al inicio del entrenamiento valores locos causen Infs en expm1
        C_log = torch.clamp(C_log, min=-1.0, max=5.0)
        
        # Deshacemos el logaritmo (expm1) para que el autograd y la PDE
        # operen matemáticamente en el espacio real físico de la clorofila.
        C_real = torch.expm1(C_log)
        
        dC_dX = torch.autograd.grad(
            C_real, X, grad_outputs=torch.ones_like(C_real),
            create_graph=True, retain_graph=True
        )[0]
        
        dC_dlat   = dC_dX[:, 0:1] / self.std_x[0]
        dC_dlon   = dC_dX[:, 1:2] / self.std_x[1]
        dC_ddepth = dC_dX[:, 2:3] / self.std_x[2]
        dC_dtime  = dC_dX[:, 3:4] / self.std_x[3]
        
        sec_per_day = 86400.0
        m_per_degree = 111139.0
        
        u = u_velocities[:, 0:1] * sec_per_day / m_per_degree
        v = u_velocities[:, 1:2] * sec_per_day / m_per_degree
        w = u_velocities[:, 2:3] * sec_per_day
        
        advection = u * dC_dlon + v * dC_dlat + w * dC_ddepth
        
        # 5. Ecuación de Reacción-Transporte Biogeoquímico
        z_phys = X[:, 2:3] * self.std_x[2]
        
        f_light = torch.exp(-torch.abs(self.k_e) * z_phys)
        
        T_max = 25.0
        T_min = 10.0
        f_nutrients = torch.clamp((T_max - temperature) / (T_max - T_min), 0.0, 1.0)
        
        growth = torch.abs(self.mu_max) * f_light * f_nutrients * C_real
        mortality = torch.abs(self.m) * C_real
        
        pde_residual = dC_dtime + advection - growth + mortality
        
        # --- MÁSCARA DE TIERRA ---
        if bathymetry is not None:
            land_mask = (bathymetry <= 0).float()
            pde_residual = pde_residual * land_mask
            
        physics_loss = torch.mean(pde_residual ** 2)
        
        return physics_loss
