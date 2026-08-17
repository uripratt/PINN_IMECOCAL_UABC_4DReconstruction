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
        Calcula el residuo de la Ecuación Diferencial (Physics Loss) usando autograd,
        matemáticamente transformada y evaluada directamente en el ESPACIO LOGARÍTMICO.
        """
        if not X.requires_grad:
            X.requires_grad_(True)
            
        # L = log1p(C) = log(C + 1)
        C_log = model(X)
        
        # Derivamos L directamente, EVITANDO expm1() que causaba Gradient Explosion
        dL_dX = torch.autograd.grad(
            C_log, X, grad_outputs=torch.ones_like(C_log),
            create_graph=True, retain_graph=True
        )[0]
        
        dL_dlat   = dL_dX[:, 0:1] / self.std_x[0]
        dL_dlon   = dL_dX[:, 1:2] / self.std_x[1]
        dL_ddepth = dL_dX[:, 2:3] / self.std_x[2]
        dL_dtime  = dL_dX[:, 3:4] / self.std_x[3]
        
        sec_per_day = 86400.0
        m_per_degree = 111139.0
        
        u = u_velocities[:, 0:1] * sec_per_day / m_per_degree
        v = u_velocities[:, 1:2] * sec_per_day / m_per_degree
        w = u_velocities[:, 2:3] * sec_per_day
        
        advection_log = u * dL_dlon + v * dL_dlat + w * dL_ddepth
        
        # --- PARTE BIOLÓGICA (Fuente / Sumidero) ---
        z_phys = X[:, 2:3] * self.std_x[2]
        f_light = torch.exp(-torch.abs(self.k_e) * z_phys)
        
        T_max = 25.0
        T_min = 10.0
        f_nutrients = torch.clamp((T_max - temperature) / (T_max - T_min), 0.0, 1.0)
        
        # Tasa de reacción Neta: R = (Crecimiento - Mortalidad)
        net_rate = (torch.abs(self.mu_max) * f_light * f_nutrients) - torch.abs(self.m)
        
        # Según la Regla de la Cadena, si L = log(C+1), el residuo en espacio log es:
        # dL/dt + u*dL/dx + ... = R * C / (C+1) = R * (1 - e^-L)
        # donde e^-L = exp(-C_log)
        source_log = net_rate * (1.0 - torch.exp(-C_log))
        
        # --- RESIDUO DE LA ECUACIÓN TRANSFORMADA ---
        pde_residual = dL_dtime + advection_log - source_log
        
        # --- MÁSCARA DE TIERRA ---
        if bathymetry is not None:
            land_mask = (bathymetry <= 0).float()
            pde_residual = pde_residual * land_mask
            
        physics_loss = torch.mean(pde_residual ** 2)
        
        return physics_loss
