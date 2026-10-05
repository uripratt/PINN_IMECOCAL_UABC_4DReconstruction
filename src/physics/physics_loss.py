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

    def compute_physics_loss(self, model, X, u_velocities, temperature, bathymetry=None, valid_mask=None):
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
        
        # X son coordenadas CRUDAS (grados, grados, m, días) y el modelo normaliza
        # internamente, así que el autograd YA devuelve dL/dX en unidades crudas
        # (verificado contra diferencias finitas, 2026-09-25: cociente 1.000).
        # Antes se dividía otra vez por std_x (doble normalización): dL/dt salía
        # ~1563x y dL/dz ~410x demasiado pequeñas, y z_phys = X*std_x hacía
        # f_light=exp(-k_e*z)~0 para cualquier z>1 m. self.std_x se conserva solo
        # por compatibilidad de API y ya no se usa.
        dL_dlat   = dL_dX[:, 0:1]
        dL_dlon   = dL_dX[:, 1:2]
        dL_ddepth = dL_dX[:, 2:3]
        dL_dtime  = dL_dX[:, 3:4]

        sec_per_day = 86400.0
        m_per_degree = 111139.0
        cos_lat = torch.cos(X[:, 0:1] * (3.141592653589793 / 180.0)).clamp(min=0.2)

        u = u_velocities[:, 0:1] * sec_per_day / (m_per_degree * cos_lat)   # grados de lon / día
        v = u_velocities[:, 1:2] * sec_per_day / m_per_degree              # grados de lat / día
        w = u_velocities[:, 2:3] * sec_per_day                             # m / día

        advection_log = u * dL_dlon + v * dL_dlat + w * dL_ddepth
        
        # --- PARTE BIOLÓGICA (Fuente / Sumidero) ---
        z_phys = X[:, 2:3]  # profundidad en metros (cruda)
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
        
        # --- MÁSCARAS ---
        # 1) Tierra (bathymetry>0): no hay agua donde exigir advección. Esto NO impone
        #    C=0 en tierra: eso es compute_dirichlet_loss() (ver Bug A, Sección 8.2).
        # 2) valid_mask (2026-09-25): 1 solo donde la forzante (u,v,w,T) es REAL. Donde
        #    era imputada (0 m/s, 15 C) exigir la PDE metería una física inventada.
        # La media se normaliza por el nº de puntos VÁLIDOS: antes se dividía entre
        # todos (incluidos ~4N puntos de tierra con residuo 0), diluyendo la pérdida
        # ~5x sin motivo.
        mask = torch.ones_like(pde_residual)
        if bathymetry is not None:
            mask = mask * (bathymetry <= 0).float()
        if valid_mask is not None:
            mask = mask * valid_mask.float()
        pde_residual = pde_residual * mask
        physics_loss = torch.sum(pde_residual ** 2) / torch.clamp(torch.sum(mask), min=1.0)

        return physics_loss

    def compute_dirichlet_loss(self, model, X_land):
        """
        Condición de frontera Dirichlet: penaliza que la clorofila predicha
        sea distinta de cero sobre tierra firme (X_land son puntos de
        colocación con bathymetry>0, generados en experiment_harness.py).

        Añadido 2026-09-03 (Fase 0 de propuesta_integracion_datos_pinn.md,
        Bug A): antes no existía ningún término que usara estos puntos para
        forzar C_tierra≈0 -- solo se anulaba su residuo de PDE, lo cual no es
        lo mismo. La salida del modelo está en espacio log1p (L=log(C+1)),
        así que penalizar L**2 ya empuja L→0, es decir C→0.
        """
        C_log_land = model(X_land)
        return torch.mean(C_log_land ** 2)
