import torch

class CoastalPhysicsPINN:
    """
    Agente 2 (Physical Modeler): Operadores de Física para la Red Neuronal.
    
    Este módulo implementa el cálculo de las Ecuaciones en Derivadas Parciales (PDEs)
    que gobiernan la dinámica oceánica. Para la clorofila, utilizamos la ecuación 
    de Advección-Difusión-Reacción.
    """
    def __init__(self, diff_coef=0.1, decay_rate=0.01):
        # Coeficiente de difusión Turbulenta (K) y tasa de mortalidad/hundimiento (R)
        self.K = diff_coef
        self.R = decay_rate

    def compute_physics_loss(self, model, X, u_velocities):
        """
        Calcula el residuo de la Ecuación Diferencial (Physics Loss) usando autograd.
        
        Parámetros:
        - model: La red neuronal (PINN) que predice la Clorofila.
        - X: Tensor de entrada [N, 4] -> (lat, lon, depth, time)
        - u_velocities: Tensor de velocidades [N, 2] -> (u, v) provistas por CMEMS
        """
        # Es fundamental que las entradas tengan 'requires_grad=True' 
        # para que PyTorch pueda calcular las derivadas parciales.
        if not X.requires_grad:
            X.requires_grad_(True)
            
        # Predicción de la concentración de Clorofila (C) en las coordenadas X
        C = model(X)
        
        # 1. Primeras Derivadas Espaciales y Temporales
        # dC_dX contiene: dC/dLat, dC/dLon, dC/dDepth, dC/dTime
        dC_dX = torch.autograd.grad(
            C, X, grad_outputs=torch.ones_like(C),
            create_graph=True, retain_graph=True
        )[0]
        
        dC_dlat   = dC_dX[:, 0:1]
        dC_dlon   = dC_dX[:, 1:2]
        dC_ddepth = dC_dX[:, 2:3]
        dC_dtime  = dC_dX[:, 3:4]
        
        # 2. Segundas Derivadas Espaciales (para la Difusión)
        d2C_dlat2 = torch.autograd.grad(
            dC_dlat, X, grad_outputs=torch.ones_like(dC_dlat),
            create_graph=True, retain_graph=True
        )[0][:, 0:1]
        
        d2C_dlon2 = torch.autograd.grad(
            dC_dlon, X, grad_outputs=torch.ones_like(dC_dlon),
            create_graph=True, retain_graph=True
        )[0][:, 1:2]
        
        # 3. Término de Advección (transporte por corrientes CMEMS)
        u = u_velocities[:, 0:1] # Zonal
        v = u_velocities[:, 1:2] # Meridional
        
        advection = u * dC_dlon + v * dC_dlat
        
        # 4. Término de Difusión
        diffusion = self.K * (d2C_dlon2 + d2C_dlat2)
        
        # 5. Ecuación de Reacción-Transporte (Residuo PDE)
        # dC/dt + u*grad(C) = K*laplaciano(C) - R*C
        pde_residual = dC_dtime + advection - diffusion + self.R * C
        
        # La pérdida física es el error cuadrático medio del residuo
        loss_physics = torch.mean(pde_residual ** 2)
        
        return loss_physics
