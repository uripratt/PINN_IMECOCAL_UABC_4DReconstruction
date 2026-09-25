import math
import torch
import torch.nn as nn


class AnisotropicFourierFeatures(nn.Module):
    """
    Embedding posicional de Fourier aleatorio (Tancik et al., 2020), con una
    varianza de frecuencia distinta POR DIMENSIÓN de entrada ("anisotrópico").

    Motivación (2026-09-14, diagnóstico de rendimiento val/test): la PINN
    actual pasa (lat, lon, z, t) normalizados directamente a una MLP con
    Tanh, que sufre el "sesgo espectral" bien documentado de este tipo de
    redes hacia funciones de baja frecuencia -- ver
    `propuesta_integracion_datos_pinn.md`, Sección 11, y
    `comands/INR_PINN_Embedding_Anisotropic` (donde ya se había propuesto
    esta misma solución para un dominio distinto, OBSEA/Mediterráneo, pero
    el razonamiento matemático es el mismo). Esto es consistente con lo que
    el propio informe (`Reporte_IMECOCAL_Descriptivo.tex`, Sección de
    Resultados) ya documenta empíricamente: las reconstrucciones PINN
    pierden toda la estructura de mesoescala (filamentos, parches) visible
    en el satélite, convergiendo a un único núcleo suave.

    `scales` fija la desviación estándar de las frecuencias muestreadas por
    dimensión de entrada -- más alta = permite representar variación más
    fina en esa dimensión. Por defecto se da más ancho de banda a
    latitud/longitud (para intentar resolver estructura horizontal de
    mesoescala, ~10-50km, frente al núcleo único actual) que a profundidad y
    tiempo, pero es un hiperparámetro a explorar, no un valor derivado
    físicamente.

    Las frecuencias (`B`) se registran como buffer, NO como parámetro
    entrenable -- es la formulación estándar de Random Fourier Features.
    """

    def __init__(self, in_dim=4, mapping_size=64, scales=(3.0, 3.0, 1.0, 1.0), seed=0):
        super().__init__()
        assert len(scales) == in_dim, "Debe darse una escala de frecuencia por dimensión de entrada"
        generator = torch.Generator().manual_seed(seed)
        B = torch.randn(in_dim, mapping_size, generator=generator)
        # Escala cada FILA (dimensión de entrada) por su propia sigma -> anisotrópico
        scale_vec = torch.tensor(scales, dtype=torch.float32).unsqueeze(1)  # (in_dim, 1)
        B = B * scale_vec
        self.register_buffer('B', B)
        self.out_dim = in_dim + 2 * mapping_size  # se concatena la entrada cruda + sin/cos

    def forward(self, x):
        # x: (batch, in_dim), ya normalizado (Z-score) antes de llegar aquí
        proj = 2 * math.pi * (x @ self.B)  # (batch, mapping_size)
        return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)


class CoastalPINNModel(nn.Module):
    """
    Agente 3 (NN Architect): Red Neuronal Informada por la Física (PINN).

    Esta red toma las coordenadas espaciotemporales (x, y, z, t) y predice
    la concentración de Clorofila-a.
    """
    def __init__(self, num_layers=6, hidden_dim=128, input_mean=None, input_std=None,
                 use_fourier_features=False, fourier_mapping_size=64,
                 fourier_scales=(3.0, 3.0, 1.0, 1.0)):
        super(CoastalPINNModel, self).__init__()

        # --- NORMALIZACIÓN INTEGRADA ---
        # Registramos medias y desviaciones como "buffers" (se guardan en el .pth pero no se entrenan)
        if input_mean is not None and input_std is not None:
            self.register_buffer('input_mean', torch.tensor(input_mean, dtype=torch.float32))
            self.register_buffer('input_std', torch.tensor(input_std, dtype=torch.float32))
            self.normalize = True
        else:
            self.normalize = False

        # --- EMBEDDING POSICIONAL OPCIONAL (desactivado por defecto: no rompe
        # checkpoints/arneses existentes que instancian el modelo sin este
        # argumento) --- ver AnisotropicFourierFeatures arriba.
        self.use_fourier_features = use_fourier_features
        if use_fourier_features:
            self.fourier = AnisotropicFourierFeatures(
                in_dim=4, mapping_size=fourier_mapping_size, scales=fourier_scales
            )
            first_in = self.fourier.out_dim
        else:
            self.fourier = None
            first_in = 4

        # Entrada: (Latitud, Longitud, Profundidad, Tiempo) = 4 variables
        # (o su embedding de Fourier, si use_fourier_features=True)
        layers = [nn.Linear(first_in, hidden_dim), nn.Tanh()]

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
        if self.normalize:
            # Normalización Z-score interna. Autograd aplicará la regla de la cadena automáticamente.
            x = (x - self.input_mean) / (self.input_std + 1e-8)

        if self.fourier is not None:
            x = self.fourier(x)

        return self.network(x)

if __name__ == "__main__":
    # Prueba rápida de forward pass
    model = CoastalPINNModel()
    dummy_input = torch.randn(10, 4)  # 10 puntos de prueba
    output = model(dummy_input)
    print("Dummy input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Muestra de predicción:", output[:3].detach().numpy())
