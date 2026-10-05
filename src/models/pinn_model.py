import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.climatology import ClimatologyPrior


class AnisotropicFourierFeatures(nn.Module):
    """
    Embedding posicional de Fourier aleatorio (Tancik et al., 2020), con una
    varianza de frecuencia distinta POR DIMENSIÓN de entrada ("anisotrópico").

    Motivación (2026-09-14): la MLP con Tanh sufre sesgo espectral hacia
    funciones de baja frecuencia (ver propuesta_integracion_datos_pinn.md,
    Sección 10). `scales` fija la desviación estándar de las frecuencias
    muestreadas por dimensión (más alta = más variación fina en esa dimensión).
    Las frecuencias (`B`) son un buffer, NO parámetro entrenable (RFF estándar).

    OJO (revisión 2026-09-25): con t normalizado por su desviación típica
    (~1560 días) el periodo anual queda en ~4.3 ciclos/unidad, es decir a ~4 sigma
    de una N(0,1): este embedding aleatorio casi nunca muestrea la frecuencia
    estacional. Para eso está `use_seasonal` (armónicos explícitos).
    """

    def __init__(self, in_dim=4, mapping_size=64, scales=(3.0, 3.0, 1.0, 1.0), seed=0):
        super().__init__()
        assert len(scales) == in_dim, "Debe darse una escala de frecuencia por dimensión de entrada"
        generator = torch.Generator().manual_seed(seed)
        B = torch.randn(in_dim, mapping_size, generator=generator)
        scale_vec = torch.tensor(scales, dtype=torch.float32).unsqueeze(1)
        B = B * scale_vec
        self.register_buffer('B', B)
        self.out_dim = in_dim + 2 * mapping_size

    def forward(self, x):
        proj = 2 * math.pi * (x @ self.B)
        return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)


class CoastalPINNModel(nn.Module):
    """
    Agente 3 (NN Architect): Red Neuronal Informada por la Física (PINN).

    Toma (lat, lon, profundidad, tiempo_días) y predice L = log1p(Chl-a).

    Opciones (todas desactivadas por defecto -> comportamiento histórico):
    - use_fourier_features: embedding de Fourier anisotrópico (ver arriba).
    - use_seasonal: añade sin/cos del tiempo con periodos anual y semianual
      (periodo en días, sobre el tiempo crudo; autograd sigue dando dL/dt exacto).
      Sin esto la red debía aprender el ciclo anual (surgencia primaveral) a
      partir de una rampa monótona de días.
    - climatology_prior: L = softplus_beta50(clima(z,lat,lon) + delta_NN). La
      última capa de la NN se inicializa a cero, así que el modelo PARTE de la
      climatología y solo se aleja donde los datos de train lo justifican.
    """

    def __init__(self, num_layers=6, hidden_dim=128, input_mean=None, input_std=None,
                 use_fourier_features=False, fourier_mapping_size=64,
                 fourier_scales=(3.0, 3.0, 1.0, 1.0),
                 use_seasonal=False, seasonal_periods=(365.25, 182.625),
                 climatology_prior=None):
        super(CoastalPINNModel, self).__init__()

        if input_mean is not None and input_std is not None:
            self.register_buffer('input_mean', torch.tensor(input_mean, dtype=torch.float32))
            self.register_buffer('input_std', torch.tensor(input_std, dtype=torch.float32))
            self.normalize = True
        else:
            self.normalize = False

        self.use_fourier_features = use_fourier_features
        if use_fourier_features:
            self.fourier = AnisotropicFourierFeatures(
                in_dim=4, mapping_size=fourier_mapping_size, scales=fourier_scales
            )
            first_in = self.fourier.out_dim
        else:
            self.fourier = None
            first_in = 4

        self.use_seasonal = use_seasonal
        if use_seasonal:
            self.register_buffer('seasonal_periods', torch.tensor(seasonal_periods, dtype=torch.float32))
            first_in += 2 * len(seasonal_periods)

        self.prior = climatology_prior  # None o ClimatologyPrior
        self.use_prior = climatology_prior is not None

        layers = [nn.Linear(first_in, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        last = nn.Linear(hidden_dim, 1)
        layers.append(last)
        if self.use_prior:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        else:
            layers.append(nn.Softplus())  # positividad (comportamiento histórico)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: [batch, 4] = (lat, lon, depth, time_days) CRUDOS. Devuelve [batch, 1] = log1p(Chl) >= 0.
        """
        raw = x
        if self.normalize:
            x = (x - self.input_mean) / (self.input_std + 1e-8)

        if self.fourier is not None:
            x = self.fourier(x)

        if self.use_seasonal:
            ang = 2 * math.pi * raw[:, 3:4] / self.seasonal_periods[None, :]
            x = torch.cat([x, torch.sin(ang), torch.cos(ang)], dim=-1)

        out = self.network(x)
        if self.use_prior:
            clim = self.prior(raw[:, 0:1], raw[:, 1:2], raw[:, 2:3])
            out = F.softplus(clim + out, beta=50.0)
        return out

    @classmethod
    def from_state_dict(cls, state_dict, num_layers=6, hidden_dim=128):
        """Reconstruye la arquitectura exacta con la que se entrenó un checkpoint
        (fourier / estacional / prior) a partir de sus propias claves, para que
        los scripts de evaluación no necesiten conocer la configuración."""
        kw = {}
        if 'input_mean' in state_dict and 'input_std' in state_dict:
            kw['input_mean'] = state_dict['input_mean'].cpu().numpy()
            kw['input_std'] = state_dict['input_std'].cpu().numpy()
        if 'fourier.B' in state_dict:
            kw['use_fourier_features'] = True
            kw['fourier_mapping_size'] = state_dict['fourier.B'].shape[1]
        if 'seasonal_periods' in state_dict:
            kw['use_seasonal'] = True
            kw['seasonal_periods'] = tuple(state_dict['seasonal_periods'].cpu().tolist())
        if 'prior.grid' in state_dict:
            kw['climatology_prior'] = ClimatologyPrior.empty_like_shape(tuple(state_dict['prior.grid'].shape[2:]))
        model = cls(num_layers=num_layers, hidden_dim=hidden_dim, **kw)
        model.load_state_dict(state_dict)
        return model


if __name__ == "__main__":
    model = CoastalPINNModel()
    dummy_input = torch.randn(10, 4)
    output = model(dummy_input)
    print("Dummy input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Muestra de predicción:", output[:3].detach().numpy())
