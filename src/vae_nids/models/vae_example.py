"""
VAE explícito para tráfico de red (CICIDS2017 depurado) — ejemplo educativo.

Arquitectura (Sección 6.1-6.2 de la monografía):
    x in R^74 -> Encoder Dense-32 -> (mu, log_sigma2) en R^8 (latente)
    z = mu + sigma * eps, eps ~ N(0, I)          [reparametrización, Ec. 2]
    z -> Decoder Dense-32 -> (mu_x_hat, log_sigma2_x_hat) en R^74

El decoder saca una distribución completa (mu_x_hat, sigma2_x_hat), no solo un
punto reconstruido. Esto es intencional: es lo que permite calcular después el
score de "probabilidad de reconstrucción" de la Ec. 5 (An & Cho, 2015) en vez
de un MSE plano.

Corre este archivo directo (`python vae_example.py`) y vas a ver un forward
pass con datos aleatorios, para comprobar que las formas (shapes) cuadran.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Configuración explícita — nada de "números mágicos" sueltos en el código.
#    Todo lo que en OE4 vas a variar (beta, k=LATENT_DIM) vive aquí.
# ---------------------------------------------------------------------------
@dataclass
class VAEConfig:
    input_dim: int = 74        # features tras el EDA (antes R^78, ver eda_exclusion_log.json)
    hidden_dim: int = 32        # ancho de la capa oculta del encoder/decoder
    latent_dim: int = 8         # k — dimensión del espacio latente (variable en OE4)
    beta: float = 1.0           # peso del término KL en el ELBO (variable en OE4)

    # Clamps de estabilidad numérica: sin esto, log_sigma2 puede irse a valores
    # extremos durante el entrenamiento (explota o colapsa a -inf) y todo el
    # entrenamiento se vuelve NaN. Estos rangos son un punto de partida típico,
    # no un valor mágico — se pueden ajustar si ves saturación en los logs.
    logvar_min: float = -10.0
    logvar_max: float = 10.0


# ---------------------------------------------------------------------------
# 2. Encoder — implementa q_phi(z|x) = N(mu(x), sigma^2(x))   [Ec. 1]
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.act = nn.ReLU()
        # Dos "cabezas" separadas desde la misma capa oculta: una para mu,
        # otra para log(sigma^2). Se saca el LOG de la varianza (no la
        # varianza directa) porque la varianza siempre debe ser positiva, y
        # una red neuronal normal puede sacar cualquier número real — dejar
        # que la red saque log(sigma^2) y luego hacer exp() garantiza
        # positividad sin tener que restringir la salida de la capa.
        self.mu_head = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.logvar_head = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.act(self.hidden(x))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        # Clamp: evita que logvar explote durante entrenamiento (ver VAEConfig).
        logvar = torch.clamp(logvar, self.cfg.logvar_min, self.cfg.logvar_max)
        return mu, logvar


# ---------------------------------------------------------------------------
# 3. Decoder — implementa p_theta(x|z), sacando (mu_x_hat, sigma2_x_hat) [Ec. 3]
# ---------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        self.act = nn.ReLU()
        self.mu_head = nn.Linear(cfg.hidden_dim, cfg.input_dim)
        self.logvar_head = nn.Linear(cfg.hidden_dim, cfg.input_dim)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.act(self.hidden(z))
        mu_x_hat = self.mu_head(h)
        logvar_x_hat = self.logvar_head(h)
        logvar_x_hat = torch.clamp(logvar_x_hat, self.cfg.logvar_min, self.cfg.logvar_max)
        return mu_x_hat, logvar_x_hat


# ---------------------------------------------------------------------------
# 4. VAE completo — junta encoder + reparametrización + decoder
# ---------------------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps, eps ~ N(0, I)   [Ec. 2 — truco de reparametrización]

        Por qué existe esto: si muestreáramos z directamente de N(mu, sigma^2),
        esa operación de muestreo no es diferenciable, y el gradiente no podría
        propagarse hacia atrás para actualizar mu y sigma. El truco reescribe el
        muestreo como una transformación determinista (mu + sigma*eps) de una
        variable aleatoria "externa" (eps), que SÍ es diferenciable respecto a
        mu y sigma.
        """
        std = torch.exp(0.5 * logvar)  # sigma = exp(0.5 * log(sigma^2))
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        mu_x_hat, logvar_x_hat = self.decoder(z)
        # Se exponen TODOS los tensores intermedios (no solo la reconstrucción),
        # porque OE1 los necesita: mu/logvar para KL por dimensión y unidades
        # activas, z para inspección directa del espacio latente.
        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "mu_x_hat": mu_x_hat,
            "logvar_x_hat": logvar_x_hat,
        }


# ---------------------------------------------------------------------------
# 5. Pérdida — ELBO explícito   [Ec. 4]
# ---------------------------------------------------------------------------
def gaussian_nll(x: torch.Tensor, mu_hat: torch.Tensor, logvar_hat: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood Gaussiana: -log p_theta(x|z).

    Formula: 0.5 * [ log(2*pi) + logvar_hat + (x - mu_hat)^2 / exp(logvar_hat) ]
    sumada sobre las 74 dimensiones, promediada sobre el batch.

    Esto NO es MSE. MSE asume implícitamente varianza fija (=1) en todas las
    dimensiones, lo cual ignora que el decoder de este VAE aprende una
    varianza distinta por feature — justo lo que necesita la Ec. 5 después,
    para ponderar cada feature según su propia variabilidad predicha (la
    razón que da tu Sección 3, Justificación, para preferir probabilidad de
    reconstrucción sobre MSE).
    """
    var_hat = torch.exp(logvar_hat)
    log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=x.device))
    nll_per_dim = 0.5 * (log_2pi + logvar_hat + (x - mu_hat) ** 2 / var_hat)
    return nll_per_dim.sum(dim=1).mean()  # suma sobre features, promedio sobre batch


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL( N(mu, sigma^2) || N(0, I) ), forma cerrada (Kingma & Welling, 2013, Apéndice B).

    Formula: 0.5 * sum( sigma^2 + mu^2 - 1 - log(sigma^2) )  sobre las 8 dims latentes.

    No hace falta estimar esto por muestreo (a diferencia del término de
    reconstrucción, que si no fuera Gaussiano sí lo requeriría) porque la
    divergencia KL entre dos Gaussianas tiene una fórmula analítica exacta.
    """
    kl_per_dim = 0.5 * (logvar.exp() + mu.pow(2) - 1.0 - logvar)
    return kl_per_dim.sum(dim=1).mean()


def elbo_loss(
    x: torch.Tensor, outputs: dict, beta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Devuelve (loss_total, recon_term, kl_term) para poder loguear cada
    término por separado durante el entrenamiento — así se puede ver a simple
    vista si el KL se está yendo a 0 (señal de colapso del posterior).
    """
    recon = gaussian_nll(x, outputs["mu_x_hat"], outputs["logvar_x_hat"])
    kl = kl_divergence(outputs["mu"], outputs["logvar"])
    # Minimizamos NLL + beta*KL <=> maximizamos el ELBO (Ec. 4, que está en
    # forma de "maximizar log-verosimilitud menos KL").
    total = recon + beta * kl
    return total, recon, kl


# ---------------------------------------------------------------------------
# 6. Demo ejecutable — corre esto para comprobar que las formas cuadran
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = VAEConfig()  # usa los defaults: input=74, hidden=32, latent=8, beta=1.0
    model = VAE(cfg)

    batch_size = 16
    x_fake = torch.randn(batch_size, cfg.input_dim)  # datos aleatorios solo para probar shapes

    outputs = model(x_fake)
    loss, recon, kl = elbo_loss(x_fake, outputs, cfg.beta)

    print("--- Shapes ---")
    for k, v in outputs.items():
        print(f"  {k:14s} {tuple(v.shape)}")

    print("\n--- Loss (con datos aleatorios, no entrenados) ---")
    print(f"  total = {loss.item():.4f}  recon = {recon.item():.4f}  kl = {kl.item():.4f}")