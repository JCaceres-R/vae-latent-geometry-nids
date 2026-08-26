"""
VAE de producción para OE1-OE4 — misma arquitectura matemática que
`vae_example.py` (la referencia educativa del proyecto, no se toca):
    x en R^input_dim -> Encoder Dense-hidden_dim -> (mu, log_sigma2) en R^k
    z = mu + sigma * eps, eps ~ N(0, I)
    z -> Decoder Dense-hidden_dim -> (mu_x_hat, log_sigma2_x_hat) en R^input_dim

Dos diferencias respecto al ejemplo educativo, ambas pedidas para poder
entrenar sobre el dataset real y barrer configuraciones en OE4:
  - input_dim no tiene default: se deriva de feature_columns.json
    (`load_feature_columns` / `VAEConfig.input_dim`), nunca hardcodeado.
  - beta y latent_dim no se fijan aquí: son argumentos de VAEConfig que
    training/train.py expone como parámetros de línea de comandos.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from vae_nids import config as cfg


def load_feature_columns(path: Path | None = None) -> list[str]:
    """Lee la lista ordenada de features que dejó `vae_nids.data.pipeline`
    (mismo orden con el que se ajustó scaler.joblib) -- fuente de verdad
    para input_dim, en vez de hardcodear 74 en el código del modelo.
    """
    path = path or (cfg.OUTPUT_DIR / "feature_columns.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class VAEConfig:
    input_dim: int          # se deriva de feature_columns.json, sin default
    latent_dim: int = 8      # k -- variable en OE4
    beta: float = 1.0        # peso del término KL en el ELBO -- variable en OE4
    hidden_dim: int = 32     # ancho de encoder/decoder (arquitectura fija)

    # Clamps de estabilidad numérica: sin esto, log_sigma2 puede irse a
    # valores extremos durante el entrenamiento (explota o colapsa a -inf).
    logvar_min: float = -10.0
    logvar_max: float = 10.0


class Encoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.act = nn.ReLU()
        self.mu_head = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.logvar_head = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.act(self.hidden(x))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        logvar = torch.clamp(logvar, self.cfg.logvar_min, self.cfg.logvar_max)
        return mu, logvar


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


class VAE(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps, eps ~ N(0, I) -- truco de reparametrización."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        mu_x_hat, logvar_x_hat = self.decoder(z)
        # Se exponen todos los tensores intermedios: mu/logvar los necesita
        # evaluation.metrics_oe1 (KL por dimensión, unidades activas).
        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "mu_x_hat": mu_x_hat,
            "logvar_x_hat": logvar_x_hat,
        }


def gaussian_nll(x: torch.Tensor, mu_hat: torch.Tensor, logvar_hat: torch.Tensor) -> torch.Tensor:
    """-log p_theta(x|z), Gaussiana heterocedástica, sumada sobre features,
    promediada sobre el batch. No es MSE: pondera cada feature por la
    varianza que el decoder predice para ella (An & Cho, 2015)."""
    var_hat = torch.exp(logvar_hat)
    log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=x.device))
    nll_per_dim = 0.5 * (log_2pi + logvar_hat + (x - mu_hat) ** 2 / var_hat)
    return nll_per_dim.sum(dim=1).mean()


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL( N(mu, sigma^2) || N(0, I) ), forma cerrada, sumada sobre las k
    dimensiones latentes y promediada sobre el batch (Kingma & Welling,
    2013, Apéndice B). Para el desglose por dimensión, ver
    evaluation.metrics_oe1.kl_per_dimension."""
    kl_per_dim = 0.5 * (logvar.exp() + mu.pow(2) - 1.0 - logvar)
    return kl_per_dim.sum(dim=1).mean()


def elbo_loss(
    x: torch.Tensor, outputs: dict[str, torch.Tensor], beta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(loss_total, recon_term, kl_term) -- se devuelven por separado para
    poder logear cada término por época y detectar colapso del posterior
    (KL -> 0)."""
    recon = gaussian_nll(x, outputs["mu_x_hat"], outputs["logvar_x_hat"])
    kl = kl_divergence(outputs["mu"], outputs["logvar"])
    total = recon + beta * kl
    return total, recon, kl
