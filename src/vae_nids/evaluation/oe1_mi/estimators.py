"""Estimadores de entropía e información mutua para OE1-IM.

Convenciones:
  - Entropías e IM por histograma en BITS (log2).
  - IM KSG en NATS (lo que devuelve sklearn); r_info siempre se calcula
    desde nats.
  - Discretización por igual frecuencia (cuantiles) con fusión de intervalos
    duplicados; si la feature tiene <= B valores únicos se usan sus valores
    como categorías.

Referencias (sección 0 del prompt de OE1-IM):
  - Corrección de sesgo del estimador plug-in (Sharmin et al., 2019):
    E[I_hat] - I ~= (I_x - 1)(I_y - 1) / (2 N ln 2) bits.
  - Bajo independencia, 2 N ln(2) I_hat ~ chi2((I_x-1)(I_y-1)) (estadístico G).
  - Coeficiente informacional de Linfoot: r_info = sqrt(1 - exp(-2 I_nats)).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

LN2 = np.log(2.0)


@dataclass
class Discretized:
    codes: np.ndarray        # int32, 0..n_bins-1
    n_bins: int              # intervalos efectivos (categorías realmente presentes)
    categorical: bool        # True si se usaron los valores tal cual (<= B únicos)


def _qcut_codes(x: np.ndarray, n_bins: int) -> np.ndarray:
    codes = np.asarray(pd.qcut(x, n_bins, labels=False, duplicates="drop"), dtype=np.int64)
    # qcut puede dejar etiquetas sin usar tras fusionar bordes: re-etiquetar
    # a 0..k-1 para que n_bins = intervalos efectivos presentes.
    return np.unique(codes, return_inverse=True)[1]


def discretize(x: np.ndarray, n_bins: int, scheme: str = "atomos") -> Discretized:
    """Igual frecuencia con B nominal. Si la feature tiene <= B valores
    únicos, se usan sus valores como categorías.

    scheme="qcut": `pd.qcut(duplicates='drop')` literal. Con una moda que
      ocupa varios cuantiles los bordes repetidos se fusionan; si la moda
      pasa de (B-1)/B de la masa, TODO queda en un solo intervalo (H = 0)
      aunque el resto de valores sea informativo (p. ej. Fwd Bytes/Bulk Avg,
      95.9 % de ceros), y en general el átomo queda mezclado con valores
      vecinos del mismo intervalo.
    scheme="atomos" (principal): equivalente con átomos aislados -- cada
      valor con frecuencia >= N/B (ocuparía al menos un intervalo entero)
      es su propia categoría, y los valores restantes se reparten por
      igual frecuencia en max(1, round(B * n_resto / N)) intervalos. Sin
      átomos coincide exactamente con "qcut".
    """
    x = np.asarray(x)
    uniques, counts = np.unique(x, return_counts=True)
    if len(uniques) <= n_bins:
        codes = np.searchsorted(uniques, x).astype(np.int32)
        return Discretized(codes, len(uniques), True)
    if scheme == "qcut":
        codes = _qcut_codes(x, n_bins)
        return Discretized(codes.astype(np.int32), int(codes.max()) + 1, False)
    if scheme != "atomos":
        raise ValueError(scheme)
    n = len(x)
    atoms = uniques[counts >= n / n_bins]
    if len(atoms) == 0:
        codes = _qcut_codes(x, n_bins)
        return Discretized(codes.astype(np.int32), int(codes.max()) + 1, False)
    is_atom = np.isin(x, atoms)
    codes = np.empty(n, dtype=np.int64)
    codes[is_atom] = np.searchsorted(atoms, x[is_atom])
    rest = x[~is_atom]
    if len(rest):
        b_rest = max(1, int(round(n_bins * len(rest) / n)))
        if len(np.unique(rest)) <= b_rest:
            rest_codes = np.unique(rest, return_inverse=True)[1]
        else:
            rest_codes = _qcut_codes(rest, b_rest)
        codes[~is_atom] = len(atoms) + rest_codes
    present, codes = np.unique(codes, return_inverse=True)
    # Reordenar los códigos por la media de sus valores: los átomos se
    # etiquetaron antes que el resto, y la referencia gaussiana
    # (gaussian_reference_codes) necesita códigos ordenados por valor. La IM y
    # la entropía no dependen de este reetiquetado.
    k = len(present)
    means = np.bincount(codes, weights=x.astype(np.float64), minlength=k) / np.bincount(codes, minlength=k)
    rank = np.empty(k, dtype=np.int64)
    rank[np.argsort(means, kind="stable")] = np.arange(k)
    return Discretized(rank[codes].astype(np.int32), k, False)


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = counts[counts > 0].astype(np.float64)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def entropy_bits(d: Discretized) -> float:
    return entropy_from_counts(np.bincount(d.codes, minlength=d.n_bins))


def joint_entropy_bits(dx: Discretized, dy: Discretized) -> float:
    joint = dx.codes.astype(np.int64) * dy.n_bins + dy.codes
    return entropy_from_counts(np.bincount(joint, minlength=dx.n_bins * dy.n_bins))


def linfoot(i_nats) -> np.ndarray:
    """r_info = sqrt(1 - exp(-2 I)), I en nats. Igual a |rho| para gaussianas."""
    i_nats = np.maximum(np.asarray(i_nats, dtype=np.float64), 0.0)
    return np.sqrt(1.0 - np.exp(-2.0 * i_nats))


def linfoot_inverse_nats(r: float) -> float:
    """I (nats) que corresponde a r_info = r."""
    return float(-0.5 * np.log(1.0 - r ** 2))


def mi_hist(dx: Discretized, dy: Discretized, hx: float | None = None,
            hy: float | None = None) -> dict:
    """IM plug-in por tabla de contingencia, con corrección de sesgo de
    Sharmin (2019), normalizaciones, Linfoot y el estadístico G (solo
    informativo)."""
    n = len(dx.codes)
    hx = entropy_bits(dx) if hx is None else hx
    hy = entropy_bits(dy) if hy is None else hy
    hxy = joint_entropy_bits(dx, dy)
    mi = hx + hy - hxy
    dof = (dx.n_bins - 1) * (dy.n_bins - 1)
    bias = dof / (2.0 * n * LN2)
    mi_corr = max(mi - bias, 0.0)

    def _safe(num, den):
        return float(num / den) if den > 0 else np.nan

    g_stat = 2.0 * n * LN2 * mi_corr
    return {
        "H_x": hx, "H_y": hy, "H_xy": hxy,
        "bins_x": dx.n_bins, "bins_y": dy.n_bins,
        "mi_bits": mi, "bias_bits": bias, "mi_corr_bits": mi_corr,
        "nmi_sqrt": _safe(mi_corr, np.sqrt(hx * hy)),
        "nmi_max": _safe(mi_corr, max(hx, hy)),
        "u_x_given_y": _safe(mi_corr, hx),
        "u_y_given_x": _safe(mi_corr, hy),
        "r_info": float(linfoot(mi_corr * LN2)),
        "g_stat": g_stat, "g_dof": dof,
        # Informativo; con N ~ 2e6 prácticamente siempre ~0 (ver informe).
        "g_pvalue": float(stats.chi2.sf(g_stat, dof)) if dof > 0 else np.nan,
    }


def gaussian_reference_codes(d: Discretized, z: np.ndarray) -> Discretized:
    """Discretiza una N(0,1) `z` con las MISMAS probabilidades marginales de
    intervalo que la feature discretizada `d` (umbrales Phi^-1 de la
    distribución acumulada de sus códigos, que están ordenados por valor).

    Con z1, z2 de una gaussiana bivariada con correlación rho, la IM entre
    gaussian_reference_codes(dx, z1) y gaussian_reference_codes(dy, z2) es
    la IM que tendría un par con cópula gaussiana rho y exactamente las
    mismas marginales discretizadas que (X, Y): la versión "justa" del umbral
    |r| >= rho en la escala de la IM (sin techo de entropía de Linfoot ni
    pérdida por discretizar)."""
    p = np.bincount(d.codes, minlength=d.n_bins) / len(d.codes)
    thr = stats.norm.ppf(np.cumsum(p)[:-1])
    codes = np.searchsorted(thr, z, side="right").astype(np.int32)
    return Discretized(codes, d.n_bins, d.categorical)


def mi_ksg_pairwise(x: np.ndarray, target: np.ndarray, seed: int,
                    n_neighbors: int = 3, n_jobs: int | None = -1) -> np.ndarray:
    """IM KSG (nats) entre cada columna de `x` y `target`, vía
    sklearn.feature_selection.mutual_info_regression (Kraskov et al. 2004,
    estimador 1). sklearn escala a varianza unitaria y agrega ruido
    ~1e-10 para romper empates; `seed` fija ese ruido."""
    return mutual_info_regression(
        x, target, discrete_features=False, n_neighbors=n_neighbors,
        copy=True, random_state=seed, n_jobs=n_jobs,
    )
