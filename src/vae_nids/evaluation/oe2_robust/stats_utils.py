"""Primitivas estadísticas de OE2 robusto -- cada función aquí tiene un test
correspondiente en tests/oe2_robust/test_stats_utils.py. No depende de
ningún dato del proyecto: son funciones puras sobre arrays numpy, para que
puedan probarse de forma aislada y reutilizarse en las Fases 3-5.

Referencias:
- DeLong et al. (1988) + Sun & Xu (2014) para el algoritmo rápido de
  varianza/covarianza de AUC-ROC.
- Cliff (1993) para el tamaño de efecto no paramétrico delta de Cliff.
- Gretton et al. (2012) para MMD con kernel RBF (estimador insesgado).
- Rabin et al. (2011) para la distancia de Wasserstein deslizada (SW).
- Székely & Rizzo (2013) para la distancia de energía.
- Kendall & Babington Smith (1939) para W de concordancia.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel


# ============================================================
# DeLong: varianza / comparación de AUC-ROC
# ============================================================

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks (empates -> rango promedio), O(n log n)."""
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _delong_components(pos_scores: np.ndarray, neg_scores: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """AUC + componentes estructurales (v01 sobre positivos, v10 sobre
    negativos) de Sun & Xu (2014), para un único score/clasificador."""
    m, n = len(pos_scores), len(neg_scores)
    combined = np.concatenate([pos_scores, neg_scores])
    tx = _compute_midrank(pos_scores)
    ty = _compute_midrank(neg_scores)
    tz = _compute_midrank(combined)
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    auc = tz[:m].sum() / (m * n) - (m + 1.0) / (2.0 * n)
    return float(auc), v01, v10


def delong_auc_variance(pos_scores: np.ndarray, neg_scores: np.ndarray) -> tuple[float, float]:
    """AUC-ROC y su varianza analítica (DeLong), sin bootstrap."""
    auc, v01, v10 = _delong_components(np.asarray(pos_scores, dtype=float), np.asarray(neg_scores, dtype=float))
    m, n = len(pos_scores), len(neg_scores)
    s01 = np.var(v01, ddof=1) if m > 1 else 0.0
    s10 = np.var(v10, ddof=1) if n > 1 else 0.0
    var = s01 / m + s10 / n
    return auc, float(var)


def delong_test_paired(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict:
    """Comparación PAREADA de dos scores sobre los MISMOS casos (misma
    familia, dos criterios de anomalía distintos, ej. nll_mc vs.
    latent_maha). y_true en {0,1}."""
    y_true = np.asarray(y_true)
    pos = y_true == 1
    neg = y_true == 0
    auc_a, v01_a, v10_a = _delong_components(np.asarray(score_a)[pos], np.asarray(score_a)[neg])
    auc_b, v01_b, v10_b = _delong_components(np.asarray(score_b)[pos], np.asarray(score_b)[neg])
    m, n = int(pos.sum()), int(neg.sum())
    s01 = np.cov(v01_a, v01_b, ddof=1) if m > 1 else np.zeros((2, 2))
    s10 = np.cov(v10_a, v10_b, ddof=1) if n > 1 else np.zeros((2, 2))
    cov = s01 / m + s10 / n
    diff = auc_a - auc_b
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z, p = _z_and_p(diff, var_diff)
    return {"auc_a": auc_a, "auc_b": auc_b, "diff": diff, "var_diff": float(var_diff),
            "z": z, "p_value": p, "method": "delong_paired"}


def delong_test_shared_negatives(pos_scores_a: np.ndarray, pos_scores_b: np.ndarray,
                                  neg_scores: np.ndarray) -> dict:
    """Comparación de AUC entre dos grupos de POSITIVOS independientes (dos
    familias de ataque distintas) evaluados contra el MISMO conjunto de
    negativos (benigno de test). Generalización del DeLong pareado: la
    covarianza entre las dos AUC solo puede provenir del lado compartido
    (negativos, indexados igual en ambas corridas); el lado de positivos es
    independiente por construcción (dos familias distintas), así que no
    aporta covarianza. Método documentado explícitamente porque no es el
    DeLong pareado clásico (que exige exactamente los mismos casos en ambos
    brazos) -- aquí solo los negativos son compartidos.
    """
    pos_scores_a = np.asarray(pos_scores_a, dtype=float)
    pos_scores_b = np.asarray(pos_scores_b, dtype=float)
    neg_scores = np.asarray(neg_scores, dtype=float)
    auc_a, v01_a, v10_a = _delong_components(pos_scores_a, neg_scores)
    auc_b, v01_b, v10_b = _delong_components(pos_scores_b, neg_scores)
    m_a, m_b, n = len(pos_scores_a), len(pos_scores_b), len(neg_scores)
    var_a = (np.var(v01_a, ddof=1) if m_a > 1 else 0.0) / m_a + (np.var(v10_a, ddof=1) if n > 1 else 0.0) / n
    var_b = (np.var(v01_b, ddof=1) if m_b > 1 else 0.0) / m_b + (np.var(v10_b, ddof=1) if n > 1 else 0.0) / n
    cov_neg = np.cov(v10_a, v10_b, ddof=1)[0, 1] / n if n > 1 else 0.0
    diff = auc_a - auc_b
    var_diff = var_a + var_b - 2 * cov_neg
    z, p = _z_and_p(diff, var_diff)
    return {"auc_a": auc_a, "auc_b": auc_b, "diff": diff, "var_diff": float(var_diff),
            "z": z, "p_value": p, "method": "delong_shared_negatives"}


def _z_and_p(diff: float, var_diff: float) -> tuple[float, float]:
    if var_diff <= 0 or not np.isfinite(var_diff):
        return float("nan"), float("nan")
    z = diff / np.sqrt(var_diff)
    p = float(2.0 * (1.0 - sp_stats.norm.cdf(abs(z))))
    return float(z), p


# ============================================================
# Tamaño de efecto: delta de Cliff
# ============================================================

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """delta = P(X>Y) - P(X<Y) en [-1,1]. Equivalente exacto a 2*AUC-1
    usando x como clase positiva -- se calcula así por eficiencia
    (O(n log n) vía ranking, en vez de la doble suma O(n*m))."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    labels = np.concatenate([np.ones(len(x)), np.zeros(len(y))])
    scores = np.concatenate([x, y])
    auc = roc_auc_score(labels, scores)
    return float(2.0 * auc - 1.0)


def cliffs_delta_bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 500,
                               seed: int = 42, alpha: float = 0.05) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    nx, ny = len(x), len(y)
    point = cliffs_delta(x, y)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        bx = x[rng.integers(0, nx, nx)]
        by = y[rng.integers(0, ny, ny)]
        boots[b] = cliffs_delta(bx, by)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


# ============================================================
# Divergencias entre nubes de puntos (familia vs. benigno)
# ============================================================

def _median_heuristic_gamma(x: np.ndarray, y: np.ndarray) -> float:
    """Ancho de banda RBF por heurística de la mediana: gamma =
    1/(2*sigma^2), sigma = mediana de las distancias euclidianas por pares
    del conjunto combinado (submuestreado si es grande, ya se submuestrea
    aguas arriba)."""
    pooled = np.vstack([x, y])
    d = euclidean_distances(pooled, pooled)
    iu = np.triu_indices_from(d, k=1)
    sigma = np.median(d[iu])
    if sigma <= 0:
        sigma = 1e-6
    return 1.0 / (2.0 * sigma ** 2)


def mmd2_rbf(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> tuple[float, float]:
    """MMD^2 insesgado con kernel RBF (Gretton et al. 2012, ecuación 3).
    Devuelve (mmd2, gamma_usado)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m, n = len(x), len(y)
    if gamma is None:
        gamma = _median_heuristic_gamma(x, y)
    kxx = rbf_kernel(x, x, gamma=gamma)
    kyy = rbf_kernel(y, y, gamma=gamma)
    kxy = rbf_kernel(x, y, gamma=gamma)
    sum_xx = (kxx.sum() - np.trace(kxx)) / (m * (m - 1)) if m > 1 else 0.0
    sum_yy = (kyy.sum() - np.trace(kyy)) / (n * (n - 1)) if n > 1 else 0.0
    sum_xy = kxy.mean()
    mmd2 = sum_xx + sum_yy - 2 * sum_xy
    return float(mmd2), float(gamma)


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, n_projections: int = 200, seed: int = 42) -> float:
    """Distancia de Wasserstein deslizada (Rabin et al. 2011): promedio de
    la distancia W1 1-D sobre proyecciones aleatorias uniformes en la
    esfera."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x.shape[1]
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_projections, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dists = np.empty(n_projections)
    for i, direction in enumerate(dirs):
        px = x @ direction
        py = y @ direction
        dists[i] = sp_stats.wasserstein_distance(px, py)
    return float(dists.mean())


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Distancia de energía (Székely & Rizzo 2013): 2*E|X-Y| - E|X-X'| -
    E|Y-Y'|, términos intra-grupo excluyendo la diagonal (U-estadístico)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m, n = len(x), len(y)
    dxy = euclidean_distances(x, y).mean()
    dxx_full = euclidean_distances(x, x)
    dyy_full = euclidean_distances(y, y)
    dxx = (dxx_full.sum()) / (m * (m - 1)) if m > 1 else 0.0
    dyy = (dyy_full.sum()) / (n * (n - 1)) if n > 1 else 0.0
    return float(2 * dxy - dxx - dyy)


# ============================================================
# Bhattacharyya (gaussianas multivariadas)
# ============================================================

def bhattacharyya_gaussian(mu1: np.ndarray, cov1: np.ndarray,
                            mu2: np.ndarray, cov2: np.ndarray) -> tuple[float, float, float]:
    """Distancia de Bhattacharyya entre N(mu1,cov1) y N(mu2,cov2).
    Devuelve (distancia_total, termino_medias, termino_covarianza)."""
    mu1 = np.asarray(mu1, dtype=float)
    mu2 = np.asarray(mu2, dtype=float)
    sigma = (cov1 + cov2) / 2.0
    diff = mu1 - mu2
    inv_sigma = np.linalg.inv(sigma)
    term_mean = float(0.125 * diff @ inv_sigma @ diff)
    _, logdet_s = np.linalg.slogdet(sigma)
    _, logdet1 = np.linalg.slogdet(cov1)
    _, logdet2 = np.linalg.slogdet(cov2)
    term_cov = float(0.5 * (logdet_s - 0.5 * (logdet1 + logdet2)))
    return term_mean + term_cov, term_mean, term_cov


# ============================================================
# Correlación: combinación entre semillas, permutación, potencia
# ============================================================

def fisher_combine(rhos: np.ndarray, alpha: float = 0.05) -> dict:
    """Combina K estimaciones independientes de rho (una por semilla) vía
    transformada z de Fisher: rho_combinado = tanh(media(z_i)), IC a partir
    del error estándar ENTRE semillas (sd(z_i)/sqrt(K)) -- no la fórmula de
    una sola muestra 1/sqrt(n-3), porque aquí las K réplicas ya estiman esa
    incertidumbre empíricamente."""
    rhos = np.asarray(rhos, dtype=float)
    rhos_clipped = np.clip(rhos, -0.999999, 0.999999)
    z = np.arctanh(rhos_clipped)
    k = len(z)
    mean_z = z.mean()
    if k > 1:
        se_z = z.std(ddof=1) / np.sqrt(k)
    else:
        se_z = float("nan")
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    lo_z, hi_z = mean_z - z_crit * se_z, mean_z + z_crit * se_z
    return {
        "rho_combined": float(np.tanh(mean_z)),
        "ci_low": float(np.tanh(lo_z)) if np.isfinite(lo_z) else float("nan"),
        "ci_high": float(np.tanh(hi_z)) if np.isfinite(hi_z) else float("nan"),
        "n_seeds": int(k),
        "rho_per_seed_mean": float(rhos.mean()),
        "rho_per_seed_sd": float(rhos.std(ddof=1)) if k > 1 else float("nan"),
    }


def permutation_test_spearman(x: np.ndarray, y: np.ndarray, n_perm: int = 10_000,
                               seed: int = 42) -> dict:
    """Test de permutación de dos colas para Spearman rho: baraja y,
    recalcula rho, p = fracción de |rho_perm| >= |rho_obs| (+1/+1 para
    evitar p=0)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho_obs, _ = sp_stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = len(y)
    count = 0
    null_rhos = np.empty(n_perm)
    for i in range(n_perm):
        y_perm = y[rng.permutation(n)]
        r, _ = sp_stats.spearmanr(x, y_perm)
        null_rhos[i] = r
        if abs(r) >= abs(rho_obs):
            count += 1
    p_perm = (count + 1) / (n_perm + 1)
    return {"rho_obs": float(rho_obs), "p_permutation": float(p_perm), "n_perm": int(n_perm)}


def spearman_power(n: int, rho: float, alpha: float = 0.05) -> float:
    """Potencia aproximada de un test de Spearman a dos colas vía
    aproximación de Fisher z (Cohen, 1988, cap. 3 -- misma aproximación
    usada para Pearson, estándar también para Spearman con n moderado/grande).
    """
    if n <= 3:
        return float("nan")
    se = 1.0 / np.sqrt(n - 3)
    z_rho = np.arctanh(rho)
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    power = (sp_stats.norm.cdf(z_rho / se - z_crit)
             + sp_stats.norm.cdf(-z_rho / se - z_crit))
    return float(power)


def spearman_min_detectable_rho(n: int, alpha: float = 0.05) -> float:
    """rho mínimo detectable (potencia=50%, es decir el rho cuyo z de
    Fisher iguala exactamente el umbral crítico) para tamaño de muestra n."""
    if n <= 3:
        return float("nan")
    se = 1.0 / np.sqrt(n - 3)
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z_crit * se))


# ============================================================
# Corrección por comparaciones múltiples
# ============================================================

def holm_correction(pvalues: np.ndarray) -> np.ndarray:
    from statsmodels.stats.multitest import multipletests
    _, p_adj, _, _ = multipletests(pvalues, method="holm")
    return p_adj


def bh_correction(pvalues: np.ndarray) -> np.ndarray:
    from statsmodels.stats.multitest import multipletests
    _, p_adj, _, _ = multipletests(pvalues, method="fdr_bh")
    return p_adj


# ============================================================
# Concordancia de rankings entre semillas
# ============================================================

def kendalls_w(rankings: np.ndarray) -> float:
    """W de Kendall (coeficiente de concordancia), Kendall & Babington
    Smith (1939). `rankings`: matriz (n_jueces, n_items) de VALORES
    (no rangos todavía -- se convierten aquí); un juez = una semilla, un
    item = una familia. W=1: concordancia perfecta; W=0: sin relación."""
    rankings = np.asarray(rankings, dtype=float)
    m, n = rankings.shape
    ranks = np.apply_along_axis(sp_stats.rankdata, 1, rankings)
    rank_sums = ranks.sum(axis=0)
    mean_rank_sum = rank_sums.mean()
    s = np.sum((rank_sums - mean_rank_sum) ** 2)
    denom = (m ** 2) * (n ** 3 - n)
    if denom == 0:
        return float("nan")
    w = 12.0 * s / denom
    return float(w)


# ============================================================
# Bootstrap genérico
# ============================================================

def bootstrap_ci_1sample(values: np.ndarray, stat_fn, n_boot: int = 500,
                          seed: int = 42, alpha: float = 0.05) -> tuple[float, float, float]:
    """IC bootstrap no paramétrico para un estadístico de una muestra
    (ej. mediana de Mahalanobis de una familia)."""
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    n = len(values)
    point = stat_fn(values)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = stat_fn(values[rng.integers(0, n, n)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)
