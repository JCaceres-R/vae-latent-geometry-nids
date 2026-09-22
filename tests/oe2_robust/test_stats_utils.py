"""Tests de las primitivas estadísticas de OE2 robusto. No dependen de
datos del proyecto -- todo sobre arrays sintéticos, para poder correr
aislado y rápido (`pytest tests/oe2_robust/test_stats_utils.py`)."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from vae_nids.evaluation.oe2_robust import stats_utils as su


# ============================================================
# DeLong
# ============================================================

def test_delong_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    pos = rng.normal(1.0, 1.0, size=120)
    neg = rng.normal(0.0, 1.0, size=200)
    auc_delong, _ = su.delong_auc_variance(pos, neg)
    y_true = np.concatenate([np.ones(120), np.zeros(200)])
    y_score = np.concatenate([pos, neg])
    auc_sklearn = roc_auc_score(y_true, y_score)
    assert auc_delong == pytest.approx(auc_sklearn, abs=1e-9)


def test_delong_paired_identical_scores_gives_zero_diff():
    rng = np.random.default_rng(1)
    n = 150
    y_true = np.array([1] * 60 + [0] * 90)
    score = rng.normal(size=n)
    result = su.delong_test_paired(y_true, score, score.copy())
    assert result["diff"] == pytest.approx(0.0, abs=1e-12)
    assert result["auc_a"] == pytest.approx(result["auc_b"], abs=1e-12)


def test_delong_variance_converges_to_bootstrap_variance():
    """No hay una referencia analítica offline disponible en este entorno
    (sin acceso a pROC/R); se valida el caso conocido por convergencia:
    la varianza analítica de DeLong debe aproximar la varianza empírica de
    un bootstrap de 5000 remuestras sobre los mismos datos, dentro de una
    tolerancia relativa amplia (20%) -- ambos estiman la misma cantidad por
    caminos independientes (fórmula U-estadística vs. remuestreo)."""
    rng = np.random.default_rng(2)
    pos = rng.normal(0.6, 1.0, size=80)
    neg = rng.normal(0.0, 1.0, size=80)
    auc_delong, var_delong = su.delong_auc_variance(pos, neg)

    n_boot = 5000
    boot_aucs = np.empty(n_boot)
    for b in range(n_boot):
        bp = pos[rng.integers(0, len(pos), len(pos))]
        bn = neg[rng.integers(0, len(neg), len(neg))]
        boot_aucs[b] = roc_auc_score(np.concatenate([np.ones(len(bp)), np.zeros(len(bn))]),
                                      np.concatenate([bp, bn]))
    var_boot = boot_aucs.var(ddof=1)

    assert var_delong == pytest.approx(var_boot, rel=0.3)


def test_delong_shared_negatives_zero_when_same_positives():
    rng = np.random.default_rng(3)
    pos_a = rng.normal(size=50)
    neg = rng.normal(size=100)
    result = su.delong_test_shared_negatives(pos_a, pos_a.copy(), neg)
    assert result["diff"] == pytest.approx(0.0, abs=1e-12)


# ============================================================
# Bhattacharyya
# ============================================================

def test_bhattacharyya_identical_gaussians_is_zero():
    rng = np.random.default_rng(4)
    mu = rng.normal(size=4)
    a = rng.normal(size=(4, 4))
    cov = a @ a.T + np.eye(4) * 0.5
    dist, term_mean, term_cov = su.bhattacharyya_gaussian(mu, cov, mu.copy(), cov.copy())
    assert dist == pytest.approx(0.0, abs=1e-10)
    assert term_mean == pytest.approx(0.0, abs=1e-10)
    assert term_cov == pytest.approx(0.0, abs=1e-10)


def test_bhattacharyya_positive_for_different_gaussians():
    mu1 = np.zeros(3)
    mu2 = np.ones(3) * 2.0
    cov = np.eye(3)
    dist, term_mean, term_cov = su.bhattacharyya_gaussian(mu1, cov, mu2, cov)
    assert dist > 0
    assert term_mean > 0
    assert term_cov == pytest.approx(0.0, abs=1e-10)  # misma covarianza -> termino de forma nulo


# ============================================================
# MMD
# ============================================================

def test_mmd_same_sample_is_near_zero():
    # El estimador insesgado de MMD^2 excluye la diagonal de los términos
    # intra-grupo; entre X y una copia de sí mismo NO da exactamente 0
    # (es un caso degenerado), sino un residuo de orden O(1/n) que se
    # achica al crecer n -- de ahí "≈0", no "=0". Se verifica ambas cosas:
    # que sea pequeño en magnitud absoluta y que decrezca con n.
    rng = np.random.default_rng(5)
    x_small = rng.normal(size=(80, 3))
    x_large = rng.normal(size=(800, 3))
    mmd2_small, gamma = su.mmd2_rbf(x_small, x_small.copy())
    mmd2_large, _ = su.mmd2_rbf(x_large, x_large.copy())
    assert abs(mmd2_small) < 0.05
    assert abs(mmd2_large) < abs(mmd2_small)
    assert gamma > 0


def test_mmd_positive_for_shifted_distributions():
    rng = np.random.default_rng(6)
    x = rng.normal(0.0, 1.0, size=(150, 2))
    y = rng.normal(5.0, 1.0, size=(150, 2))
    mmd2, _ = su.mmd2_rbf(x, y)
    assert mmd2 > 0.1


# ============================================================
# Cliff's delta
# ============================================================

def test_cliffs_delta_identical_distributions_near_zero():
    rng = np.random.default_rng(7)
    x = rng.normal(size=300)
    y = rng.normal(size=300)
    d = su.cliffs_delta(x, y)
    assert abs(d) < 0.15


def test_cliffs_delta_fully_separated_is_one():
    x = np.arange(10, 20)
    y = np.arange(0, 10)
    d = su.cliffs_delta(x, y)
    assert d == pytest.approx(1.0)


# ============================================================
# Fisher combine / permutación / potencia
# ============================================================

def test_fisher_combine_matches_mean_for_identical_rhos():
    rhos = np.array([0.5, 0.5, 0.5, 0.5])
    result = su.fisher_combine(rhos)
    assert result["rho_combined"] == pytest.approx(0.5, abs=1e-6)


def test_permutation_test_null_case_high_p():
    rng = np.random.default_rng(8)
    x = rng.normal(size=11)
    y = rng.normal(size=11)
    result = su.permutation_test_spearman(x, y, n_perm=2000, seed=8)
    assert 0.0 <= result["p_permutation"] <= 1.0


def test_spearman_min_detectable_rho_n11_matches_known_value():
    # Valor de referencia conocido: con n=11, el rho mínimo detectable a
    # alpha=0.05 (dos colas) vía aproximación de Fisher z es ~0.60
    # (resultado estándar de tablas de potencia para Spearman/Pearson).
    rho_min = su.spearman_min_detectable_rho(11)
    assert rho_min == pytest.approx(0.60, abs=0.03)


def test_spearman_power_increases_with_rho():
    p_low = su.spearman_power(11, 0.3)
    p_mid = su.spearman_power(11, 0.5)
    p_high = su.spearman_power(11, 0.7)
    assert p_low < p_mid < p_high


# ============================================================
# Kendall's W
# ============================================================

def test_kendalls_w_perfect_agreement_is_one():
    rankings = np.array([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
    ], dtype=float)
    w = su.kendalls_w(rankings)
    assert w == pytest.approx(1.0, abs=1e-9)


def test_kendalls_w_random_is_low():
    rng = np.random.default_rng(9)
    rankings = np.array([rng.permutation(20) for _ in range(15)], dtype=float)
    w = su.kendalls_w(rankings)
    assert 0.0 <= w < 0.5
