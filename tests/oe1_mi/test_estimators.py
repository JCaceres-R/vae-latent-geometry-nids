"""Tests obligatorios del estimador de OE1-IM (fase 1). Todo sobre datos
sintéticos con semilla 42; no dependen del dataset. Los números exactos que
se reportan en el informe los produce `phase1_validation.py` con N mayor;
aquí se verifican las mismas propiedades con N suficiente para ser rápidos.

Tolerancias (justificadas con la calibración de la fase 1):
  - independencia: I_corr < 5e-4 bits (la fluctuación residual tras restar
    el sesgo es ~sqrt(2 dof) / (2 N ln 2) ~ 1e-4 bits con N=200k, B=20);
  - Gaussiana, KSG (n=20k, k=3): |r_info - rho| <= 0.03;
  - Gaussiana, histograma B=100 (N=500k): |r_info - rho| <= 0.01;
  - Gaussiana, histograma B=20: sesgo hacia abajo (r_info < rho) de a lo
    sumo 0.015 -- se testea explícitamente porque condiciona el criterio L.
"""
import numpy as np
import pytest

from vae_nids.evaluation.oe1_mi import cliques as cq
from vae_nids.evaluation.oe1_mi import estimators as est

SEED = 42


def _gauss(rho: float, n: int, seed: int = SEED):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    return z[:, 0], rho * z[:, 0] + np.sqrt(1 - rho ** 2) * z[:, 1]


# ---------------- Test 1: independencia ----------------
def test_independent_mi_corr_is_zero():
    rng = np.random.default_rng(SEED)
    n = 200_000
    x, y = rng.standard_normal(n), rng.exponential(size=n)
    r = est.mi_hist(est.discretize(x, 20), est.discretize(y, 20))
    assert r["mi_bits"] > 0            # el plug-in crudo está sesgado hacia arriba
    assert r["mi_corr_bits"] < 5e-4
    assert r["mi_bits"] == pytest.approx(r["bias_bits"], abs=5e-4)


def test_independent_discrete_mi_corr_is_zero():
    rng = np.random.default_rng(SEED)
    n = 200_000
    x = rng.integers(0, 5, n)
    y = rng.integers(0, 3, n)
    r = est.mi_hist(est.discretize(x, 20), est.discretize(y, 20))
    assert r["bins_x"] == 5 and r["bins_y"] == 3
    assert r["mi_corr_bits"] < 5e-4


# ---------------- Test 2: I(X;X) = H(X) ----------------
@pytest.mark.parametrize("kind", ["continua", "masa_en_cero", "categorica"])
def test_self_information_equals_entropy(kind):
    rng = np.random.default_rng(SEED)
    n = 100_000
    if kind == "continua":
        x = rng.lognormal(size=n)
    elif kind == "masa_en_cero":
        x = np.where(rng.random(n) < 0.6, 0.0, rng.exponential(size=n))
    else:
        x = rng.integers(0, 7, n).astype(float)
    d = est.discretize(x, 20)
    r = est.mi_hist(d, d)
    assert r["mi_bits"] == pytest.approx(r["H_x"], abs=1e-9)
    assert r["mi_corr_bits"] == pytest.approx(r["H_x"] - r["bias_bits"], abs=1e-9)
    assert r["nmi_sqrt"] == pytest.approx(1.0, abs=1e-3)


def test_quantile_bins_merge_on_heavy_mode():
    rng = np.random.default_rng(SEED)
    x = np.where(rng.random(100_000) < 0.6, 0.0, rng.exponential(size=100_000))
    d = est.discretize(x, 20)
    assert not d.categorical
    assert d.n_bins < 20       # los intervalos del 60 % en 0 se fusionan
    assert d.n_bins == len(np.unique(d.codes))


def test_atoms_scheme_keeps_minority_mass():
    # 96 % en cero: qcut literal lo colapsa a 1 intervalo (H = 0);
    # el esquema con átomos conserva cero / no-cero.
    rng = np.random.default_rng(SEED)
    n = 100_000
    x = np.where(rng.random(n) < 0.96, 0.0, rng.exponential(size=n))
    q = est.discretize(x, 20, scheme="qcut")
    a = est.discretize(x, 20, scheme="atomos")
    assert q.n_bins == 1 and est.entropy_bits(q) == 0.0
    assert a.n_bins == 2
    assert est.entropy_bits(a) == pytest.approx(0.2423, abs=0.01)  # H(0.96)


def test_atoms_scheme_equals_qcut_without_atoms():
    x = np.random.default_rng(SEED).lognormal(size=50_000)
    q = est.discretize(x, 20, scheme="qcut")
    a = est.discretize(x, 20, scheme="atomos")
    assert np.array_equal(q.codes, a.codes)


def test_atom_is_isolated_from_neighbours():
    # El átomo en 0 no se mezcla con valores positivos pequeños.
    rng = np.random.default_rng(SEED)
    x = np.where(rng.random(100_000) < 0.5, 0.0, rng.exponential(size=100_000))
    a = est.discretize(x, 20)
    assert len(np.unique(a.codes[x == 0])) == 1
    assert not np.isin(a.codes[x > 0], np.unique(a.codes[x == 0])).any()


# ---------------- Test 3: Gaussiana bivariada ----------------
@pytest.mark.parametrize("rho", [0.3, 0.7, 0.95])
def test_gaussian_rinfo_ksg(rho):
    x, y = _gauss(rho, 20_000)
    i_nats = est.mi_ksg_pairwise(x[:, None], y, seed=SEED, n_jobs=1)[0]
    assert float(est.linfoot(i_nats)) == pytest.approx(rho, abs=0.03)


@pytest.mark.parametrize("rho", [0.3, 0.7, 0.95])
def test_gaussian_rinfo_hist_high_b(rho):
    x, y = _gauss(rho, 500_000)
    r = est.mi_hist(est.discretize(x, 100), est.discretize(y, 100))
    assert r["r_info"] == pytest.approx(rho, abs=0.01)


@pytest.mark.parametrize("rho", [0.3, 0.7, 0.95])
def test_gaussian_rinfo_hist_b20_biased_down(rho):
    x, y = _gauss(rho, 500_000)
    r = est.mi_hist(est.discretize(x, 20), est.discretize(y, 20))
    assert rho - 0.015 <= r["r_info"] < rho


def test_linfoot_inverse_roundtrip():
    for r in (0.3, 0.7, 0.95):
        assert float(est.linfoot(est.linfoot_inverse_nats(r))) == pytest.approx(r, abs=1e-12)


# ---------------- Test 4: relación no lineal, no monótona ----------------
def test_nonlinear_nonmonotone_dependence():
    rng = np.random.default_rng(SEED)
    n = 200_000
    x = rng.uniform(-1, 1, n)
    y = x ** 2 + 0.05 * rng.standard_normal(n)
    pearson = abs(np.corrcoef(x, y)[0, 1])
    r = est.mi_hist(est.discretize(x, 20), est.discretize(y, 20))
    ksg = est.mi_ksg_pairwise(x[:20_000, None], y[:20_000], seed=SEED, n_jobs=1)[0]
    assert pearson < 0.01
    assert r["mi_corr_bits"] > 1.0
    assert ksg > 0.5


# ---------------- Resolución por clique ----------------
def _e(*pairs):
    return {frozenset(p) for p in pairs}


def test_clique_single():
    res = cq.resolve_groups(list("abcd"), _e("ab", "ac", "bc"))
    assert res["groups"] == [["a", "b", "c"]]


def test_clique_chain_is_not_merged():
    # a-b, b-c sin a-c: no es clique de 3; empaquetamiento no único y
    # intersección de 1 nodo -> ambigüedad, nada se fusiona.
    res = cq.resolve_groups(list("abc"), _e("ab", "bc"))
    assert res["groups"] == []
    assert len(res["ambiguous"]) == 1


def test_clique_path4_unique_packing():
    # Estructura del cluster 3: BwdMean-BwdMin-FwdMin-FwdMean.
    res = cq.resolve_groups(list("abcd"), _e("ab", "bc", "cd"))
    assert res["groups"] == [["a", "b"], ["c", "d"]]


def test_clique_two_overlapping_fuse_intersection():
    # Estructura del cluster 2: dos 5-cliques que comparten 4 nodos.
    core = "cdef"
    edges = {frozenset((u, v)) for i, u in enumerate(core) for v in core[i + 1:]}
    edges |= {frozenset(("a", u)) for u in core} | {frozenset(("b", u)) for u in core}
    res = cq.resolve_groups(list("abcdef"), edges)
    assert res["groups"] == [["c", "d", "e", "f"]]


def test_clique_two_cliques_sharing_one_node_fuse_private_parts():
    left, right = "abc", "efg"
    edges = set()
    for block in (left + "d", right + "d"):
        edges |= {frozenset((u, v)) for i, u in enumerate(block) for v in block[i + 1:]}
    res = cq.resolve_groups(list("abcdefg"), edges)
    assert sorted(res["groups"]) == [["a", "b", "c"], ["e", "f", "g"]]


# ---------------- Referencia gaussiana (criterio G) ----------------
def _gauss_ref_mi(dx, dy, rho, n, seed=SEED + 1):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    z1, z2 = z[:, 0], rho * z[:, 0] + np.sqrt(1 - rho ** 2) * z[:, 1]
    return est.mi_hist(est.gaussian_reference_codes(dx, z1),
                       est.gaussian_reference_codes(dy, z2))["mi_corr_bits"]


def test_atom_codes_are_value_ordered():
    rng = np.random.default_rng(SEED)
    x = np.where(rng.random(50_000) < 0.5, 3.0, rng.exponential(size=50_000) * 6)
    d = est.discretize(x, 20)
    means = [x[d.codes == k].mean() for k in range(d.n_bins)]
    assert means == sorted(means)


@pytest.mark.parametrize("rho,passes", [(0.97, True), (0.93, False)])
def test_gaussian_reference_separates_rho(rho, passes):
    n = 400_000
    x, y = _gauss(rho, n)
    dx, dy = est.discretize(x, 20), est.discretize(y, 20)
    obs = est.mi_hist(dx, dy)["mi_corr_bits"]
    ref = _gauss_ref_mi(dx, dy, 0.95, n)
    assert bool(obs >= ref) is passes


def test_gaussian_reference_has_no_entropy_ceiling():
    # Dos binarias idénticas: Linfoot no puede pasar 0.95 (techo 0.866) pero
    # G sí las marca como al menos tan dependientes como rho = 0.95.
    rng = np.random.default_rng(SEED)
    x = (rng.random(200_000) < 0.3).astype(float)
    d = est.discretize(x, 20)
    r = est.mi_hist(d, d)
    assert r["r_info"] < 0.95
    assert r["mi_corr_bits"] >= _gauss_ref_mi(d, d, 0.95, 200_000)
