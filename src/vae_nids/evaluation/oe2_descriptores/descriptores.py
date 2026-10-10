"""Descriptores de la geometria del espacio latente probabilistico de un VAE.

Cada flujo es una gaussiana diagonal q(z|x) = N(mu, diag(sigma^2)). Las funciones reciben
arrays numpy (n, J). Solo se usan numpy, scipy y scikit-learn; si POT o ripser estan
disponibles se usan como alternativa de referencia (ver `BACKENDS`).

Referencias de las formulas (ver el informe del OE2, catalogo de descriptores):
  - Fisher-Rao diagonal: Miyamoto et al. (2023), Ec. (5.3).
  - W2 entre gaussianas: Delon y Desolneux (2020), Ec. (2.7); mezcla: Prop. 4, Ec. (4.4).
  - Bhattacharyya / Hellinger / verosimilitud esperada: Jebara et al. (2004), Ecs. (2), (3), (5).
  - MLS: Shi y Jain (2019), Ec. (3).
  - MMD insesgado: Gretton et al. (2012), Lema 6, Ec. (3). Energia: Sejdinovic et al. (2013), Ec. (2.5).
  - TwoNN: Facco et al. (2017), Ecs. (5)-(7). MLE: Levina y Bickel (2004), Ec. (8). LID: Ma et al. (2018), Ec. (4).
  - Rango efectivo: Roy y Vetterli (2007), Def. 1. Angulos principales: Bjorck y Golub (1973), Teo. 1.
  - Distancia afin-invariante entre covarianzas: Pennec et al. (2006), Sec. 3.
  - Silhouette macro: Pavlopoulos et al. (2024), Ecs. (3), (5), (6).
"""
from __future__ import annotations

import numpy as np
from scipy import linalg as sla
from scipy.spatial import cKDTree
from scipy.special import gammaln

BACKENDS = {}
try:  # opcional: Python Optimal Transport
    import ot  # noqa: F401
    BACKENDS["POT"] = ot.__version__
except Exception as e:  # noqa: BLE001
    BACKENDS["POT"] = f"no disponible ({type(e).__name__})"
try:  # opcional: ripser
    import ripser as _ripser  # noqa: F401
    BACKENDS["ripser"] = _ripser.__version__
except Exception as e:  # noqa: BLE001
    BACKENDS["ripser"] = f"no disponible ({type(e).__name__})"


# ---------------------------------------------------------------------------
# 1. Posicion puntual
# ---------------------------------------------------------------------------
def fit_gaussian(x):
    return x.mean(0), np.cov(x, rowvar=False)


def mahalanobis(x, mean, cov):
    """Distancia de Mahalanobis de cada fila de x a N(mean, cov)."""
    L = np.linalg.cholesky(cov)
    y = sla.solve_triangular(L, (x - mean).T, lower=True).T
    return np.sqrt((y ** 2).sum(1))


def whiten(x, mean, cov):
    """Coordenadas blanqueadas: en ellas la distancia euclidiana es la Mahalanobis."""
    L = np.linalg.cholesky(cov)
    return sla.solve_triangular(L, (x - mean).T, lower=True).T


def typicality_two_sided(r_query, r_ref):
    """Puntaje bilateral de atipicidad radial: 1 - p, con p el p-valor bilateral del radio
    del flujo respecto de la distribucion empirica de radios de referencia (benigno).
    0 = radio mediano del benigno; cerca de 1 = radio muy raro (demasiado grande O demasiado chico)."""
    ref = np.sort(r_ref)
    u = np.searchsorted(ref, r_query, side="right") / ref.size
    return np.abs(2.0 * u - 1.0)


def log_prior_density(mu):
    """log N(mu; 0, I): posicion respecto del prior."""
    return -0.5 * (mu ** 2).sum(1) - 0.5 * mu.shape[1] * np.log(2 * np.pi)


# ---------------------------------------------------------------------------
# 2 y 7. Divergencias entre posteriores gaussianos diagonales (usan sigma)
# ---------------------------------------------------------------------------
def kl_to_prior_per_dim(mu, logvar):
    """KL( N(mu, s^2) || N(0,1) ) por dimension (nats). Sumar en axis=1 da la tasa por flujo."""
    return 0.5 * (np.exp(logvar) + mu ** 2 - 1.0 - logvar)


def kl_diag(mu1, lv1, mu2, lv2):
    """KL( N1 || N2 ) fila a fila entre gaussianas diagonales."""
    return 0.5 * (np.exp(lv1 - lv2) + (mu1 - mu2) ** 2 / np.exp(lv2) - 1.0 + lv2 - lv1).sum(-1)


def jeffreys_diag(mu1, lv1, mu2, lv2):
    return kl_diag(mu1, lv1, mu2, lv2) + kl_diag(mu2, lv2, mu1, lv1)


def fisher_rao_diag(mu1, s1, mu2, s2):
    """Distancia de Fisher-Rao entre gaussianas diagonales (Miyamoto et al. 2023, Ec. 5.3).
    s = desviaciones estandar. Admite difusion (broadcast)."""
    num = (mu1 - mu2) ** 2 + 2.0 * (s1 - s2) ** 2
    den = (mu1 - mu2) ** 2 + 2.0 * (s1 + s2) ** 2
    a = np.arctanh(np.sqrt(np.clip(num / den, 0.0, 1.0 - 1e-15)))
    return 2.0 * np.sqrt(2.0) * np.sqrt((a ** 2).sum(-1))


def w2_diag(mu1, s1, mu2, s2):
    """W2 entre gaussianas diagonales: euclidiana en el espacio (mu, sigma) (Delon-Desolneux 2.7)."""
    return np.sqrt(((mu1 - mu2) ** 2).sum(-1) + ((s1 - s2) ** 2).sum(-1))


def bhattacharyya_diag(mu1, lv1, mu2, lv2):
    """Distancia de Bhattacharyya D_B = -log BC entre gaussianas diagonales."""
    v1, v2 = np.exp(lv1), np.exp(lv2)
    vm = 0.5 * (v1 + v2)
    return (0.125 * (mu1 - mu2) ** 2 / vm + 0.5 * (np.log(vm) - 0.5 * (lv1 + lv2))).sum(-1)


def hellinger_diag(mu1, lv1, mu2, lv2):
    """Hellinger en [0,1]: H = sqrt(1 - BC)."""
    return np.sqrt(np.clip(1.0 - np.exp(-bhattacharyya_diag(mu1, lv1, mu2, lv2)), 0.0, 1.0))


def mls_diag(mu1, lv1, mu2, lv2):
    """Mutual Likelihood Score (Shi y Jain 2019, Ec. 3) = log del nucleo de verosimilitud esperada."""
    v = np.exp(lv1) + np.exp(lv2)
    return (-0.5 * ((mu1 - mu2) ** 2 / v + np.log(v)) - 0.5 * np.log(2 * np.pi)).sum(-1)


def nearest_reference_posterior(mu_q, s_q, mu_ref, s_ref, k=10, kind="fr", n_candidates=64):
    """Para cada posterior de consulta, distancia a su k-esimo posterior de referencia mas cercano.
    W2 es euclidiana en (mu, sigma) => arbol KD exacto. Para Fisher-Rao se preseleccionan
    `n_candidates` vecinos W2 y se recalcula la distancia exacta sobre ellos (aproximacion)."""
    ref = np.hstack([mu_ref, s_ref])
    tree = cKDTree(ref)
    q = np.hstack([mu_q, s_q])
    if kind == "w2":
        d, _ = tree.query(q, k=k, workers=-1)
        return d[:, -1] if k > 1 else d
    J, kk, out = mu_q.shape[1], max(k, n_candidates), []
    for i in range(0, len(q), 20000):                                   # por bloques: memoria acotada
        _, idx = tree.query(q[i:i + 20000], k=kk, workers=-1)
        cand = ref[idx]                                                 # (b, kk, 2J)
        fr = fisher_rao_diag(mu_q[i:i + 20000, None, :], s_q[i:i + 20000, None, :], cand[:, :, :J], cand[:, :, J:])
        out.append(np.sort(fr, axis=1)[:, k - 1])
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# 3. Distancias entre distribuciones agregadas
# ---------------------------------------------------------------------------
def _sqdist(a, b):
    return np.maximum((a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2.0 * a @ b.T, 0.0)


def median_bandwidth(x, max_n=2000, seed=0):
    rng = np.random.default_rng(seed)
    s = x[rng.choice(len(x), min(max_n, len(x)), replace=False)]
    d2 = _sqdist(s, s)
    return float(np.sqrt(np.median(d2[np.triu_indices_from(d2, 1)])))


def mmd2_unbiased(x, y, bw, self_y=None):
    """MMD^2 insesgado con nucleo RBF (Gretton et al. 2012, Ec. 3). O(n*m): submuestrear.
    `self_y`: termino propio de y ya calculado (referencia fija)."""
    g = 1.0 / (2.0 * bw ** 2)
    kxx, kxy = np.exp(-g * _sqdist(x, x)), np.exp(-g * _sqdist(x, y))
    m, n = len(x), len(y)
    if self_y is None:
        kyy = np.exp(-g * _sqdist(y, y))
        self_y = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    return (kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + self_y - 2.0 * kxy.mean()


def mmd2_biased(x, y, bw):
    g = 1.0 / (2.0 * bw ** 2)
    return (np.exp(-g * _sqdist(x, x)).mean() + np.exp(-g * _sqdist(y, y)).mean()
            - 2.0 * np.exp(-g * _sqdist(x, y)).mean())


def _expected_rbf_block(mu_a, v_a, mu_b, v_b, bw):
    """E[k(z,z')] para z~N(mu_a, v_a), z'~N(mu_b, v_b), nucleo RBF de ancho bw (forma cerrada).
    Devuelve la matriz (na, nb)."""
    h2 = bw ** 2
    s = h2 + v_a[:, None, :] + v_b[None, :, :]                          # (na, nb, J)
    return np.exp(0.5 * (np.log(h2) - np.log(s)).sum(-1)
                  - 0.5 * ((mu_a[:, None, :] - mu_b[None, :, :]) ** 2 / s).sum(-1))


def mmd2_aggregated_posteriors(mu_a, lv_a, mu_b, lv_b, bw, chunk=512, self_b=None):
    """MMD^2 analitico entre dos posteriores AGREGADOS (mezclas uniformes de gaussianas diagonales).
    Usa mu y sigma sin muestrear z. Estimador insesgado en los indices de flujo (se excluye i=j).
    `self_b`: termino propio del grupo b ya calculado (ahorra tiempo si b es una referencia fija)."""
    va, vb = np.exp(lv_a), np.exp(lv_b)

    def mean_block(m1, v1, m2, v2, same):
        tot, cnt = 0.0, 0
        for i in range(0, len(m1), chunk):
            k = _expected_rbf_block(m1[i:i + chunk], v1[i:i + chunk], m2, v2, bw)
            if same:
                r = np.arange(i, min(i + chunk, len(m1)))
                k[r - i, r] = 0.0
                cnt += k.size - len(r)
            else:
                cnt += k.size
            tot += k.sum()
        return tot / cnt

    bb = mean_block(mu_b, vb, mu_b, vb, True) if self_b is None else self_b
    return mean_block(mu_a, va, mu_a, va, True) + bb - 2.0 * mean_block(mu_a, va, mu_b, vb, False)


def aggregated_self_term(mu, lv, bw, chunk=512):
    """Termino propio E[k(z,z')] de un posterior agregado (para reutilizarlo como `self_b`)."""
    v, tot, cnt = np.exp(lv), 0.0, 0
    for i in range(0, len(mu), chunk):
        k = _expected_rbf_block(mu[i:i + chunk], v[i:i + chunk], mu, v, bw)
        r = np.arange(i, min(i + chunk, len(mu)))
        k[r - i, r] = 0.0
        tot += k.sum(); cnt += k.size - len(r)
    return tot / cnt


def energy_distance(x, y, self_y=None):
    """Distancia de energia (Sejdinovic et al. 2013, Ec. 2.5), version U (insesgada)."""
    dxy = np.sqrt(_sqdist(x, y)).mean()
    dxx = np.sqrt(_sqdist(x, x))
    m, n = len(x), len(y)
    if self_y is None:
        self_y = np.sqrt(_sqdist(y, y)).sum() / (n * (n - 1))
    return 2.0 * dxy - dxx.sum() / (m * (m - 1)) - self_y


def sliced_wasserstein(x, y, n_proj=200, seed=0, p=2, n_quant=1000):
    """Wasserstein rebanado de orden p entre nubes de tamanos distintos (cuantiles 1-D)."""
    rng = np.random.default_rng(seed)
    th = rng.standard_normal((n_proj, x.shape[1])); th /= np.linalg.norm(th, axis=1, keepdims=True)
    q = (np.arange(n_quant) + 0.5) / n_quant
    qx = np.quantile(x @ th.T, q, axis=0); qy = np.quantile(y @ th.T, q, axis=0)
    return float((np.abs(qx - qy) ** p).mean() ** (1.0 / p))


def sliced_wasserstein_posteriors(mu_a, s_a, mu_b, s_b, **kw):
    """Aproximacion rebanada de MW2 (Delon-Desolneux, Prop. 4) entre posteriores agregados:
    Wasserstein rebanado entre las nubes de vectores (mu, sigma) en R^{2J}."""
    return sliced_wasserstein(np.hstack([mu_a, s_a]), np.hstack([mu_b, s_b]), **kw)


def wasserstein_exact(x, y, seed=0):
    """W2 exacto entre nubes pequenas (transporte discreto). Usa POT si se puede cargar; si no,
    asignacion optima de SciPy sobre submuestras del mismo tamano (el grupo grande se submuestrea)."""
    if not BACKENDS["POT"].startswith("no"):
        import ot
        return float(np.sqrt(ot.emd2(ot.unif(len(x)), ot.unif(len(y)), _sqdist(x, y))))
    from scipy.optimize import linear_sum_assignment
    rng = np.random.default_rng(seed)
    n = min(len(x), len(y))
    xs = x[rng.choice(len(x), n, replace=False)]; ys = y[rng.choice(len(y), n, replace=False)]
    c = _sqdist(xs, ys)
    r, cidx = linear_sum_assignment(c)
    return float(np.sqrt(c[r, cidx].mean()))


# ---------------------------------------------------------------------------
# 4. Forma y dispersion
# ---------------------------------------------------------------------------
def effective_rank(x):
    """Rango efectivo (Roy y Vetterli 2007) de la nube centrada, sobre autovalores de la covarianza."""
    ev = np.clip(np.linalg.eigvalsh(np.cov(x, rowvar=False)), 0, None)
    p = ev / ev.sum()
    return float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))


def shrink_cov(x):
    """Covarianza con contraccion de Ledoit-Wolf (necesaria con n pequeno)."""
    from sklearn.covariance import LedoitWolf
    return LedoitWolf().fit(x).covariance_


def airm_distance(c1, c2):
    """Distancia riemanniana afin-invariante entre covarianzas (Pennec et al. 2006):
    sqrt(sum log^2 lambda_i(c1^{-1} c2))."""
    lam = sla.eigvalsh(c2, c1)
    return float(np.sqrt((np.log(lam) ** 2).sum()))


def principal_angles_deg(x, y, r=2):
    """Angulos principales (grados) entre los subespacios de las r primeras CP de x y de y."""
    def top(a):
        w, v = np.linalg.eigh(np.cov(a, rowvar=False))
        return v[:, np.argsort(w)[::-1][:r]]
    return np.degrees(sla.subspace_angles(top(x), top(y)))


def logdet_cov(x, shrink=False):
    c = shrink_cov(x) if shrink else np.cov(x, rowvar=False)
    return float(np.linalg.slogdet(c)[1])


# ---------------------------------------------------------------------------
# 5 y 8. Densidad local, soporte y dimension intrinseca
# ---------------------------------------------------------------------------
def knn_distance(query, tree, k=10):
    """Distancia al k-esimo vecino de referencia (Sun et al. 2022)."""
    d, _ = tree.query(query, k=k, workers=-1)
    return d[:, -1] if k > 1 else d


def lid_relative(query, tree, k=20):
    """LID de cada punto estimada con sus k vecinos EN LA REFERENCIA (Ma et al. 2018, Ec. 4)."""
    d, _ = tree.query(query, k=k, workers=-1)
    d = np.maximum(d, 1e-12)
    return -1.0 / np.mean(np.log(d / d[:, -1:]), axis=1).clip(max=-1e-12)


def twonn(x):
    """Dimension intrinseca TwoNN (Facco et al. 2017), estimador MLE de la ley de Pareto de mu=r2/r1."""
    d, _ = cKDTree(x).query(x, k=3, workers=-1)
    ok = d[:, 1] > 0
    mu = d[ok, 2] / d[ok, 1]
    mu = mu[mu > 1.0]
    return float(len(mu) / np.log(mu).sum()) if len(mu) > 2 else float("nan")


def levina_bickel(x, k=10):
    """MLE de Levina y Bickel (2004), Ec. (8), con correccion k-2, promediado sobre puntos."""
    d, _ = cKDTree(x).query(x, k=k + 1, workers=-1)
    d = np.maximum(d[:, 1:], 1e-12)
    inv = np.log(d[:, -1:] / d[:, :-1]).sum(1) / (k - 2)
    return float(np.mean(1.0 / np.maximum(inv, 1e-12)))


def knn_log_density_ratio(query, tree_ref, n_ref, tree_own, n_own, k=10, dim=None):
    """Razon de log-densidades kNN: log p_propio(x) - log p_ref(x) (>0: mas denso en su grupo)."""
    dim = dim or query.shape[1]
    r_ref = np.maximum(knn_distance(query, tree_ref, k), 1e-12)
    r_own = np.maximum(knn_distance(query, tree_own, k + 1), 1e-12)      # +1: se excluye a si mismo
    return dim * (np.log(r_ref) - np.log(r_own)) + np.log(n_ref) - np.log(max(n_own - 1, 1))


# ---------------------------------------------------------------------------
# 10. Separabilidad
# ---------------------------------------------------------------------------
def silhouette_per_cluster(d_within_a, d_within_b, d_cross):
    """Silhouette por grupo a partir de matrices de distancia (cualquier disimilitud).
    d_within_a: (na,na), d_within_b: (nb,nb), d_cross: (na,nb). Devuelve (S_A, S_B, micro, macro)."""
    na, nb = d_cross.shape
    a_a = (d_within_a.sum(1)) / max(na - 1, 1); b_a = d_cross.mean(1)
    a_b = (d_within_b.sum(1)) / max(nb - 1, 1); b_b = d_cross.mean(0)
    s_a = (b_a - a_a) / np.maximum(np.maximum(a_a, b_a), 1e-300)
    s_b = (b_b - a_b) / np.maximum(np.maximum(a_b, b_b), 1e-300)
    micro = (s_a.sum() + s_b.sum()) / (na + nb)
    return float(s_a.mean()), float(s_b.mean()), float(micro), float(0.5 * (s_a.mean() + s_b.mean()))


def pairwise_euclid(a, b):
    return np.sqrt(_sqdist(a, b))


def pairwise_fr(mu_a, s_a, mu_b, s_b):
    return fisher_rao_diag(mu_a[:, None, :], s_a[:, None, :], mu_b[None, :, :], s_b[None, :, :])


def linear_probe_auc(x_a, x_b, seed=0, max_n=20000, folds=5):
    """AUC (validacion cruzada) de una regresion logistica balanceada que separa A de B."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    xa = x_a[rng.choice(len(x_a), min(max_n, len(x_a)), replace=False)]
    xb = x_b[rng.choice(len(x_b), min(max_n, len(x_b)), replace=False)]
    x = np.vstack([xa, xb]); y = np.r_[np.ones(len(xa)), np.zeros(len(xb))]
    f = int(min(folds, len(xa), len(xb)))
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500))
    p = cross_val_predict(clf, x, y, cv=StratifiedKFold(f, shuffle=True, random_state=seed),
                          method="decision_function")
    return float(roc_auc_score(y, p))


def fisher_ratio(x_a, x_b):
    """Razon de Fisher multivariada: (m_a-m_b)^T (S_a+S_b)^{-1} (m_a-m_b)."""
    d = x_a.mean(0) - x_b.mean(0)
    s = np.cov(x_a, rowvar=False) + np.cov(x_b, rowvar=False)
    return float(d @ np.linalg.solve(s, d))


# ---------------------------------------------------------------------------
# 9. Topologia: homologia persistente (H0 exacta; H1 por reduccion de matriz, nubes pequenas)
# ---------------------------------------------------------------------------
def maxmin_landmarks(x, m, seed=0):
    """Submuestreo maxmin (farthest point sampling): m puntos bien repartidos."""
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(x)))]
    d = np.linalg.norm(x - x[idx[0]], axis=1)
    for _ in range(m - 1):
        j = int(np.argmax(d)); idx.append(j)
        d = np.minimum(d, np.linalg.norm(x - x[j], axis=1))
    return x[idx]


def rips_persistence(x, maxdim=1):
    """Diagramas de persistencia de Vietoris-Rips en dimensiones 0 y 1.
    Usa ripser si se puede cargar; si no, una reduccion estandar de la matriz de borde en Z/2
    (valida, pero solo practica para ~<= 80 puntos)."""
    if not BACKENDS["ripser"].startswith("no"):
        import ripser
        dg = ripser.ripser(x, maxdim=maxdim)["dgms"]
        return dg[0][np.isfinite(dg[0][:, 1])], dg[1]
    n = len(x)
    dm = pairwise_euclid(x, x)
    iu = np.triu_indices(n, 1)
    order = np.argsort(dm[iu], kind="stable")
    edges = [(int(iu[0][o]), int(iu[1][o])) for o in order]
    ew = dm[iu][order]
    eidx = {e: i for i, e in enumerate(edges)}
    # H0: union-find; las aristas que unen componentes "matan" una clase de H0
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    h0, positive = [], np.zeros(len(edges), bool)
    for i, (u, v) in enumerate(edges):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv; h0.append((0.0, ew[i]))
        else:
            positive[i] = True                       # arista que crea un 1-ciclo
    if maxdim < 1:
        return np.array(h0), np.zeros((0, 2))
    # triangulos ordenados por su arista mas larga
    tri, tw = [], []
    for a in range(n):
        for b in range(a + 1, n):
            dab = dm[a, b]
            for c in range(b + 1, n):
                tri.append((a, b, c)); tw.append(max(dab, dm[a, c], dm[b, c]))
    tw = np.asarray(tw); torder = np.argsort(tw, kind="stable")
    low_to_col, h1 = {}, []
    for t in torder:
        a, b, c = tri[t]
        col = {eidx[(a, b)], eidx[(a, c)], eidx[(b, c)]}
        while col:
            low = max(col)
            if low in low_to_col:
                col ^= low_to_col[low]
            else:
                low_to_col[low] = col
                if tw[t] > ew[low]:
                    h1.append((ew[low], tw[t]))
                break
    return np.array(h0), (np.array(h1) if h1 else np.zeros((0, 2)))


def persistence_summary(dgm):
    """Resumenes de un diagrama: numero de barras, persistencia total y maxima, entropia persistente."""
    if len(dgm) == 0:
        return dict(n=0, total=0.0, maxp=0.0, entropy=0.0, mean_death=0.0)
    p = dgm[:, 1] - dgm[:, 0]
    q = p / p.sum()
    return dict(n=int(len(p)), total=float(p.sum()), maxp=float(p.max()),
                entropy=float(-(q * np.log(q + 1e-300)).sum()), mean_death=float(dgm[:, 1].mean()))


# ---------------------------------------------------------------------------
# 6 y 11. Geometria inducida por el decodificador (1 capa oculta ReLU, como el modelo del proyecto)
# ---------------------------------------------------------------------------
class ReluDecoder:
    """Decodificador gaussiano heterocedastico z -> (mu_x, logvar_x) con una capa oculta ReLU."""

    def __init__(self, w1, b1, wm, bm, wv, bv, lv_min=-10.0, lv_max=10.0):
        self.w1, self.b1, self.wm, self.bm, self.wv, self.bv = [np.asarray(a, float) for a in (w1, b1, wm, bm, wv, bv)]
        self.lv_min, self.lv_max = lv_min, lv_max

    def forward(self, z):
        pre = z @ self.w1.T + self.b1
        h = np.maximum(pre, 0.0)
        lv_raw = h @ self.wv.T + self.bv
        return h @ self.wm.T + self.bm, np.clip(lv_raw, self.lv_min, self.lv_max)

    def metric(self, z):
        """Metrica pull-back de Fisher-Rao G(z) = J_mu^T diag(e^{-l}) J_mu + 1/2 J_l^T J_l
        (Arvanitidis et al. 2022, Prop. 3.2 con la Fisher de la normal en (mu, log sigma^2))."""
        pre = z @ self.w1.T + self.b1
        act = (pre > 0).astype(float)
        h = pre * act
        lv_raw = h @ self.wv.T + self.bv
        lv = np.clip(lv_raw, self.lv_min, self.lv_max)
        free = ((lv_raw > self.lv_min) & (lv_raw < self.lv_max)).astype(float)
        jm = np.einsum("dh,nh,hk->ndk", self.wm, act, self.w1)
        jl = np.einsum("dh,nh,hk->ndk", self.wv, act, self.w1) * free[:, :, None]
        return np.einsum("ndk,nd,ndl->nkl", jm, np.exp(-lv), jm) + 0.5 * np.einsum("ndk,ndl->nkl", jl, jl)

    def log_magnification(self, z, chunk=20000):
        """1/2 log det(I + G(z)): log-volumen local (factor de magnificacion regularizado)."""
        out = []
        for i in range(0, len(z), chunk):
            g = self.metric(z[i:i + chunk])
            out.append(0.5 * np.linalg.slogdet(np.eye(g.shape[1]) + g)[1])
        return np.concatenate(out)

    def embed(self, z):
        """Embebido 'visto por el decodificador': media de salida estandarizada por su propia sigma_x.
        La distancia euclidiana aqui aproxima la raiz de 2*KL entre reconstrucciones."""
        m, lv = self.forward(z)
        return m * np.exp(-0.5 * lv)


def kumar_poole_predicted_var(decoder, z, beta=1.0):
    """Varianza del codificador predicha por Kumar y Poole (2020, Ec. 11) con familia diagonal:
    sigma_j^2 = 1 / (1 + G_jj / beta)."""
    out = []
    for i in range(0, len(z), 20000):
        g = decoder.metric(z[i:i + 20000])
        out.append(1.0 / (1.0 + np.diagonal(g, axis1=1, axis2=2) / beta))
    return np.concatenate(out)
