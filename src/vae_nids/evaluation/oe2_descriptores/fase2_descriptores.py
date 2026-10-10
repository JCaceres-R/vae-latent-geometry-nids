"""Fase 2 -- descriptores geométricos por familia y semilla.

Referencia benigna R = val_benign sin duplicados de (mu, logvar), submuestreada a
REF_MAX_N. Control benigno C = submuestra de test_benign (el negativo del AUC).

Descriptores POR FLUJO (se evalúan en filas únicas y se expanden):
  maha    1.1  Mahalanobis al centroide de R (dims. activas)             mayor = más atípico
  knn     5.1  distancia al 10.º vecino de R, mu blanqueado            mayor = más atípico
  fr      2.4  Fisher-Rao al 10.º posterior de R (8 dims, usa sigma)    mayor = más atípico
  rate    7.1  KL(q(z|x) || N(0,I)) por flujo (tasa)                   sin dirección a priori
  dec_knn 6.2  10.º vecino de R en el espacio decodificado mu_x/sigma_x  mayor = más atípico
  logmag  6.1  1/2 log det(I + G(mu)), métrica pull-back de Fisher-Rao  sin dirección a priori
  lid     8.1  LID relativa (k=20 vecinos de R, mu blanqueado)          sin dirección a priori
  gmm     1.2  -log densidad de una mezcla de 8 gaussianas ajustada a R mayor = más atípico
  typ     1.4  tipicidad radial bilateral respecto de los radios de R   mayor = más atípico
  rmd     1.3  Mahalanobis relativa al prior D_M^2 - ||mu_act||^2       mayor = más atípico
  logvol  7.2  log-volumen del posterior sum_j log sigma_j^2            sin dirección a priori
  kp      11.1 |log sigma^2 observado - log 1/(1+G_jj)| medio (Kumar-Poole) sin dirección
Para cada uno se guardan tres resúmenes por familia: mediana, fracción por
encima del p95 del control y AUC del descriptor (familia frente a control).

Descriptores DE CONJUNTO (submuestras sin duplicados, SET_REPEATS repeticiones):
  mmd_agg 3.2  MMD^2 analítico entre posteriores agregados (familia vs R)
  airm    4.2  distancia afín-invariante entre covarianzas (Ledoit-Wolf)
  sil_w   10.1 silhouette del grupo de ataque, euclidiana sobre mu blanqueado
  sil_fr  10.1 silhouette del grupo de ataque con Fisher-Rao
  probe   10.3 AUC de una sonda lineal sobre (mu, log sigma^2) activos (familia vs C)
  erank   4.1  rango efectivo de la covarianza de la familia (exploratorio)

Uso:
    python -m vae_nids.evaluation.oe2_descriptores.fase2_descriptores [--seed 42]
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from vae_nids.evaluation.oe2_descriptores import config as C
from vae_nids.evaluation.oe2_descriptores import descriptores as D
from vae_nids.evaluation.oe2_descriptores.fase1_detectabilidad import active_dims_for
from vae_nids.evaluation.oe2_robust.phase2_encode import checkpoint_path_for
from vae_nids.evaluation.oe2_robust.phase3_geometry import load_group_mu_logvar

PER_FLOW = ["maha", "knn", "fr", "rate", "dec_knn", "logmag", "lid", "gmm", "typ", "rmd", "logvol", "kp"]
SET_LEVEL = ["mmd_agg", "airm", "sil_w", "sil_fr", "probe", "erank"]


def load_decoder(seed: int) -> D.ReluDecoder:
    sd = torch.load(checkpoint_path_for(seed), map_location="cpu", weights_only=False)["model_state_dict"]
    g = lambda k: sd[k].numpy().astype(float)  # noqa: E731
    return D.ReluDecoder(g("decoder.hidden.weight"), g("decoder.hidden.bias"),
                         g("decoder.mu_head.weight"), g("decoder.mu_head.bias"),
                         g("decoder.logvar_head.weight"), g("decoder.logvar_head.bias"),
                         C.LOGVAR_MIN, C.LOGVAR_MAX)


def unique_rows(mu, lv):
    key = np.hstack([mu, lv])
    _, first, inv, cnt = np.unique(key, axis=0, return_index=True, return_inverse=True, return_counts=True)
    return first, inv.ravel(), cnt


class Reference:
    """Todo lo que depende solo de la referencia benigna de una semilla."""

    def __init__(self, seed: int, act: list[int], dec: D.ReluDecoder, rng: np.random.Generator):
        z = np.load(C.CACHE_DIR / f"val_benign_seed{seed}.npz")
        mu, lv = z["mu"].astype(float), np.clip(z["logvar"].astype(float), C.LOGVAR_MIN, C.LOGVAR_MAX)
        first, _, _ = unique_rows(mu, lv)
        if len(first) > C.REF_MAX_N:
            first = rng.choice(first, C.REF_MAX_N, replace=False)
        self.mu, self.lv = mu[first], lv[first]
        self.n_val, self.n_unique_val = len(mu), len(np.unique(np.hstack([mu, lv]), axis=0))
        self.act, self.dec = act, dec
        a = self.mu[:, act]
        self.mean, self.cov = a.mean(0), np.cov(a, rowvar=False)
        self.w = D.whiten(a, self.mean, self.cov)
        self.tree_w = cKDTree(self.w)
        self.s = np.exp(0.5 * self.lv)
        self.tree_ms = cKDTree(np.hstack([self.mu, self.s]))
        self.phi = dec.embed(self.mu)
        self.nn_phi = NearestNeighbors(n_neighbors=C.K_KNN, algorithm="brute", n_jobs=-1).fit(self.phi)
        self.r_ref = D.mahalanobis(a, self.mean, self.cov)
        self.gmm = GaussianMixture(C.GMM_K, covariance_type="full", random_state=C.RNG_SEED).fit(self.w)
        hb = np.median(np.sqrt(D._sqdist(self.mu[:2000], self.mu[:2000])))
        self.bw = float(hb)

    # ---------------- descriptores por flujo (sobre filas únicas) ----------------
    def per_flow(self, mu, lv) -> dict[str, np.ndarray]:
        lv = np.clip(lv, C.LOGVAR_MIN, C.LOGVAR_MAX)
        a = mu[:, self.act]
        w = D.whiten(a, self.mean, self.cov)
        s = np.exp(0.5 * lv)
        out = {}
        out["maha"] = np.sqrt((w ** 2).sum(1))
        d, _ = self.tree_w.query(w, k=C.K_LID, workers=-1)
        out["knn"] = d[:, C.K_KNN - 1]
        dk = d[:, -1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            lid = -1.0 / np.mean(np.log(np.maximum(d, 1e-300) / dk), axis=1)
        lid[(dk[:, 0] <= 0) | ~np.isfinite(lid)] = np.nan
        out["lid"] = lid
        fr = []
        for i in range(0, len(mu), 20000):
            _, idx = self.tree_ms.query(np.hstack([mu[i:i + 20000], s[i:i + 20000]]), k=C.FR_CANDIDATES, workers=-1)
            f = D.fisher_rao_diag(mu[i:i + 20000, None, :], s[i:i + 20000, None, :], self.mu[idx], self.s[idx])
            fr.append(np.sort(f, axis=1)[:, C.K_KNN - 1])
        out["fr"] = np.concatenate(fr)
        out["rate"] = D.kl_to_prior_per_dim(mu, lv).sum(1)
        dphi, _ = self.nn_phi.kneighbors(self.dec.embed(mu))
        out["dec_knn"] = dphi[:, -1]
        out["logmag"] = self.dec.log_magnification(mu)
        out["gmm"] = -self.gmm.score_samples(w)
        out["typ"] = D.typicality_two_sided(out["maha"], self.r_ref)
        out["rmd"] = out["maha"] ** 2 - (a ** 2).sum(1)
        out["logvol"] = lv.sum(1)
        gdiag = np.concatenate([np.diagonal(self.dec.metric(mu[i:i + 20000]), axis1=1, axis2=2)
                                for i in range(0, len(mu), 20000)])
        pred = -np.log1p(gdiag[:, self.act] / C.BETA)
        out["kp"] = np.abs(lv[:, self.act] - pred).mean(1)
        return out

    # ---------------- descriptores de conjunto ----------------
    def set_level(self, mu, lv, mu_ctrl, lv_ctrl, rng) -> dict[str, float]:
        lv = np.clip(lv, C.LOGVAR_MIN, C.LOGVAR_MAX)
        first, _, _ = unique_rows(mu, lv)
        mu, lv = mu[first], lv[first]
        a = mu[:, self.act]
        res = {k: [] for k in ("mmd_agg", "sil_w", "sil_fr")}
        for r in range(C.SET_REPEATS):
            ia = rng.choice(len(mu), min(C.SET_MAX_N, len(mu)), replace=False)
            ib = rng.choice(len(self.mu), C.SET_MAX_N, replace=False)
            res["mmd_agg"].append(D.mmd2_aggregated_posteriors(mu[ia], lv[ia], self.mu[ib], self.lv[ib], self.bw))
            ja = ia[:C.SIL_MAX_N]
            jb = ib[:C.SIL_MAX_N]
            wa = D.whiten(a[ja], self.mean, self.cov)
            wb = self.w[jb]
            s_a, _, _, _ = D.silhouette_per_cluster(D.pairwise_euclid(wa, wa), D.pairwise_euclid(wb, wb),
                                                     D.pairwise_euclid(wa, wb))
            res["sil_w"].append(s_a)
            sa, sb = np.exp(0.5 * lv[ja]), self.s[jb]
            f_aa = D.pairwise_fr(mu[ja], sa, mu[ja], sa)
            f_bb = D.pairwise_fr(self.mu[jb], sb, self.mu[jb], sb)
            f_ab = D.pairwise_fr(mu[ja], sa, self.mu[jb], sb)
            s_a, _, _, _ = D.silhouette_per_cluster(f_aa, f_bb, f_ab)
            res["sil_fr"].append(s_a)
        out = {k: float(np.mean(v)) for k, v in res.items()}
        if len(a) > len(self.act) + 1:
            cov_f = D.shrink_cov(a)
            out["airm"] = D.airm_distance(D.shrink_cov(self.mu[:, self.act]), cov_f)
            out["erank"] = D.effective_rank(a)
        else:
            out["airm"] = out["erank"] = np.nan
        lc = np.clip(lv_ctrl, C.LOGVAR_MIN, C.LOGVAR_MAX)
        fc, _, _ = unique_rows(mu_ctrl, lc)
        xa = np.hstack([mu[:, self.act], lv[:, self.act]])
        xb = np.hstack([mu_ctrl[fc][:, self.act], lc[fc][:, self.act]])
        out["probe"] = D.linear_probe_auc(xa, xb, seed=C.RNG_SEED, max_n=C.PROBE_MAX_N) if len(xa) >= 10 else np.nan
        return out


def summaries(vals: np.ndarray, ctrl: np.ndarray) -> dict[str, float]:
    ok, okc = np.isfinite(vals), np.isfinite(ctrl)
    v, c = vals[ok], ctrl[okc]
    p95 = np.percentile(c, C.CONTROL_PERCENTILE)
    y = np.r_[np.ones(len(v)), np.zeros(len(c))]
    return dict(median=float(np.median(v)), frac_p95=float(np.mean(v > p95)),
                auc=float(roc_auc_score(y, np.r_[v, c])), frac_nan=float(1 - ok.mean()))


def descriptors_for_seed(seed: int) -> pd.DataFrame:
    out_csv = C.CACHE_DIR / f"descriptores_seed{seed}.csv"
    if out_csv.exists():
        print(f"[fase2 seed={seed}] ya calculado")
        return pd.read_csv(out_csv)
    t0 = time.time()
    rng = np.random.default_rng(C.RNG_SEED + seed)
    act = active_dims_for(seed)
    dec = load_decoder(seed)
    ref = Reference(seed, act, dec, rng)
    scores = np.load(C.CACHE_DIR / f"scores_seed{seed}.npz")

    mu_c, lv_c = load_group_mu_logvar(seed, "latent_benign_test")
    ic = rng.choice(len(mu_c), min(50_000, len(mu_c)), replace=False)
    mu_c, lv_c = mu_c[ic].astype(float), lv_c[ic].astype(float)
    nll_c = scores["BENIGN_nll"][ic]

    def flow_desc(mu, lv):
        first, inv, cnt = unique_rows(mu, lv)
        pf = ref.per_flow(mu[first], lv[first])
        return {k: v[inv] for k, v in pf.items()}, first, cnt

    ctrl, _, _ = flow_desc(mu_c, lv_c)
    rng_flow = np.random.default_rng(C.RNG_SEED + 10_000 + seed)   # aparte: no altera rng
    flow = {}
    rows = []
    groups = ["BENIGN_control"] + C.ALL_GROUPS
    for g in groups:
        if g == "BENIGN_control":
            mu, lv, nll, vals = mu_c, lv_c, nll_c, ctrl
            first, _, cnt = unique_rows(mu, lv)
        else:
            mu, lv = (x.astype(float) for x in load_group_mu_logvar(seed, g))
            nll = scores[f"{g}_nll"]
            vals, first, cnt = flow_desc(mu, lv)
        row = dict(seed=seed, group=g, n=len(mu), n_unique=len(first),
                   n_eff=float(cnt.sum() ** 2 / (cnt.astype(float) ** 2).sum()))
        for k in PER_FLOW:
            for sk, sv in summaries(vals[k], ctrl[k]).items():
                row[f"{k}__{sk}"] = sv
            ok = np.isfinite(vals[k])
            row[f"{k}__rho_flow_nll"] = float(spearmanr(vals[k][ok], nll[ok])[0]) if ok.sum() > 3 else np.nan
        for k, v in ref.set_level(mu, lv, mu_c, lv_c, rng).items():
            row[f"{k}__value"] = v
        rows.append(row)
        if g in C.MAIN_FAMILIES:
            idx = np.sort(rng_flow.choice(len(mu), min(len(mu), C.FLOW_SAMPLE_MAX), replace=False))
            flow[f"{g}__nll"] = np.asarray(nll)[idx].astype(np.float32)
            for k in C.FLOW_SAMPLE_DESCRIPTORS:
                flow[f"{g}__{k}"] = vals[k][idx].astype(np.float32)
        print(f"[fase2 seed={seed}] {g:32s} n={len(mu):7d} únicos={len(first):7d} "
              f"AUC_knn={row['knn__auc']:.3f} sil_w={row['sil_w__value']:.3f}  ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df["ref_n_val"], df["ref_n_unique_val"], df["ref_n_used"] = ref.n_val, ref.n_unique_val, len(ref.mu)
    np.savez_compressed(C.CACHE_DIR / f"flujo_seed{seed}.npz", **flow)
    df.to_csv(out_csv, index=False)
    return df


def main(seeds=None):
    torch.set_num_threads(8)
    dfs = [descriptors_for_seed(s) for s in (seeds or C.SEEDS)]
    if seeds is None:
        pd.concat(dfs, ignore_index=True).to_csv(C.TABLES_DIR / "descriptores_por_semilla.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    main([a.seed] if a.seed is not None else None)
