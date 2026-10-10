"""Fase 4 -- diagnósticos del modelo real que condicionan la lectura de los descriptores.

1. Duplicados por grupo (semilla 42): n, mu únicos, (mu, logvar) únicos, n efectivo.
2. Por semilla: KL media por dimensión (benigno de control), rango efectivo de la
   métrica pull-back G(z) en z ~ N(0, I), fracción de log sigma_x^2 del decodificador en
   el piso, y la comprobación de Kumar y Poole (2020): sigma_j^2 observado frente a
   1 / (1 + G_jj) en dims. activas (benigno de control).
3. UMAP de la semilla 42 sin duplicados y con grupos balanceados (máx. 1500 por grupo).

Uso:
    python -m vae_nids.evaluation.oe2_descriptores.fase4_diagnosticos
"""
import numpy as np
import pandas as pd

from vae_nids.evaluation.oe2_descriptores import config as C
from vae_nids.evaluation.oe2_descriptores import descriptores as D
from vae_nids.evaluation.oe2_descriptores.fase1_detectabilidad import active_dims_for
from vae_nids.evaluation.oe2_descriptores.fase2_descriptores import load_decoder
from vae_nids.evaluation.oe2_robust.phase3_geometry import load_group_mu_logvar


def duplicates_table(seed=42):
    rows = []
    for g in ["latent_benign_test"] + C.ALL_GROUPS:
        mu, lv = load_group_mu_logvar(seed, g)
        _, cnt = np.unique(mu, axis=0, return_counts=True)
        _, cnt2 = np.unique(np.hstack([mu, lv]), axis=0, return_counts=True)
        rows.append(dict(group="BENIGN_test" if g == "latent_benign_test" else g, n=len(mu),
                         mu_unicos=len(cnt), mu_logvar_unicos=len(cnt2),
                         frac_unicos=len(cnt) / len(mu), max_repeticiones=int(cnt.max()),
                         n_efectivo=float(cnt.sum() ** 2 / (cnt.astype(float) ** 2).sum())))
    return pd.DataFrame(rows)


def per_seed_diagnostics(seed, rng):
    act = active_dims_for(seed)
    dec = load_decoder(seed)
    mu, lv = load_group_mu_logvar(seed, "latent_benign_test")
    i = rng.choice(len(mu), 20000, replace=False)
    mu, lv = mu[i].astype(float), np.clip(lv[i].astype(float), C.LOGVAR_MIN, C.LOGVAR_MAX)
    kl = D.kl_to_prior_per_dim(mu, lv).mean(0)
    z = rng.standard_normal((20000, mu.shape[1]))
    g = dec.metric(z)
    ev = np.linalg.eigvalsh(g)
    ev = np.clip(ev, 1e-300, None)
    p = ev / ev.sum(1, keepdims=True)
    erank = np.exp(-(p * np.log(p)).sum(1))
    _, lvx = dec.forward(mu)
    floor = float(np.mean(lvx <= C.LOGVAR_MIN + 1e-6))
    gd = np.diagonal(dec.metric(mu), axis1=1, axis2=2)
    pred = 1.0 / (1.0 + gd[:, act] / C.BETA)
    obs = np.exp(lv[:, act])
    lr = np.log(obs) - np.log(pred)
    row = dict(seed=seed, n_activas=len(act), dims_activas=" ".join(map(str, act)),
               kl_total=float(kl.sum()), erank_G_mediana=float(np.median(erank)),
               erank_G_p90=float(np.percentile(erank, 90)), frac_logvar_dec_en_piso=floor,
               sum_sigma2_obs=float(obs.sum(1).mean()), sum_sigma2_pred=float(pred.sum(1).mean()),
               kp_log_ratio_mediana=float(np.median(lr)), kp_spearman=float(
                   pd.Series(np.log(obs).ravel()).corr(pd.Series(np.log(pred).ravel()), method="spearman")))
    for j, v in enumerate(kl):
        row[f"kl_dim{j}"] = float(v)
    return row


def umap_balanced(seed=42, per_group=1500):
    import umap
    rng = np.random.default_rng(C.RNG_SEED)
    act = active_dims_for(seed)
    xs, labs = [], []
    for g in ["latent_benign_test"] + C.MAIN_FAMILIES:
        mu, _ = load_group_mu_logvar(seed, g)
        u = np.unique(mu, axis=0)
        u = u[rng.choice(len(u), min(per_group, len(u)), replace=False)]
        xs.append(u[:, act]); labs += ["BENIGN" if g == "latent_benign_test" else g] * len(u)
    x = np.vstack(xs)
    emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=C.RNG_SEED).fit_transform(x)
    return pd.DataFrame(dict(u1=emb[:, 0], u2=emb[:, 1], group=labs))


def main():
    rng = np.random.default_rng(C.RNG_SEED)
    duplicates_table().to_csv(C.TABLES_DIR / "duplicados_seed42.csv", index=False)
    pd.DataFrame([per_seed_diagnostics(s, rng) for s in C.SEEDS]).to_csv(
        C.TABLES_DIR / "diagnosticos_por_semilla.csv", index=False)
    umap_balanced().to_csv(C.TABLES_DIR / "umap_sin_duplicados_seed42.csv", index=False)
    print("[fase4] listo")


if __name__ == "__main__":
    main()
