"""OE3 -- calidad del tráfico sintético que genera el decodificador frente al benigno real.

Enunciado (propuesta, 22-07-2026): evaluar la calidad del tráfico sintético generado por el
decodificador mediante pruebas estadísticas de similitud con la distribución benigna real:
KS por variable, una distancia agregada (Wasserstein) y un clasificador real frente a sintético.

Diseño (fijado antes de ver resultados):
- Modelos: los del OE1 saneado (perfil saneado: 66 variables, semillas 42-49).
- Variantes de generación, N = 50 000 flujos cada una:
    prior_muestra       z ~ N(0, I), x ~ N(mu_x(z), sigma_x(z)^2)       <- principal (generación genuina)
    prior_media         z ~ N(0, I), x = mu_x(z)
    posterior_agregado  z ~ q(z | x_val), x ~ N(mu_x(z), sigma_x(z)^2)    <- aísla el desajuste prior/posterior
- Real de comparación: test_benign (no se usó para entrenar). Referencia real-real: val_benign frente a
  test_benign, que fija cuánto difieren dos muestras reales (el "piso" de cada indicador).
- Todo en la escala del modelo (MinMax ajustado en train_benign, [0, 1] en entrenamiento).
- Indicadores:
    KS (estadístico D) y Wasserstein-1 por variable; fracción de valores fuera de [0, 1].
    Wasserstein-2 rebanado multivariado (200 proyecciones, 5000 por lado).
    Clasificador de dos muestras (C2ST): HistGradientBoosting, AUC con validación cruzada de 5 pliegues,
      20 000 por lado. AUC 0,5 = indistinguible.
    Memorización (DCR): distancia al registro de train más cercano (referencia de 100 000), para 5000
      sintéticos y 5000 reales de test. Si los sintéticos estuvieran mucho más cerca que los reales, el
      decodificador estaría copiando entrenamiento.

Uso:
    VAE_NIDS_OE2_PERFIL=saneado python -m vae_nids.evaluation.oe3.calidad_generativa
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import ks_2samp, wasserstein_distance  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from vae_nids import config as base_cfg  # noqa: E402
from vae_nids.evaluation.oe2_robust import config as rcfg  # noqa: E402
from vae_nids.evaluation.oe2_robust.phase2_encode import load_model  # noqa: E402

ROOT = base_cfg.PROJECT_ROOT / "outputs" / "oe3"
TABLES, FIGURES = ROOT / "tables", ROOT / "figures"
N_GEN = 50_000
N_C2ST = 20_000
N_SW, N_PROJ = 5000, 200
N_DCR_REF, N_DCR_Q = 100_000, 5000
RNG_SEED = 2026
VARIANTES = ["prior_muestra", "prior_media", "posterior_agregado"]
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8985", "#0b0b0b", "#52514e"


def load_real(cols, rng):
    def take(part, n):
        df = pd.read_parquet(rcfg.data_path(part), columns=cols)
        x = df.to_numpy(np.float32)
        return x[rng.choice(len(x), min(n, len(x)), replace=False)]
    return take("test_benign", N_GEN), take("val_benign", N_GEN), take("train_benign", N_DCR_REF)


@torch.no_grad()
def generate(model, x_val, seed):
    g = torch.Generator().manual_seed(RNG_SEED + seed)
    k = model.cfg.latent_dim
    z = torch.randn(N_GEN, k, generator=g)
    mx, lx = model.decoder(z)
    out = {"prior_media": mx.numpy(),
           "prior_muestra": (mx + torch.exp(0.5 * lx) * torch.randn(mx.shape, generator=g)).numpy()}
    mu, lv = model.encoder(torch.from_numpy(x_val[:N_GEN]))
    z = mu + torch.exp(0.5 * lv) * torch.randn(mu.shape, generator=g)
    mx, lx = model.decoder(z)
    out["posterior_agregado"] = (mx + torch.exp(0.5 * lx) * torch.randn(mx.shape, generator=g)).numpy()
    return out


def per_variable(a, b, cols):
    return pd.DataFrame({"variable": cols,
                         "ks": [ks_2samp(a[:, j], b[:, j]).statistic for j in range(len(cols))],
                         "w1": [wasserstein_distance(a[:, j], b[:, j]) for j in range(len(cols))],
                         "frac_fuera_0_1": ((a < 0) | (a > 1)).mean(0)})


def sliced_w2(a, b, rng):
    a = a[rng.choice(len(a), min(N_SW, len(a)), replace=False)]
    b = b[rng.choice(len(b), min(N_SW, len(b)), replace=False)]
    th = rng.normal(size=(a.shape[1], N_PROJ))
    th /= np.linalg.norm(th, axis=0, keepdims=True)
    pa, pb = np.sort(a @ th, axis=0), np.sort(b @ th, axis=0)
    return float(np.sqrt(np.mean((pa - pb) ** 2)))


def c2st(a, b, rng):
    a = a[rng.choice(len(a), min(N_C2ST, len(a)), replace=False)]
    b = b[rng.choice(len(b), min(N_C2ST, len(b)), replace=False)]
    x = np.vstack([a, b])
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=RNG_SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=RNG_SEED)
    return float(cross_val_score(clf, x, y, cv=cv, scoring="roc_auc").mean())


def dcr(q, nn):
    d, _ = nn.kneighbors(q, n_neighbors=1)
    return d[:, 0]


def fig_ks(tab, seed, path):
    t = tab[(tab.seed == seed) & (tab.comparacion.isin(["prior_muestra", "real_val"]))]
    p = t.pivot(index="variable", columns="comparacion", values="ks").sort_values("prior_muestra")
    fig, ax = plt.subplots(figsize=(7, max(5, 0.14 * len(p))))
    y = np.arange(len(p))
    ax.barh(y, p.prior_muestra, color=BLUE, height=0.7, label="sintético (z ~ prior)")
    ax.plot(p.real_val, y, "|", color=ORANGE, ms=8, mew=2, label="real frente a real (val vs. test)")
    ax.set_yticks(y, p.index, fontsize=5.5)
    ax.set_xlabel("estadístico KS frente a test_benign", color=MUTED)
    ax.set_title(f"KS por variable, semilla {seed}", loc="left", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_hist(real, sint, cols, tab, seed, path):
    t = tab[(tab.seed == seed) & (tab.comparacion == "prior_muestra")].sort_values("ks")
    pick = list(t.variable.head(3)) + list(t.variable.tail(3))
    fig, axs = plt.subplots(2, 3, figsize=(10, 5.5))
    for ax, v in zip(axs.ravel(), pick):
        j = cols.index(v)
        lo, hi = np.percentile(np.r_[real[:, j], sint[:, j]], [0.5, 99.5])
        bins = np.linspace(lo, hi if hi > lo else lo + 1e-3, 60)
        ax.hist(real[:, j], bins=bins, density=True, alpha=0.55, color=GRAY, label="real (test)")
        ax.hist(sint[:, j], bins=bins, density=True, histtype="step", lw=1.5, color=BLUE, label="sintético")
        ax.set_title(f"{v}\nKS = {t.set_index('variable').ks[v]:.3f}".replace(".", ","), fontsize=7.5, loc="left")
        ax.tick_params(labelsize=6)
    axs[0, 0].legend(frameon=False, fontsize=6.5)
    fig.suptitle(f"Las 3 variables mejor y peor reproducidas (semilla {seed}, z ~ prior)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_c2st(res, path):
    p = res.groupby("comparacion").c2st_auc.agg(["mean", "min", "max"]).reindex(["real_val"] + VARIANTES)
    fig, ax = plt.subplots(figsize=(6.5, 3))
    x = np.arange(len(p))
    ax.bar(x, p["mean"], color=[ORANGE] + [BLUE] * len(VARIANTES), width=0.6)
    ax.errorbar(x, p["mean"], yerr=[p["mean"] - p["min"], p["max"] - p["mean"]], fmt="none", ecolor=INK, capsize=3, lw=1)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.set_xticks(x, ["real vs. real", "prior (muestra)", "prior (media)", "posterior agregado"], fontsize=8)
    ax.set_ylim(0.4, 1.02); ax.set_ylabel("AUC del clasificador", color=MUTED)
    ax.set_title("Clasificador real frente a sintético (media y rango entre semillas)", loc="left", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def md_table(df, d=3):
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(f"{v:.{d}f}" if isinstance(v, (float, np.floating)) else str(v).replace("|", r"\|")
                                     for v in r) + " |")
    return "\n".join(out)


def main():
    assert rcfg.PERFIL == "saneado", "el OE3 se corre sobre los modelos del OE1 saneado (VAE_NIDS_OE2_PERFIL=saneado)"
    TABLES.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    var_rows, res_rows = [], []
    for seed in rcfg.ALL_SEEDS:
        rng = np.random.default_rng(RNG_SEED + seed)
        model, cols = load_model(seed)
        x_test, x_val, x_train = load_real(cols, rng)
        gen = generate(model, x_val, seed)
        nn = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=-1).fit(x_train)
        d_real = dcr(x_test[:N_DCR_Q], nn)
        comps = {"real_val": x_val, **gen}
        for name, a in comps.items():
            pv = per_variable(a, x_test, cols)
            pv.insert(0, "comparacion", name); pv.insert(0, "seed", seed)
            var_rows.append(pv)
            row = dict(seed=seed, comparacion=name, ks_mediana=float(pv.ks.median()), ks_max=float(pv.ks.max()),
                       frac_variables_ks_menor_0_1=float((pv.ks < 0.1).mean()), w1_media=float(pv.w1.mean()),
                       frac_fuera_0_1=float(pv.frac_fuera_0_1.mean()), sw2=sliced_w2(a, x_test, rng),
                       c2st_auc=c2st(a, x_test, rng))
            if name != "prior_media":
                d = dcr(a[:N_DCR_Q], nn)
                row.update(dcr_mediana=float(np.median(d)), dcr_mediana_real_test=float(np.median(d_real)),
                           dcr_razon=float(np.median(d) / np.median(d_real)),
                           frac_copias_exactas=float((d < 1e-6).mean()))
            res_rows.append(row)
            print(f"[oe3 seed={seed}] {name:20s} KS med={row['ks_mediana']:.3f} C2ST={row['c2st_auc']:.3f} "
                  f"SW2={row['sw2']:.4f}", flush=True)
        if seed == rcfg.ALL_SEEDS[0]:
            var_tab = pd.concat(var_rows, ignore_index=True)
            fig_ks(var_tab, seed, FIGURES / f"ks_por_variable_seed{seed}.png")
            fig_hist(x_test, gen["prior_muestra"], cols, var_tab, seed, FIGURES / f"histogramas_seed{seed}.png")
    var_tab = pd.concat(var_rows, ignore_index=True)
    res = pd.DataFrame(res_rows)
    var_tab.to_csv(TABLES / "ks_w1_por_variable.csv", index=False)
    res.to_csv(TABLES / "resumen_por_semilla.csv", index=False)
    fig_c2st(res, FIGURES / "c2st_auc.png")

    agg = res.groupby("comparacion").mean(numeric_only=True).drop(columns="seed").reindex(["real_val"] + VARIANTES).reset_index()
    worst = (var_tab[var_tab.comparacion == "prior_muestra"].groupby("variable").ks.mean()
             .sort_values(ascending=False).head(10).reset_index().rename(columns={"ks": "ks_media"}))
    json.dump({"semillas": rcfg.ALL_SEEDS, "n_gen": N_GEN, "n_c2st": N_C2ST, "variantes": VARIANTES},
              open(TABLES / "meta.json", "w", encoding="utf-8"), indent=2)
    lines = ["# OE3 — calidad generativa del decodificador", "",
             f"Modelos del OE1 saneado ({len(rcfg.ALL_SEEDS)} semillas). Comparación frente a `test_benign` en la "
             "escala del modelo. `real_val` = val_benign frente a test_benign: lo que difieren dos muestras reales.", "",
             "## Resumen (media entre semillas)", "", md_table(agg, 4), "",
             "- `c2st_auc`: 0,5 = el clasificador no distingue; cuanto más cerca de 1, más fácil separar sintético de real.",
             "- `dcr_razon` < 1: los sintéticos quedan más cerca de train que los reales de test (señal de memorización); "
             "`frac_copias_exactas` > 0: copias literales.", "",
             "## Las 10 variables peor reproducidas (z ~ prior, KS medio entre semillas)", "", md_table(worst), "",
             f"Figuras: `outputs/oe3/figures/` (KS por variable e histogramas de la semilla {rcfg.ALL_SEEDS[0]}; C2ST).", ""]
    (ROOT / "RESULTADOS_OE3.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[oe3] resumen en {ROOT / 'RESULTADOS_OE3.md'}")


if __name__ == "__main__":
    main()
