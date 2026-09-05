"""
OE2 Fase 6 -- proyección 2D del espacio latente (UMAP), apoyo visual.

NO reemplaza ni valida el resultado cuantitativo de la Fase 5 (rho=0.38,
no significativo) -- es solo para ver espacialmente dónde cae cada
familia. Usa las mu de 8 dimensiones de la Fase 2, recortadas a las 6
dims activas de la Fase 3 ([0, 1, 3, 4, 5, 6]) para que la proyección sea
consistente con la geometría ya reportada cuantitativamente.

Una sola proyección UMAP ajustada sobre benigno (submuestreado a 3,000,
seed=42) + las 11 familias principales concatenadas -- nunca una
proyección por familia por separado (quedarían en sistemas de coordenadas
distintos, no comparables).

Uso:
    python -m vae_nids.evaluation.umap_projection_oe2
"""
import time

_t0 = time.perf_counter()
print("[import] cargando matplotlib/numpy/umap (numba JIT compila en el primer uso, "
      "puede tardar 1-3 min)...", flush=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP

from vae_nids import config as cfg

print(f"[import] listo en {time.perf_counter() - _t0:.1f}s", flush=True)

LATENT_DIR = cfg.PROJECT_ROOT / "outputs" / "latent_vectors"
FIGURES_DIR = cfg.PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ACTIVE_DIMS = [0, 1, 3, 4, 5, 6]  # misma exclusión de dims 2 y 7 que la Fase 3
BENIGN_SUBSAMPLE_N = 3000
SEED = 42

MAIN_FAMILIES = [
    "PortScan", "DoS_Hulk", "DDoS", "DoS_GoldenEye", "DoS_slowloris",
    "FTP_Patator", "SSH_Patator", "DoS_Slowhttptest", "Bot", "Web_Attack",
    "Infiltration",
]


def main():
    rng = np.random.default_rng(SEED)

    mu_benign = np.load(LATENT_DIR / "latent_benign_test.npy")
    idx = rng.choice(mu_benign.shape[0], size=BENIGN_SUBSAMPLE_N, replace=False)
    mu_benign_sub = mu_benign[idx][:, ACTIVE_DIMS]

    groups = [("BENIGN", mu_benign_sub)]
    for name in MAIN_FAMILIES:
        mu = np.load(LATENT_DIR / f"latent_attack_{name}.npy")[:, ACTIVE_DIMS]
        groups.append((name, mu))

    X = np.vstack([g[1] for g in groups])
    labels = np.concatenate([[name] * len(g) for name, g in groups])
    print(f"[paso 1-3] datos combinados: n={X.shape[0]:,} "
          f"(benigno submuestreado={BENIGN_SUBSAMPLE_N:,} + 11 familias completas)", flush=True)
    for name, g in groups:
        print(f"  {name}: n={len(g):,}", flush=True)

    # --- Paso 4: una sola proyección UMAP sobre el conjunto combinado ---
    print("[paso 4] ajustando UMAP (verbose=True -- va a imprimir su propio progreso "
          "de vecinos/épocas más abajo)...", flush=True)
    t_fit = time.perf_counter()
    reducer = UMAP(random_state=SEED, n_neighbors=15, min_dist=0.1, n_components=2, verbose=True)
    embedding = reducer.fit_transform(X)
    print(f"[paso 4] UMAP ajustado en {time.perf_counter() - t_fit:.1f}s: "
          f"embedding shape={embedding.shape}", flush=True)

    # --- Paso 5: plot con centroides ---
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab20")
    group_names = [name for name, _ in groups]
    colors = {name: cmap(i / max(len(group_names) - 1, 1)) for i, name in enumerate(group_names)}

    for name in group_names:
        mask = labels == name
        pts = embedding[mask]
        is_benign = name == "BENIGN"
        ax.scatter(pts[:, 0], pts[:, 1], s=6 if is_benign else 10,
                   alpha=0.3 if is_benign else 0.6, color=colors[name],
                   label=name, zorder=1)
        centroid = pts.mean(axis=0)
        ax.scatter(*centroid, s=220, color=colors[name], edgecolor="black",
                   linewidth=1.5, marker="X", zorder=3)

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Proyección UMAP del espacio latente (6 dims activas) -- "
                  "benigno (n=3,000) + 11 familias\nApoyo visual, no valida la Fase 5 "
                  "(Spearman rho=0.38, no significativo)")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, markerscale=1.5)
    fig.tight_layout()
    fig_path = FIGURES_DIR / "oe2_latent_umap_projection.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[paso 5] figura guardada: {fig_path}", flush=True)
    print(f"[done] total: {time.perf_counter() - _t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
