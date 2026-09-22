"""Configuración central de OE2 robusto -- rutas, semillas, constantes de
cada fase. Reutiliza la definición de grupos de ataque de
`encode_latent_oe2` (Fase 2 del OE2 original) en vez de redefinirla, para
que ambos pipelines codifiquen exactamente los mismos 19 archivos por
semilla.
"""
from pathlib import Path

from vae_nids import config as base_cfg
from vae_nids.evaluation.encode_latent_oe2 import (
    ATTACK_GROUPS,
    ATTEMPTED_GROUPS,
    EXCLUDED_LABELS,
)

# --- Semillas ---
# 42 = checkpoint oficial de OE1/OE2 (outputs/checkpoints/vae_k8_beta1_input51_best.pt),
# NO se reentrena -- se usa tal cual para el análisis. Las otras 9 son nuevas,
# entrenadas por este paquete. Ver Fase 0 del informe para la justificación
# de usar las 10 (vs. el mínimo de 5 pedido en el prompt).
OFFICIAL_SEED = 42
NEW_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
ALL_SEEDS = [OFFICIAL_SEED] + NEW_SEEDS

# --- Rutas (todo bajo outputs/oe2_robust/, nunca se escribe en outputs/ raíz) ---
ROOT = base_cfg.PROJECT_ROOT / "outputs" / "oe2_robust"
CHECKPOINTS_DIR = ROOT / "checkpoints"
LOGS_DIR = ROOT / "logs"
LATENT_DIR = ROOT / "latent_vectors"
METRICS_DIR = ROOT / "metrics"
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"
REPORTS_DIR = ROOT

for d in (CHECKPOINTS_DIR, LOGS_DIR, LATENT_DIR, METRICS_DIR, TABLES_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

OFFICIAL_CHECKPOINT_PATH = base_cfg.PROJECT_ROOT / "outputs" / "checkpoints" / "vae_k8_beta1_input51_best.pt"

# --- Arquitectura / entrenamiento (idéntico a OE1/OE2 original) ---
LATENT_DIM = 8
BETA = 1.0
HIDDEN_DIM = 32
BATCH_SIZE = 1024
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
MIN_DELTA = 0.0

# --- Fase 2: L muestras Monte Carlo del posterior ---
L_MC_SAMPLES = 20
MC_SEED_BASE = 1000  # semilla de muestreo MC = MC_SEED_BASE + seed_modelo, documentado en run_metadata

# --- Fase 3: geometría distribucional ---
ACTIVE_THRESHOLD = 0.01
# Tamaños de submuestra por lado para las métricas O(n^2) de esta fase
# (kernel RBF, matrices de distancia par-a-par, silhouette). Reducidos de
# 5000 a 1500 tras medir timing real: a n=5000, MMD+SW+energy+silhouette
# tomaban ~11s por repetición; a n=1500, ~1.3s -- con 18 familias x 20
# repeticiones x 10 semillas (3600 llamadas), 5000 hacía el pipeline
# inviable (>6h solo en estas dos subfases) mientras que 1500 lo deja en
# ~75-90 min. Decisión tomada en la Fase 0/inicio de la Fase 3, ANTES de
# ver ningún resultado de geometría o detectabilidad -- documentada en el
# informe, Sección 10.
SILHOUETTE_MAX_N = 1500
N_SILHOUETTE_REPEATS = 20
DIVERGENCE_MAX_N = 1500
N_DIVERGENCE_REPEATS = 20
N_BOOTSTRAP_GEOMETRY = 500
KNN_K = 10
KNN_REFERENCE_MAX_N = 20000  # tamaño del conjunto de referencia benigno+ataque para pureza de vecindad / densidad

# --- Fase 4: detectabilidad ---
TAU_PERCENTILE = 95
N_BOOTSTRAP_AUC = 1000
N_BOOTSTRAP_SEED = 42
FPR_TARGETS = [0.01, 0.05, 0.10]
PAUC_FPR_MAX = 0.10
SAMPLE_SIZE_CONTROL_REPEATS = 100

# --- Fase 5: correlación ---
MAIN_FAMILIES = [
    "PortScan", "DoS_Hulk", "DDoS", "DoS_GoldenEye", "DoS_slowloris",
    "FTP_Patator", "SSH_Patator", "DoS_Slowhttptest", "Bot", "Web_Attack",
    "Infiltration",
]
ATTEMPTED_PAIRS = [
    ("DoS_Hulk", "DoS_Hulk_attempted"),
    ("DoS_GoldenEye", "DoS_GoldenEye_attempted"),
    ("DoS_slowloris", "DoS_slowloris_attempted"),
    ("DoS_Slowhttptest", "DoS_Slowhttptest_attempted"),
    ("Bot", "Bot_attempted"),
    ("WebAttack_BruteForce", "WebAttack_BruteForce_attempted"),
]
N_PERMUTATION = 10_000
PERMUTATION_SEED = 42
MIXED_MODEL_MAX_N_PER_GROUP_PER_SEED = 5000
MIXED_MODEL_SEED = 42

# --- Fase 6: visualización ---
UMAP_SEEDS = [42, 0, 1]  # semillas representativas para la proyección UMAP
UMAP_RANDOM_STATE = 42
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_BENIGN_SUBSAMPLE = 3000

__all__ = [
    "ATTACK_GROUPS", "ATTEMPTED_GROUPS", "EXCLUDED_LABELS",
    "OFFICIAL_SEED", "NEW_SEEDS", "ALL_SEEDS",
    "ROOT", "CHECKPOINTS_DIR", "LOGS_DIR", "LATENT_DIR", "METRICS_DIR",
    "TABLES_DIR", "FIGURES_DIR", "REPORTS_DIR", "OFFICIAL_CHECKPOINT_PATH",
]
