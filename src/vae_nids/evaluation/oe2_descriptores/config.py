"""Configuración de la evaluación de descriptores geométricos del OE2.

Reutiliza las semillas, familias y vectores latentes del pipeline robusto
(`oe2_robust`): no reentrena ni recodifica nada que ya exista. Todo lo nuevo
se escribe bajo outputs/oe2_descriptores/.

Diseño (fijado antes de cruzar ningún descriptor nuevo con el AUC):
- Referencia benigna = val_benign (no se usó para entrenar ni para el AUC),
  sin duplicados exactos de mu. Control benigno = test_benign (el mismo
  negativo del AUC). Así los umbrales "percentil 95 del benigno" se fijan sin
  fuga y cada descriptor tiene un valor nulo de referencia.
- Desenlace = AUC-ROC del puntaje principal del proyecto (nll_mc, L=20), por
  familia y semilla, con el mismo muestreo que phase4_detectability.
- Un solo resumen por descriptor, declarado en DESCRIPTORES (abajo).
"""
from vae_nids import config as base_cfg
from vae_nids.evaluation.oe2_robust import config as rcfg

SEEDS = rcfg.ALL_SEEDS
MAIN_FAMILIES = rcfg.MAIN_FAMILIES
ATTEMPTED_PAIRS = rcfg.ATTEMPTED_PAIRS
ALL_GROUPS = list(rcfg.ATTACK_GROUPS) + list(rcfg.ATTEMPTED_GROUPS)
GROUP_LABELS = {**rcfg.ATTACK_GROUPS, **rcfg.ATTEMPTED_GROUPS}
LATENT_DIR = rcfg.LATENT_DIR

# Perfil (VAE_NIDS_OE2_PERFIL, ver oe2_robust/config.py): original = modelo de 51 variables;
# saneado = modelos del OE1 saneado (66 variables), resultados aparte.
PERFIL = rcfg.PERFIL
ROOT_ORIGINAL = base_cfg.PROJECT_ROOT / "outputs" / "oe2_descriptores"
if PERFIL == "original":
    ROOT = ROOT_ORIGINAL
elif rcfg.CONFIG_PERSONALIZADA:
    ROOT = rcfg.SALIDAS_CONFIG / "oe2_descriptores"
else:
    ROOT = base_cfg.PROJECT_ROOT / "outputs" / "oe2_descriptores_saneado"
CACHE_DIR = ROOT / "_cache"
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"
for _d in (CACHE_DIR, TABLES_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

ACTIVE_THRESHOLD = rcfg.ACTIVE_THRESHOLD
L_MC = rcfg.L_MC_SAMPLES
MC_SEED_BASE = rcfg.MC_SEED_BASE
BETA = rcfg.BETA
LOGVAR_MIN, LOGVAR_MAX = -10.0, 10.0

# Referencia benigna y vecinos
REF_MAX_N = 50_000           # filas únicas de val_benign usadas como referencia
K_KNN = 10                   # Sun et al. (2022): k-ésimo vecino
K_LID = 20                   # Ma et al. (2018): k=20 vecinos de referencia
FR_CANDIDATES = 64           # preselección W2 para Fisher-Rao
CONTROL_PERCENTILE = 95      # umbral "más lejos que el p95 del control"

# Descriptores de conjunto (O(n^2)): submuestras sin duplicados
SET_MAX_N = 1500
SET_REPEATS = 5
SIL_MAX_N = 1000
PROBE_MAX_N = 20_000
GMM_K = 8

# Muestra por flujo (fase 2 -> fase 7): hasta FLOW_SAMPLE_MAX flujos por familia principal, con un
# generador aparte (no altera el resto de la fase 2), para separar el efecto entre familias del efecto
# dentro de cada familia.
FLOW_SAMPLE_MAX = 5000
FLOW_SAMPLE_DESCRIPTORS = ["maha", "knn", "fr", "rate", "dec_knn", "logmag"]

# Estadística
N_BOOT = 2000
N_PERM = 10_000
RNG_SEED = 42
