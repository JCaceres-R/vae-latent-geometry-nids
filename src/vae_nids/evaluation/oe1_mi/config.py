"""Configuración de OE1-IM -- rutas, semillas, constantes del estimador y la
definición documentada de la reducción original (74 -> 51), leída de los
artefactos existentes en vez de reescribirla a mano.
"""
import json

from vae_nids import config as base_cfg

# --- Semilla (submuestreo KSG, ruido KSG, simulaciones de calibración) ---
SEED = 42

# --- Rutas (todo bajo outputs/oe1_mi/, nunca se escribe en outputs/ raíz) ---
ROOT = base_cfg.PROJECT_ROOT / "outputs" / "oe1_mi"
METRICS_DIR = ROOT / "metrics"
FIGURES_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"
# Caché del dataset EDA reconstruido (~2.1M x 74, pesado) -- en data/processed/,
# que ya está en .gitignore; es regenerable desde data/raw/.
CACHE_DIR = base_cfg.OUTPUT_DIR / "oe1_mi_cache"

for d in (METRICS_DIR, FIGURES_DIR, TABLES_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

EXCLUSION_LOG_PATH = base_cfg.OUTPUT_DIR / "eda_exclusion_log.json"
FEATURE_COLUMNS_51_PATH = base_cfg.OUTPUT_DIR / "feature_columns.json"

# --- Estimador ---
B_DEFAULT = 20              # intervalos nominales (igual frecuencia)
B_SENSITIVITY = [10, 20, 50]
KSG_N_NEIGHBORS = 3
KSG_SUBSAMPLE = 50_000

# --- Criterios ---
PEARSON_THRESHOLD = 0.95    # criterio P (original)
RINFO_THRESHOLD = 0.95      # criterio L (Linfoot)
ADAPTIVE_K = [0.5, 1.0, 1.5, 2.0]
LOW_INFO_HNORM = 0.1
LOW_INFO_MODE = 0.99

# Orden de esquema CICFlowMeter de las 74 features (= orden de columnas del
# CSV crudo tras quitar IDs/puertos/duplicados exactos; es exactamente la
# lista `feature_columns` del checkpoint vae_k8_beta1_input74_best.pt).
FEATURES_74 = [
    "Protocol", "Flow Duration", "Total Fwd Packet", "Total Bwd packets",
    "Total Length of Fwd Packet", "Total Length of Bwd Packet",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
    "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max",
    "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags",
    "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s",
    "Bwd Packets/s", "Packet Length Min", "Packet Length Max",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWR Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Fwd Bytes/Bulk Avg", "Fwd Packet/Bulk Avg",
    "Fwd Bulk Rate Avg", "Bwd Bytes/Bulk Avg", "Bwd Packet/Bulk Avg",
    "Bwd Bulk Rate Avg", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "FWD Init Win Bytes",
    "Bwd Init Win Bytes", "Fwd Act Data Pkts", "Fwd Seg Size Min", "Active Mean",
    "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max",
    "Idle Min",
]
assert len(FEATURES_74) == 74

# Los 13 clusters del EDA (union-find |r|>=0.9, celda 38 de
# notebooks/eda_cicids2017.ipynb), sin los 3 duplicados exactos ya fuera.
EDA_CLUSTERS = {
    0: ["Bwd Packet Length Max", "Bwd Packet Length Mean", "Bwd Packet Length Std",
        "Packet Length Max", "Packet Length Mean", "Packet Length Std",
        "Packet Length Variance", "Subflow Bwd Bytes"],
    1: ["ACK Flag Count", "Bwd Header Length", "Fwd Header Length",
        "Total Bwd packets", "Total Fwd Packet", "Total Length of Bwd Packet"],
    2: ["Bwd IAT Max", "Flow IAT Max", "Fwd IAT Max", "Idle Max", "Idle Mean", "Idle Min"],
    3: ["Bwd IAT Mean", "Bwd IAT Min", "Flow IAT Mean", "Flow IAT Std",
        "Fwd IAT Mean", "Fwd IAT Min"],
    4: ["Fwd Packet Length Mean", "Subflow Fwd Bytes"],
    5: ["Bwd IAT Total", "Flow Duration", "Fwd IAT Total"],
    6: ["Fwd URG Flags", "URG Flag Count"],
    7: ["Bwd Bytes/Bulk Avg", "Bwd Packet/Bulk Avg"],
    8: ["Flow Packets/s", "Fwd Packets/s"],
    9: ["Bwd PSH Flags", "PSH Flag Count"],
    10: ["Fwd Packet Length Max", "Fwd Packet Length Std"],
    11: ["Bwd IAT Std", "Fwd IAT Std"],
    12: ["Active Mean", "Active Min"],
}

# Grupos fusionados documentados (eda_exclusion_log.json -> resolved_clusters /
# cluster_review_oe1.md): {representante: [miembros del grupo, incl. el rep]}.
# Packet Length Variance NO está aquí: se descartó a priori por var = std^2
# (dropped.cluster_collapse[0]), no por el criterio de clique.
DOCUMENTED_GROUPS = {
    "Packet Length Std": ["Bwd Packet Length Max", "Bwd Packet Length Std",
                          "Packet Length Max", "Packet Length Std"],
    "Packet Length Mean": ["Bwd Packet Length Mean", "Packet Length Mean", "Subflow Bwd Bytes"],
    "Total Fwd Packet": ["ACK Flag Count", "Bwd Header Length", "Fwd Header Length",
                         "Total Bwd packets", "Total Fwd Packet", "Total Length of Bwd Packet"],
    "Flow IAT Max": ["Flow IAT Max", "Fwd IAT Max", "Idle Max", "Idle Mean"],
    "Bwd IAT Mean": ["Bwd IAT Mean", "Bwd IAT Min"],
    "Fwd IAT Mean": ["Fwd IAT Mean", "Fwd IAT Min"],
    "Fwd Packet Length Mean": ["Fwd Packet Length Mean", "Subflow Fwd Bytes"],
    "Flow Duration": ["Bwd IAT Total", "Flow Duration", "Fwd IAT Total"],
    "URG Flag Count": ["Fwd URG Flags", "URG Flag Count"],
    "Bwd Bytes/Bulk Avg": ["Bwd Bytes/Bulk Avg", "Bwd Packet/Bulk Avg"],
    "Flow Packets/s": ["Flow Packets/s", "Fwd Packets/s"],
    "PSH Flag Count": ["Bwd PSH Flags", "PSH Flag Count"],
}
A_PRIORI_DROPS = {"Packet Length Variance": "Packet Length Std"}


def load_features_51() -> list[str]:
    with open(FEATURE_COLUMNS_51_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_documented_drops() -> list[str]:
    """Las 23 columnas descartadas según `config.CORRELATED_CLUSTER_DROP_COLS`
    (fuente de verdad del pipeline)."""
    return list(base_cfg.CORRELATED_CLUSTER_DROP_COLS)


def feature_to_eda_cluster() -> dict[str, int]:
    return {f: cid for cid, members in EDA_CLUSTERS.items() for f in members}
