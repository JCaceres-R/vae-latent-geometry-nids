"""
Configuración central del pipeline de datos — CICIDS2017 (versión corregida, Engelen et al. 2021)

Ajusta DATA_DIR a donde tengas los 5 CSV (Monday...Friday-WorkingHours.csv) en tu máquina.
"""
from pathlib import Path

# --- Rutas ---
# Ancladas a la raíz del proyecto (no al cwd del proceso que importa este módulo),
# para que funcionen igual desde `python -m vae_nids.data.pipeline` (cwd = raíz)
# como desde un notebook en notebooks/ (cwd = notebooks/).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"            # carpeta con los 5 CSV originales
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"    # aquí se guardan los .parquet + scaler
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILES = [
    "Monday-WorkingHours.csv",
    "Tuesday-WorkingHours.csv",
    "Wednesday-WorkingHours.csv",
    "Thursday-WorkingHours.csv",
    "Friday-WorkingHours.csv",
]

# --- Reproducibilidad ---
RANDOM_SEED = 42

# --- Lectura ---
CHUNKSIZE = 200_000  # ajustar según RAM disponible

# --- Columnas a excluir como features ---
# Flow ID, IPs y Timestamp: excluidas por fuga de datos (las IPs de
# atacante/víctima son fijas durante toda la simulación en CICIDS2017;
# dejarlas permitiría al modelo "memorizar direcciones" en vez de
# aprender patrones de tráfico).
IDENTIFIER_COLS = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]

# Puertos: opcional excluirlos también si te preocupa que el modelo
# aprenda "puerto 22 = SSH-Patator" en vez de patrones de tráfico.
# Se dejan fuera de las features por defecto; cámbialo aquí si tu
# arquitectura los necesita.
PORT_COLS = ["Src Port", "Dst Port"]

# Duplicados exactos (r = 1.0), identificados en el EDA (notebooks/eda_cicids2017.ipynb,
# Sección 7.3; ver data/processed/eda_exclusion_log.json -> dropped.exact_duplicates).
# Cada columna listada aquí reproduce, valor por valor, una estadística que ya está
# presente con el nombre estándar de CICFlowMeter -- se conserva ese nombre y se
# descarta el alias.
DUPLICATE_COLS = [
    "Bwd Segment Size Avg",  # == Bwd Packet Length Mean
    "Average Packet Size",   # == Packet Length Mean
    "Fwd Segment Size Avg",  # == Fwd Packet Length Mean
]

LABEL_COL = "Label"

# --- Split (sobre el subconjunto BENIGN únicamente) ---
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9

# Estratificar el split benigno por día, para que train/val/test
# conserven la misma diversidad temporal (no todo Monday en train, etc.)
STRATIFY_BENIGN_BY_DAY = True