"""Carga de la configuración única del OE1 saneado (configs/oe1_saneado.yaml).

No hay valores por defecto en el código: si falta una clave, falla. Las rutas son
relativas a la raíz del repositorio y se pueden sobrescribir con variables de entorno.
"""
import hashlib
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "oe1_saneado.yaml"
ENV_OVERRIDES = {"raw_dir": "VAE_NIDS_RAW_DIR", "processed_dir": "VAE_NIDS_PROCESSED_DIR"}

REQUIRED = {
    "datos": ["raw_dir", "processed_dir", "csv", "columnas_no_flujo", "etiqueta", "semilla_split",
              "fraccion_train", "fraccion_val", "fraccion_test", "estratificar_por_dia",
              "filtrar_duracion_negativa"],
    "columnas": ["conjunto", "sha256"],
    "modelo": ["latent_dim", "hidden_dim", "beta", "logvar_min", "logvar_max"],
    "entrenamiento": ["optimizador", "lr", "batch_size", "patience", "min_delta", "max_epochs",
                      "fraccion_muestra", "nombre_corrida"],
    "semillas": ["principal", "referencia_74", "general"],
    "cpu": ["hilos", "deterministico"],
    "evaluacion": ["tau_percentil", "umbral_unidades_activas"],
    "salidas": ["dir"],
}


class ConfigError(RuntimeError):
    pass


def load_config(path=None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise ConfigError(f"No existe el archivo de configuración {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path} no contiene un mapeo YAML")
    for sec, keys in REQUIRED.items():
        if sec not in cfg:
            raise ConfigError(f"Falta la sección '{sec}' en {path}")
        for k in keys:
            if k not in cfg[sec]:
                raise ConfigError(f"Falta la clave '{sec}.{k}' en {path}")
    cfg["_path"] = str(path)
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    """Ruta absoluta de datos.<key> o salidas.dir, con variable de entorno si existe."""
    if key == "salidas":
        p = os.environ.get("VAE_NIDS_OUTPUTS_DIR", cfg["salidas"]["dir"])
    else:
        p = os.environ.get(ENV_OVERRIDES.get(key, ""), cfg["datos"][key])
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def repo_path(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def columns_hash(columns: list[str]) -> str:
    return hashlib.sha256("\n".join(columns).encode("utf-8")).hexdigest()


def load_feature_set(cfg: dict) -> list[str]:
    """Lista ordenada de columnas que quedan, verificada contra el hash de la configuración."""
    with open(repo_path(cfg["columnas"]["conjunto"]), encoding="utf-8") as f:
        fs = yaml.safe_load(f)
    cols = fs["columnas_que_quedan"]
    h = columns_hash(cols)
    if h != fs["sha256"]:
        raise ConfigError("El hash del conjunto congelado no coincide con su lista")
    if cfg["columnas"]["sha256"] != h:
        raise ConfigError(f"El hash de la configuración ({cfg['columnas']['sha256']}) no coincide con el "
                          f"del conjunto congelado ({h})")
    return cols
