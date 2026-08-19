"""
Pipeline de datos — CICIDS2017 corregido (Engelen et al., 2021)

Uso:
    python -m vae_nids.data.pipeline

Salida (en OUTPUT_DIR):
    train_benign.parquet   -> 70% del tráfico benigno (para entrenar el AE/VAE)
    val_benign.parquet     -> 15% del tráfico benigno (early stopping, umbral tau)
    test_benign.parquet    -> 15% del tráfico benigno (evaluación FPR)
    test_attacks.parquet   -> 100% del tráfico malicioso (evaluación TPR por familia)
    scaler.joblib           -> MinMaxScaler ajustado SOLO sobre train_benign
    feature_columns.json    -> lista ordenada de columnas usadas como features
    label_report.csv        -> conteo final de labels tras la limpieza
"""
import json
import glob
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from vae_nids import config as cfg


def load_and_merge() -> pd.DataFrame:
    """Carga los 5 CSV por chunks y los concatena, agregando 'Day' para trazabilidad.

    Memoria: descarta IDENTIFIER_COLS/PORT_COLS y convierte features a float32
    ya en el momento de lectura (en vez de cargar las 85 columnas completas y
    recortar después). Con esto, 2.1M filas caben cómodamente en <1GB de RAM
    en lugar de los ~3-4GB que ocupa la versión "ingenua".
    """
    drop_at_load = set(cfg.IDENTIFIER_COLS + cfg.PORT_COLS)
    frames = []
    for fname in cfg.CSV_FILES:
        fpath = cfg.DATA_DIR / fname
        if not fpath.exists():
            raise FileNotFoundError(f"No encuentro {fpath}. Revisa config.DATA_DIR")
        day = fname.split("-")[0]
        print(f"[load] leyendo {fname} ...")

        day_chunks = []
        n_rows = 0
        for chunk in pd.read_csv(fpath, chunksize=cfg.CHUNKSIZE, low_memory=False):
            chunk.columns = chunk.columns.str.strip()
            cols_to_drop = [c for c in drop_at_load if c in chunk.columns]
            chunk = chunk.drop(columns=cols_to_drop)

            numeric_cols = chunk.select_dtypes(include=[np.number]).columns
            chunk[numeric_cols] = chunk[numeric_cols].astype("float32")
            chunk["Day"] = day

            day_chunks.append(chunk)
            n_rows += len(chunk)

        df_day = pd.concat(day_chunks, ignore_index=True)
        del day_chunks
        frames.append(df_day)
        print(f"[load]   {n_rows:,} filas")

    full = pd.concat(frames, ignore_index=True)
    del frames
    print(f"[load] total combinado: {len(full):,} filas, {full.shape[1]} columnas")
    return full


def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas con nulos o infinitos en columnas numéricas (fuga inducida
    por divisiones por cero en tasas de paquetes temporales, ver Flow Bytes/s,
    Flow Packets/s, Flow IAT *)."""
    n_before = len(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=numeric_cols)

    n_after = len(df)
    print(f"[sanitize] eliminadas {n_before - n_after:,} filas "
          f"({(n_before - n_after) / n_before:.4%}) por nulos/infinitos")
    return df.reset_index(drop=True)


def build_label_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva columnas auxiliares de etiquetado sin perder granularidad:

    - is_benign: bool, True solo para 'BENIGN'
    - is_attempted: bool, True para cualquier label con sufijo '- Attempted'
    - attack_family: familia base del ataque sin el sufijo Attempted
      (se conserva SOLO como referencia para agregaciones; el label
      original permanece intacto en 'Label' para evaluación aislada
      por clase, tal como se decidió: Attempted queda como clase propia).
    """
    df["is_benign"] = df[cfg.LABEL_COL] == "BENIGN"
    df["is_attempted"] = df[cfg.LABEL_COL].str.contains("- Attempted", regex=False)
    df["attack_family"] = df[cfg.LABEL_COL].str.replace(
        " - Attempted", "", regex=False
    )
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    drop_cols = set(
        cfg.IDENTIFIER_COLS
        + cfg.PORT_COLS
        + cfg.DUPLICATE_COLS
        + [cfg.LABEL_COL, "Day", "is_benign", "is_attempted", "attack_family"]
    )
    feature_cols = [c for c in df.columns if c not in drop_cols]
    # Solo columnas numéricas entran al modelo
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    print(f"[features] {len(feature_cols)} columnas seleccionadas como features "
          f"(verifica que coincida con tu arquitectura, ej. x ∈ R^74)")
    return feature_cols


def split_benign(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/15/15 estratificado por día, SOLO sobre tráfico benigno."""
    benign = df[df["is_benign"]].reset_index(drop=True)
    strat = benign["Day"] if cfg.STRATIFY_BENIGN_BY_DAY else None

    train, temp = train_test_split(
        benign,
        train_size=cfg.TRAIN_FRAC,
        random_state=cfg.RANDOM_SEED,
        stratify=strat,
    )
    val_size_rel = cfg.VAL_FRAC / (cfg.VAL_FRAC + cfg.TEST_FRAC)
    strat_temp = temp["Day"] if cfg.STRATIFY_BENIGN_BY_DAY else None
    val, test = train_test_split(
        temp,
        train_size=val_size_rel,
        random_state=cfg.RANDOM_SEED,
        stratify=strat_temp,
    )
    print(f"[split] benign train={len(train):,} val={len(val):,} test={len(test):,}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def scale_no_leakage(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
    attacks: pd.DataFrame, feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Ajusta MinMaxScaler EXCLUSIVAMENTE sobre train y lo aplica de forma
    estática a val/test/attacks, evitando fuga de datos.

    Transforma in-place (sin .copy() de los DataFrames completos) para
    mantener el pico de memoria bajo.
    """
    scaler = MinMaxScaler()
    train[feature_cols] = scaler.fit_transform(train[feature_cols]).astype("float32")
    val[feature_cols] = scaler.transform(val[feature_cols]).astype("float32")
    test[feature_cols] = scaler.transform(test[feature_cols]).astype("float32")
    attacks[feature_cols] = scaler.transform(attacks[feature_cols]).astype("float32")

    return train, val, test, attacks, scaler


def main():
    df = load_and_merge()
    df = sanitize(df)
    df = build_label_taxonomy(df)

    label_report = df[cfg.LABEL_COL].value_counts().rename_axis("Label").reset_index(name="count")
    label_report.to_csv(cfg.OUTPUT_DIR / "label_report.csv", index=False)
    print("[report] label_report.csv guardado")

    feature_cols = get_feature_columns(df)
    with open(cfg.OUTPUT_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    train, val, test = split_benign(df)
    attacks = df[~df["is_benign"]].reset_index(drop=True)
    print(f"[split] total ataques (todas las familias, incl. Attempted): {len(attacks):,}")

    train_s, val_s, test_s, attacks_s, scaler = scale_no_leakage(
        train, val, test, attacks, feature_cols
    )

    train_s.to_parquet(cfg.OUTPUT_DIR / "train_benign.parquet", index=False)
    val_s.to_parquet(cfg.OUTPUT_DIR / "val_benign.parquet", index=False)
    test_s.to_parquet(cfg.OUTPUT_DIR / "test_benign.parquet", index=False)
    attacks_s.to_parquet(cfg.OUTPUT_DIR / "test_attacks.parquet", index=False)
    joblib.dump(scaler, cfg.OUTPUT_DIR / "scaler.joblib")

    print(f"[done] archivos guardados en {cfg.OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()