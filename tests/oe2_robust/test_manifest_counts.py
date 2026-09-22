"""Verifica que el manifest por semilla de la Fase 2 de OE2 robusto tenga
EXACTAMENTE los mismos conteos por grupo que el manifest del OE2 original
de una sola semilla (mismos labels, mismas exclusiones, mismo solape
documentado de Web Attack Brute Force) -- si los conteos no coinciden, algo
en el filtrado de labels se desvió entre ambos pipelines.

Requiere que la Fase 2 ya se haya corrido al menos para seed=42 (genera
outputs/oe2_robust/latent_vectors/seed42/manifest.json); se salta si no
existe todavía, en vez de fallar, porque la Fase 2 corre por separado del
resto de la suite de tests."""
import json

import pytest

from vae_nids import config as base_cfg
from vae_nids.evaluation.oe2_robust import config as rcfg

NEW_MANIFEST_PATH = rcfg.LATENT_DIR / "seed42" / "manifest.json"
ORIGINAL_MANIFEST_PATH = base_cfg.PROJECT_ROOT / "outputs" / "latent_vectors" / "manifest.json"


@pytest.mark.skipif(not NEW_MANIFEST_PATH.exists(), reason="Fase 2 de oe2_robust no se ha corrido todavía para seed=42")
def test_seed42_manifest_counts_match_original_oe2():
    with open(NEW_MANIFEST_PATH, encoding="utf-8") as f:
        new_manifest = json.load(f)
    with open(ORIGINAL_MANIFEST_PATH, encoding="utf-8") as f:
        original_manifest = json.load(f)

    checked = 0
    for fname, info in original_manifest.items():
        if info.get("skipped"):
            continue
        name = "latent_benign_test" if fname == "latent_benign_test.npy" else fname[len("latent_attack_"):-len(".npy")]
        assert name in new_manifest, f"{name} ausente en el manifest nuevo"
        assert new_manifest[name]["n"] == info["n"], (
            f"{name}: n_original={info['n']} vs n_nuevo={new_manifest[name]['n']}"
        )
        checked += 1

    # 18 grupos de ataque + 1 de benigno = 19 entradas no-skipped en el
    # manifest original (el propio latent_benign_test.npy es una entrada más).
    assert checked == 19, f"esperaba verificar 19 grupos (18 ataque + 1 benigno), verifiqué {checked}"


def test_label_taxonomy_fully_covered_by_groups_and_exclusions():
    """Verifica cobertura de labels, no sumas de conteos de archivo: la
    suma directa de los 18 .npy (442,405) NO coincide con el total de
    test_attacks.parquet (442,952) porque faltan sumarle los labels con n
    insuficiente que se dejan fuera de todo grupo (698 filas en total:
    Heartbleed=11 -- listado en EXCLUDED_LABELS -- más otras 4 variantes
    "- Attempted" que simplemente no aparecen en ningún grupo, mismo
    comportamiento documentado en encode_latent_oe2.py original, mensaje
    "no pedidos, se dejan fuera"). El documento OBJETIVO_2_DETALLADO_v9
    original ya sumaba esas 698 filas en su cifra de "443,103" sin decirlo
    explícitamente. Aquí se verifica por cobertura de labels (más robusto
    que reproducir esa aritmética) y se fija la lista exacta de labels sin
    cubrir como chequeo de regresión: si aparece uno nuevo, debe ser una
    decisión explícita, no un olvido silencioso."""
    import pandas as pd

    from vae_nids.evaluation.oe2_robust.config import ATTACK_GROUPS, ATTEMPTED_GROUPS, EXCLUDED_LABELS

    EXPECTED_UNCOVERED = {
        "FTP-Patator - Attempted", "SSH-Patator - Attempted",
        "Infiltration - Attempted", "Web Attack - XSS - Attempted",
    }

    df = pd.read_parquet(base_cfg.OUTPUT_DIR / "test_attacks.parquet")
    covered_labels = set(EXCLUDED_LABELS)
    for labels in {**ATTACK_GROUPS, **ATTEMPTED_GROUPS}.values():
        covered_labels.update(labels)
    actual_labels = set(df[base_cfg.LABEL_COL].unique())
    uncovered = actual_labels - covered_labels

    assert uncovered == EXPECTED_UNCOVERED, (
        f"labels sin cubrir distintos de los esperados -- diff: "
        f"{uncovered.symmetric_difference(EXPECTED_UNCOVERED)}"
    )

    covered_row_count = df[df[base_cfg.LABEL_COL].isin(covered_labels)].shape[0]
    uncovered_row_count = df[df[base_cfg.LABEL_COL].isin(uncovered)].shape[0]
    assert covered_row_count + uncovered_row_count == len(df)
    # 687 = 11 (FTP-Patator - Attempted) + 8 (SSH-Patator - Attempted) +
    # 16 (Infiltration - Attempted) + 652 (Web Attack - XSS - Attempted).
    # Heartbleed (11 filas) también queda sin encodear pero SÍ está en
    # EXCLUDED_LABELS (cubierto), así que no entra en "uncovered".
    assert uncovered_row_count == 687
