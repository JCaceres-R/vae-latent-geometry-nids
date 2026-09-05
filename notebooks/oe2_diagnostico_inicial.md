# OE2 — Diagnóstico inicial (previo a implementación)

Solo lectura: conteo de labels, estado de git y confirmación del checkpoint
oficial antes de tocar modelo o pipeline. Contexto: cierre de OE1 en
`oe1_report.md` y `cluster_review_oe1.md`.

## 1. Taxonomía real de labels en `test_attacks.parquet`

442,952 filas, columna `Label` (`config.py:85`). 23 valores distintos —
incluye subtipos DoS por separado y las variantes `- Attempted` como
categorías propias (consistente con `README.md`, sección de preprocesamiento).

| Filas | % | Label exacto |
|---:|---:|---|
| 159,023 | 35.90% | `PortScan` |
| 158,469 | 35.78% | `DoS Hulk` |
| 95,123 | 21.47% | `DDoS` |
| 7,567 | 1.71% | `DoS GoldenEye` |
| 4,001 | 0.90% | `DoS slowloris` |
| 3,973 | 0.90% | `FTP-Patator` |
| 3,367 | 0.76% | `DoS Slowhttptest - Attempted` |
| 2,980 | 0.67% | `SSH-Patator` |
| 1,742 | 0.39% | `DoS Slowhttptest` |
| 1,706 | 0.39% | `DoS slowloris - Attempted` |
| 1,470 | 0.33% | `Bot - Attempted` |
| 1,214 | 0.27% | `Web Attack - Brute Force - Attempted` |
| 738 | 0.17% | `Bot` |
| 652 | 0.15% | `Web Attack - XSS - Attempted` |
| 579 | 0.13% | `DoS Hulk - Attempted` |
| 151 | 0.03% | `Web Attack - Brute Force` |
| 80 | 0.02% | `DoS GoldenEye - Attempted` |
| 32 | 0.01% | `Infiltration` |
| 27 | 0.01% | `Web Attack - XSS` |
| 16 | 0.00% | `Infiltration - Attempted` |
| 12 | 0.00% | `Web Attack - Sql Injection` |
| 11 | 0.00% | `FTP-Patator - Attempted` |
| 11 | 0.00% | `Heartbleed` |
| 8 | 0.00% | `SSH-Patator - Attempted` |

Nota: `DDoS` y `PortScan` no tienen subtipos ni variante `- Attempted`.
`Sql Injection` y `Heartbleed` no tienen variante `- Attempted` en este
dataset (0 filas cada una).

## 2. Estado de git — Sección 4 y datos

**Tracked** (todos los `.py` de modelo/training/métricas/viz — working tree
limpio, sin cambios pendientes):

- `src/vae_nids/config.py`
- `src/vae_nids/models/__init__.py`, `models/vae.py`, `models/vae_example.py`
- `src/vae_nids/training/__init__.py`, `training/train.py`
- `src/vae_nids/evaluation/__init__.py`, `evaluation/metrics_oe1.py`
- `src/vae_nids/viz/__init__.py`, `viz/training_curves.py`

**Solo en working tree** (ignorados por `.gitignore`; solo el `.gitkeep` de
cada carpeta está trackeado):

| Carpeta | Tamaño total |
|---|---:|
| `data/processed/` | 239 MB |
| `outputs/checkpoints/` | 204 KB |

Detalle:

| Archivo | Tamaño |
|---|---:|
| `data/processed/train_benign.parquet` | 146,085,234 B (~139 MB) |
| `data/processed/test_attacks.parquet` | 39,558,749 B (~38 MB) |
| `data/processed/test_benign.parquet` | 32,231,118 B (~31 MB) |
| `data/processed/val_benign.parquet` | 32,051,709 B (~31 MB) |
| `outputs/checkpoints/vae_k8_beta1_input74_best.pt` | 39,283 B |
| `outputs/checkpoints/vae_k8_beta1_input51_best.pt` | 30,019 B |
| `outputs/checkpoints/vae_k8_beta1_input51_seed43_best.pt` | 30,145 B |
| `outputs/checkpoints/vae_k8_beta1_input51_seed44_best.pt` | 30,145 B |
| `outputs/checkpoints/vae_k8_beta1_input51_seed45_best.pt` | 30,145 B |
| `outputs/checkpoints/vae_k8_beta1_input51_seed46_best.pt` | 30,145 B |

Los `.pt` son pequeños (~30 KB, arquitectura J=8/hidden=32) — trackearlos no
sería un problema de tamaño si se decide cambiar el `.gitignore`. Los
`.parquet` sí son pesados (239 MB total) y probablemente deban seguir
ignorados salvo que se use Git LFS.

## 3. Checkpoint oficial seed=42, input_dim=51

Ruta: `outputs/checkpoints/vae_k8_beta1_input51_best.pt`

Verificado inspeccionando el dict del checkpoint (no se cargó el modelo):

| Campo | Valor |
|---|---|
| `seed` | 42 |
| `epoch` | 52 |
| `val_loss` | -185.27911716408389 (≈ -185.2791 del doc de cierre) |
| `config` | `{input_dim: 51, latent_dim: 8, beta: 1.0, hidden_dim: 32, logvar_min: -10.0, logvar_max: 10.0}` |
| `feature_columns` | 51 nombres, termina en `Idle Min` (consistente con features conservadas independientes en la revisión de clusters) |

Confirmado: existe, en la ruta esperada, con los valores exactos reportados
en el cierre de OE1. No se cargó ni se usó para nada más.
