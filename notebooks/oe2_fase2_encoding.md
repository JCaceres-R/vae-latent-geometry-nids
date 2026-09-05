# OE2 — Fase 2: codificación de familias de ataque a mu

Solo codificación y guardado (mu = media de q_phi(z|x), nunca z muestreado
con reparameterization trick). Sin métricas todavía — eso es la fase
siguiente, sobre estos `.npy`. Contexto: `oe2_diagnostico_inicial.md`
(Fase 1).

## 0. Git

Commit `7e3b8c7` — "track OE1 checkpoints (seed sweep + base run)": 6
checkpoints `.pt` trackeados + `.gitignore` actualizado (se quitó la regla
`outputs/checkpoints/*`).

## 1-2. Modelo

Cargado en modo eval desde
`outputs/checkpoints/vae_k8_beta1_input51_best.pt`, con un assert interno
que verifica `seed==42` y `epoch==52` antes de codificar nada — si el
checkpoint no fuera el esperado, el script falla en vez de codificar con
el modelo equivocado. No se reentrenó nada.

## 3-5. Archivos generados

19 archivos `.npy` en `outputs/latent_vectors/`, todos con shape `(n, 8)`
confirmado (ninguno falló), más `manifest.json` con 19 entradas.

| Archivo | n | Labels |
|---|---:|---|
| `latent_benign_test.npy` | 248,561 | BENIGN |
| `latent_attack_PortScan.npy` | 159,023 | PortScan |
| `latent_attack_DoS_Hulk.npy` | 158,469 | DoS Hulk |
| `latent_attack_DDoS.npy` | 95,123 | DDoS |
| `latent_attack_DoS_GoldenEye.npy` | 7,567 | DoS GoldenEye |
| `latent_attack_DoS_slowloris.npy` | 4,001 | DoS slowloris |
| `latent_attack_FTP_Patator.npy` | 3,973 | FTP-Patator |
| `latent_attack_SSH_Patator.npy` | 2,980 | SSH-Patator |
| `latent_attack_DoS_Slowhttptest.npy` | 1,742 | DoS Slowhttptest |
| `latent_attack_Bot.npy` | 738 | Bot |
| `latent_attack_Web_Attack.npy` | 190 | Brute Force + XSS + Sql Injection (fusión) |
| `latent_attack_Infiltration.npy` | 32 | Infiltration |
| `latent_attack_DoS_Hulk_attempted.npy` | 579 | DoS Hulk - Attempted |
| `latent_attack_DoS_GoldenEye_attempted.npy` | 80 | DoS GoldenEye - Attempted |
| `latent_attack_DoS_slowloris_attempted.npy` | 1,706 | DoS slowloris - Attempted |
| `latent_attack_DoS_Slowhttptest_attempted.npy` | 3,367 | DoS Slowhttptest - Attempted |
| `latent_attack_Bot_attempted.npy` | 1,470 | Bot - Attempted |
| `latent_attack_WebAttack_BruteForce.npy` | 151 | Web Attack - Brute Force (solo, no la fusión) |
| `latent_attack_WebAttack_BruteForce_attempted.npy` | 1,214 | Web Attack - Brute Force - Attempted |

Todos los conteos coinciden exactamente con la taxonomía real de Fase 1.

**Verificación de consistencia**: suma de filas codificadas (443,103, con
el solape intencional de 151 filas de Brute Force contadas tanto en
`Web_Attack` como en `WebAttack_BruteForce`) menos ese solape = 442,952 =
total exacto de `test_attacks.parquet`.

**Excluidos de esta fase** (no pedidos, no un olvido):
- `Heartbleed` (11 filas, insuficiente)
- `FTP-Patator - Attempted` (11)
- `SSH-Patator - Attempted` (8)
- `Infiltration - Attempted` (16)
- `Web Attack - XSS - Attempted` (652)

Script: `src/vae_nids/evaluation/encode_latent_oe2.py`.
