"""Tests del OE1 saneado (no dependen de los datos reales ni de outputs/)."""
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from vae_nids.models.vae import VAE, VAEConfig
from vae_nids.oe1 import config as ocfg
from vae_nids.oe1 import datos, seleccion
from vae_nids.oe1.preproceso import MinMax64

UNIVERSE = list(yaml.safe_load(open(ocfg.REPO_ROOT / "configs/features/oe1_info_nula.yaml", encoding="utf-8"))["columnas"])


def synthetic_train(n=4000, seed=0):
    """Datos sintéticos con las 77 columnas, construidas con las fórmulas de cada identidad."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.integers(0, 50, n).astype(float) for c in UNIVERSE})
    df["Flow Duration"] = rng.integers(1, 10**8, n).astype(float)
    df["Total Fwd Packet"] = rng.integers(0, 40, n).astype(float)
    df["Total Bwd packets"] = rng.integers(0, 40, n).astype(float)
    df.loc[df["Total Fwd Packet"] + df["Total Bwd packets"] < 2, "Total Fwd Packet"] = 2.0
    df["Total Length of Fwd Packet"] = rng.integers(0, 10**5, n).astype(float)
    df["Total Length of Bwd Packet"] = rng.integers(0, 10**6, n).astype(float)
    df["Fwd IAT Total"] = np.where(df["Total Fwd Packet"] >= 2, rng.integers(0, 10**7, n), 0).astype(float)
    df["Bwd IAT Total"] = np.where(df["Total Bwd packets"] >= 2, rng.integers(0, 10**7, n), 0).astype(float)
    df["Packet Length Std"] = rng.random(n) * 300
    x = {c: df[c].to_numpy(np.float64) for c in df.columns}
    for idt in seleccion.IDENTIDADES:
        df[idt["col"]] = idt["f"](x)
        x[idt["col"]] = df[idt["col"]].to_numpy()
    for copia, keep in seleccion.COPIAS_PIPELINE.items():
        df[copia] = df[keep]
    df["Label"] = "BENIGN"
    df["Day"] = "Monday"
    df["row_id"] = np.arange(n)
    return df


@pytest.fixture
def cfg_stub(monkeypatch):
    calls = []
    df = synthetic_train()

    def fake_load_parts(cfg, names=("train_benign",), columns=None):
        calls.append(tuple(names))
        return {nm: df for nm in names}
    monkeypatch.setattr(seleccion, "load_parts", fake_load_parts)
    return calls


def test_identidades_sinteticas():
    df = synthetic_train()
    for idt in seleccion.IDENTIDADES:
        r, ok = seleccion.verify_identity(df, idt["col"], idt["f"], idt["tipo"], seleccion.EPS[idt["col"]][0])
        assert r["fallos"] == 0, idt["col"]


def test_identidad_falla_si_se_perturba():
    df = synthetic_train()
    df.loc[0, "Flow Packets/s"] *= 1 + 1e-9
    r, _ = seleccion.verify_identity(df, "Flow Packets/s", seleccion.IDENTIDADES[0]["f"], "continua",
                                     seleccion.EPS["Flow Packets/s"][0])
    assert r["fallos"] == 1


def test_seleccion_determinista_y_sin_circularidad(cfg_stub):
    _, _, est1, final1, _, recon1 = seleccion.select({})
    _, _, est2, final2, _, _ = seleccion.select({})
    assert final1 == final2
    assert ocfg.columns_hash(final1) == ocfg.columns_hash(final2)
    assert len(final1) == 77 - 3 - len(seleccion.IDENTIDADES)       # sintético: todas las identidades son exactas
    assert all(r["frac_verificada"] == 1.0 for r in recon1.values())


def test_seleccion_no_lee_ataques(cfg_stub):
    seleccion.select({})
    leidas = {n for call in cfg_stub for n in call}
    assert leidas == {"train_benign"}


def test_filtro_duracion_negativa_por_particion():
    df = pd.DataFrame({"Flow Duration": [5.0, -1.0, 3.0, -7.0], "row_id": [10, 11, 12, 13], "Label": ["BENIGN"] * 4})
    cfg = {"datos": {"filtrar_duracion_negativa": True, "etiqueta": "Label"}}
    out, rep = datos.filter_negative_duration({"train_benign": df}, cfg)
    assert out["train_benign"]["row_id"].tolist() == [10, 12]
    assert rep["train_benign"]["eliminadas"] == 2
    assert rep["train_benign"]["hash_ids_despues"] == datos.ids_hash([10, 12])


def test_escalador_solo_train_y_float64():
    tr = pd.DataFrame({"a": [0.0, 10.0, 5.0]})
    te = pd.DataFrame({"a": [20.0, -10.0]})
    sc = MinMax64().fit(tr, ["a"])
    assert sc.data_min.dtype == np.float64 and sc.data_max[0] == 10.0
    assert np.allclose(sc.transform(te)[:, 0], [2.0, -1.0])
    assert MinMax64.from_dict(sc.to_dict()).data_min[0] == 0.0


def test_config_falla_si_falta_clave(tmp_path):
    bad = tmp_path / "c.yaml"
    d = yaml.safe_load(open(ocfg.DEFAULT_CONFIG, encoding="utf-8"))
    del d["entrenamiento"]["min_delta"]
    yaml.safe_dump(d, open(bad, "w", encoding="utf-8"))
    with pytest.raises(ocfg.ConfigError):
        ocfg.load_config(bad)


def test_checkpoint_autocontenido(tmp_path):
    from vae_nids.oe1.entrenamiento import load_checkpoint
    cols = ["a", "b", "c"]
    mc = VAEConfig(input_dim=3, latent_dim=2, hidden_dim=4)
    torch.manual_seed(0)
    m = VAE(mc)
    sc = MinMax64([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], cols)
    ck = {"model_state_dict": m.state_dict(), "config": {}, "columnas": cols, "columnas_sha256": ocfg.columns_hash(cols),
          "escalador": sc.to_dict(), "modelo_config": {k: getattr(mc, k) for k in mc.__dataclass_fields__}}
    p = tmp_path / "ck.pt"
    torch.save(ck, p)
    m2, sc2, _ = load_checkpoint(p)
    x = torch.rand(5, 3)
    m.eval()
    with torch.no_grad():
        assert torch.equal(m.encoder(x)[0], m2.encoder(x)[0])
    assert sc2.columns == cols
    del ck["escalador"]
    torch.save(ck, p)
    with pytest.raises(KeyError):
        load_checkpoint(p)
