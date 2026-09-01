# -*- coding: utf-8 -*-
"""
Pruebas del análisis por producto.

Se ejecutan contra un dataset sintético construido a propósito, porque el POS
todavía no tiene historial. Verifican la lógica, que es lo que debe estar
correcto el día que lleguen los datos reales.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import productos


@pytest.fixture
def ventas():
    """
    90 días, 4 productos con cuotas de facturación muy desiguales.

    Las cuotas están puestas a mano para poder comprobar la clasificación ABC:
    'Alas' domina, 'Salsa' es residual.
    """
    fechas = pd.date_range("2026-01-05", periods=90, freq="D")
    fechas = [f for f in fechas if f.day_name() != "Sunday"]
    catalogo = [
        ("Alas", "Combos", 60.0),
        ("Hamburguesa", "Hamburguesas", 25.0),
        ("Chunks", "Combos", 12.0),
        ("Salsa extra", "Extras", 3.0),
    ]
    filas = []
    for f in fechas:
        for nombre, cat, peso in catalogo:
            filas.append({"fecha": f, "producto": nombre, "categoria": cat,
                          "canal": "local", "unidades": peso / 10, "importe": peso})
    return pd.DataFrame(filas).assign(dia_semana=lambda d: d["fecha"].dt.day_name())


def test_pareto_ordena_y_clasifica(ventas):
    abc = productos.pareto(ventas)
    assert list(abc.index[:2]) == ["Alas", "Hamburguesa"]
    assert abc.loc["Alas", "cuota %"] == pytest.approx(60.0, abs=0.1)
    # el acumulado termina en 100
    assert abc["acumulado %"].iloc[-1] == pytest.approx(100.0, abs=0.1)
    # la cola es clase C
    assert abc.loc["Salsa extra", "clase"] == "C"


def test_pareto_clase_a_cubre_hasta_el_80(ventas):
    abc = productos.pareto(ventas)
    clase_a = abc[abc["clase"] == "A"]
    assert clase_a["acumulado %"].max() <= 80.0


def test_serie_producto_rellena_los_dias_sin_venta_con_cero(ventas):
    """Un día sin línea es demanda cero, no un dato ausente."""
    recortado = ventas[~((ventas["producto"] == "Chunks") &
                         (ventas["fecha"] == ventas["fecha"].iloc[10]))]
    serie = productos.serie_producto(recortado, "Chunks")
    assert serie["total"].isna().sum() == 0
    assert (serie["total"] == 0).sum() >= 1


def test_serie_producto_excluye_domingos(ventas):
    serie = productos.serie_producto(ventas, "Alas")
    assert "Sunday" not in serie["dia_semana"].values


def test_serie_producto_es_continua_en_el_tiempo(ventas):
    serie = productos.serie_producto(ventas, "Alas")
    assert list(serie["t"]) == list(range(len(serie)))
    assert serie["fecha"].is_monotonic_increasing


def test_producto_inexistente_falla_claro(ventas):
    with pytest.raises(KeyError):
        productos.serie_producto(ventas, "no existe")


def test_suficiencia_bloquea_con_poco_historial():
    """La puerta que impide publicar métricas sobre una muestra insuficiente."""
    corto = pd.DataFrame({
        "fecha": pd.to_datetime(["2026-08-10"] * 3),
        "producto": ["a", "b", "c"],
        "categoria": ["x"] * 3, "canal": ["local"] * 3,
        "unidades": [1.0] * 3, "importe": [1.0] * 3,
    })
    v = productos.comprobar_suficiencia(corto)
    assert v["suficiente"] is False
    assert v["dias"] == 1
    assert "1 días" in v["motivo"]


def test_suficiencia_acepta_historial_largo(ventas):
    v = productos.comprobar_suficiencia(ventas)
    assert v["suficiente"] is True
    assert v["productos_modelables"] == 4


def test_mezcla_por_canal_reparte_el_cien_por_ciento(ventas):
    mezclado = pd.concat([ventas, ventas.assign(canal="delivery", importe=ventas["importe"] / 2)])
    t = productos.mezcla_por_canal(mezclado)
    for col in [c for c in t.columns if c.endswith("%")]:
        assert t[col].sum() == pytest.approx(100.0, abs=0.1)


def test_patron_semanal_solo_dias_operativos(ventas):
    t = productos.patron_semanal(ventas)
    assert "Sunday" not in t.columns
    assert len(t) <= 8
