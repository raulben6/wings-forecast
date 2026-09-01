# -*- coding: utf-8 -*-
"""
La prueba que de verdad importa en un proyecto de series temporales.

Un modelo con fuga de información produce métricas excelentes y es inútil en
producción. Estas pruebas comprueban, en lugar de asumir, que ninguna variable
del día t contiene información del día t o posterior.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import caracteristicas, evaluacion, referencias


@pytest.fixture
def serie():
    """Serie sintética de 60 aperturas con patrón semanal claro."""
    dias = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    fechas = pd.bdate_range("2025-01-06", periods=60, freq="C",
                            weekmask="Mon Tue Wed Thu Fri Sat")
    rng = np.random.default_rng(0)
    base = {"Monday": 100, "Tuesday": 95, "Wednesday": 98,
            "Friday": 140, "Thursday": 115, "Saturday": 75}
    filas = []
    for i, f in enumerate(fechas):
        d = f.day_name()
        filas.append({"fecha": f, "dia_semana": d,
                      "total": base[d] + rng.normal(0, 5),
                      "efectivo": 50.0, "tarjeta": 30.0, "delivery": 20.0,
                      "t": i})
    return pd.DataFrame(filas)


def test_retardo_1_es_el_valor_anterior(serie):
    X = caracteristicas.construir(serie)
    # lag_1 en la fila i debe ser exactamente el total de la fila i-1.
    assert np.allclose(X["lag_1"].iloc[1:], serie["total"].iloc[:-1], equal_nan=False)


def test_retardo_semanal_apunta_a_la_semana_anterior(serie):
    X = caracteristicas.construir(serie)
    k = caracteristicas.SEMANA
    assert np.allclose(X[f"lag_{k}"].iloc[k:], serie["total"].iloc[:-k])
    # y cae en el mismo día de la semana, que es el sentido de usar 6 y no 7
    assert (X["dia_semana"].iloc[k:].to_numpy()
            == serie["dia_semana"].iloc[:-k].to_numpy()).all()


def test_medias_moviles_no_incluyen_el_dia_actual(serie):
    X = caracteristicas.construir(serie)
    # media_3 en la fila i = media de los totales i-3, i-2, i-1
    esperado = serie["total"].iloc[3:6].mean()
    assert X["media_3"].iloc[6] == pytest.approx(esperado)


def test_media_dia_semana_solo_mira_atras(serie):
    X = caracteristicas.construir(serie)
    fila = 20
    dia = X["dia_semana"].iloc[fila]
    previos = serie.loc[(serie.index < fila) & (serie["dia_semana"] == dia), "total"]
    assert X["media_dia_semana"].iloc[fila] == pytest.approx(previos.mean())


def test_ninguna_variable_correlaciona_perfecto_con_el_objetivo(serie):
    """Una correlación de 1.0 con el objetivo delata que la variable es el objetivo."""
    X = caracteristicas.construir(serie)
    cols = caracteristicas.columnas_predictoras(X)
    completo = X.dropna(subset=cols + ["total"])
    for c in cols:
        r = np.corrcoef(completo[c], completo["total"])[0, 1]
        assert abs(r) < 0.999, f"{c} reproduce el objetivo: posible fuga"


def test_walk_forward_nunca_entrega_el_futuro(serie):
    """El histórico que recibe el método debe terminar antes del día objetivo."""
    vistos = []

    def espia(historico, fila):
        vistos.append((historico["fecha"].max(), fila["fecha"]))
        return 0.0

    evaluacion.walk_forward(serie, espia, inicio=10)
    assert vistos, "no se evaluó ningún día"
    for ultimo_visto, objetivo in vistos:
        assert ultimo_visto < objetivo


def test_estacional_ingenuo_devuelve_el_mismo_dia_de_la_semana(serie):
    fila = serie.iloc[30]
    historico = serie.iloc[:30]
    pred = referencias.estacional_ingenuo(historico, fila)
    hace_una_semana = serie.iloc[30 - caracteristicas.SEMANA]
    assert pred == pytest.approx(hace_una_semana["total"])
    assert hace_una_semana["dia_semana"] == fila["dia_semana"]


def test_metricas_conocidas():
    real = np.array([100.0, 100.0, 100.0, 100.0])
    pred = np.array([90.0, 110.0, 100.0, 100.0])
    m = evaluacion.metricas(real, pred)
    assert m["MAE"] == pytest.approx(5.0)
    assert m["RMSE"] == pytest.approx(np.sqrt(50.0))
    assert m["MAPE"] == pytest.approx(5.0)


def test_prediccion_perfecta_da_error_cero():
    real = np.array([10.0, 20.0, 30.0])
    m = evaluacion.metricas(real, real)
    assert m["MAE"] == 0 and m["RMSE"] == 0 and m["MAPE"] == 0
