# -*- coding: utf-8 -*-
"""
Construcción de variables predictoras.

Regla que gobierna todo este módulo: **ninguna variable puede usar información
que no existiera el día anterior a la predicción.** Un retardo mira hacia
atrás; una media móvil se calcula cerrada en t-1. Cualquier cosa que use el
valor del propio día es fuga de información y produce métricas falsas.
"""

import numpy as np
import pandas as pd

# 6 aperturas = una semana natural, porque el domingo no está en la serie.
SEMANA = 6

DIAS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def construir(df, retardos=(1, 2, SEMANA, SEMANA * 2), ventanas=(3, SEMANA, SEMANA * 4)):
    """Añade calendario, retardos y medias móviles. Devuelve una copia."""
    X = df.copy()

    # --- Calendario ---------------------------------------------------------
    for dia in DIAS[1:]:                       # lunes queda como categoría base
        X[f"es_{dia.lower()}"] = (X["dia_semana"] == dia).astype(int)

    X["dia_mes"] = X["fecha"].dt.day
    X["mes"] = X["fecha"].dt.month

    # En El Salvador los sueldos se pagan alrededor del 15 y de fin de mes.
    # Es una hipótesis de dominio, no un hecho: el modelo dirá si aporta.
    X["quincena"] = X["dia_mes"].between(14, 17).astype(int)
    X["fin_de_mes"] = ((X["dia_mes"] >= 28) | (X["dia_mes"] <= 2)).astype(int)

    # Tendencia lineal suave, para que el modelo pueda captar deriva.
    X["tendencia"] = X["t"] / len(X)

    # --- Retardos -----------------------------------------------------------
    for k in retardos:
        X[f"lag_{k}"] = X["total"].shift(k)

    # --- Medias móviles, cerradas en t-1 ------------------------------------
    for w in ventanas:
        X[f"media_{w}"] = X["total"].shift(1).rolling(w).mean()

    # Media histórica del mismo día de la semana, expandida y desplazada:
    # en la fila t solo entra información de días anteriores a t.
    X["media_dia_semana"] = (
        X.groupby("dia_semana")["total"]
         .transform(lambda s: s.shift(1).expanding().mean())
    )

    return X


def columnas_predictoras(X):
    """Todo menos identificadores y el objetivo."""
    excluir = {"fecha", "dia_semana", "total", "efectivo", "tarjeta", "delivery", "t"}
    return [c for c in X.columns if c not in excluir]


def matriz(X):
    """Separa (predictoras, objetivo) descartando las filas sin retardo completo."""
    cols = columnas_predictoras(X)
    completo = X.dropna(subset=cols + ["total"]).reset_index(drop=True)
    return completo[cols].to_numpy(dtype=float), completo["total"].to_numpy(dtype=float), completo, cols
