# -*- coding: utf-8 -*-
"""Carga y limpieza de la serie diaria de ventas."""

from pathlib import Path

import pandas as pd

RUTA_DEFECTO = Path(__file__).resolve().parent.parent / "data" / "ventas_diarias.csv"

# El negocio abre de lunes a sábado. En 229 días solo hay 2 domingos, que son
# aperturas excepcionales: no hay muestra para estimar un efecto de domingo, y
# dejarlos dentro mete ruido en el patrón semanal. Se excluyen y se documenta.
DIAS_OPERATIVOS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def cargar(ruta=RUTA_DEFECTO, excluir_domingos=True):
    """Devuelve la serie ordenada por fecha, con índice temporal."""
    df = pd.read_csv(ruta, parse_dates=["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)

    df["dia_semana"] = df["fecha"].dt.day_name()
    if excluir_domingos:
        df = df[df["dia_semana"].isin(DIAS_OPERATIVOS)].reset_index(drop=True)

    # Posición dentro de la serie de días abiertos. Es el reloj correcto para
    # los retardos: "hace una semana" son 6 aperturas atrás, no 7 días de
    # calendario, porque el domingo no existe en esta serie.
    df["t"] = range(len(df))
    return df


def resumen(df):
    """Cifras descriptivas para el informe."""
    total = df["total"]
    return {
        "observaciones": len(df),
        "desde": df["fecha"].min().date().isoformat(),
        "hasta": df["fecha"].max().date().isoformat(),
        "media": round(total.mean(), 2),
        "mediana": round(total.median(), 2),
        "desviacion": round(total.std(), 2),
        "coef_variacion": round(total.std() / total.mean(), 4),
        "minimo": round(total.min(), 2),
        "maximo": round(total.max(), 2),
        "mezcla_canales": {
            canal: round(df[canal].sum() / total.sum(), 4)
            for canal in ("efectivo", "tarjeta", "delivery")
        },
    }


def por_dia_semana(df):
    """Media, mediana y dispersión por día de la semana."""
    g = df.groupby("dia_semana")["total"].agg(["count", "mean", "median", "std"])
    return g.reindex(DIAS_OPERATIVOS).round(2)
