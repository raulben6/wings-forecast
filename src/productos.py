# -*- coding: utf-8 -*-
"""
Demanda por producto.

Predecir el total del día sirve para saber cuánto se va a facturar. Para
decidir **qué comprar** hace falta bajar al producto: un pronóstico de
ingresos no dice cuántas libras de alas pedir.

Este módulo consume la salida de `scripts/exportar_pos.py`.

Sobre la suficiencia de datos
-----------------------------
Bajar al producto multiplica el número de series y divide la muestra de cada
una. Con pocos meses de POS, la mayoría de productos tendrán una decena de
observaciones, y un modelo entrenado ahí produce números que parecen
resultados sin serlo.

Por eso `comprobar_suficiencia()` es una puerta explícita, no un aviso: el
análisis reporta descriptivos siempre, pero **no entrega métricas de modelo
mientras no haya historial**. Es la misma decisión que llevó a no estimar un
efecto de festivo con tres observaciones.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RUTA_DEFECTO = Path(__file__).resolve().parent.parent / "data" / "ventas_producto.csv"

# Umbrales mínimos para que un modelo por producto signifique algo.
DIAS_MINIMOS = 60
OBSERVACIONES_MINIMAS_POR_PRODUCTO = 30


def existe(ruta=RUTA_DEFECTO):
    return Path(ruta).exists()


def cargar(ruta=RUTA_DEFECTO):
    df = pd.read_csv(ruta, parse_dates=["fecha"])
    df["dia_semana"] = df["fecha"].dt.day_name()
    return df.sort_values(["fecha", "producto"]).reset_index(drop=True)


def resumen(df):
    return {
        "filas": len(df),
        "productos": int(df["producto"].nunique()),
        "categorias": int(df["categoria"].nunique()),
        "dias": int(df["fecha"].nunique()),
        "desde": df["fecha"].min().date().isoformat(),
        "hasta": df["fecha"].max().date().isoformat(),
        "unidades": round(float(df["unidades"].sum()), 2),
        "canales": sorted(df["canal"].dropna().unique().tolist()),
    }


def comprobar_suficiencia(df, dias_minimos=DIAS_MINIMOS,
                          obs_minimas=OBSERVACIONES_MINIMAS_POR_PRODUCTO):
    """
    ¿Hay base para modelar? Devuelve el veredicto y el motivo.

    No lanza excepción: el análisis descriptivo sigue teniendo sentido con
    pocos datos. Lo que se bloquea es publicar métricas de predicción.
    """
    dias = int(df["fecha"].nunique())
    por_producto = df.groupby("producto")["fecha"].nunique()
    modelables = por_producto[por_producto >= obs_minimas]

    return {
        "suficiente": bool(dias >= dias_minimos and len(modelables) > 0),
        "dias": dias,
        "dias_minimos": dias_minimos,
        "productos_modelables": int(len(modelables)),
        "productos_totales": int(len(por_producto)),
        "motivo": (
            "ok" if dias >= dias_minimos and len(modelables) > 0
            else f"solo {dias} días de historial (mínimo {dias_minimos}) y "
                 f"{len(modelables)} de {len(por_producto)} productos alcanzan "
                 f"{obs_minimas} días con venta"
        ),
    }


def pareto(df, metrica="importe"):
    """
    Análisis ABC: qué productos sostienen la facturación.

    En hostelería la regla suele cumplirse con fuerza, y decide dónde importa
    de verdad acertar el pronóstico. Errar en el producto que hace el 30% de
    la caja no es comparable a errar en uno que hace el 1%.
    """
    g = (df.groupby("producto")[metrica].sum()
           .sort_values(ascending=False)
           .to_frame(metrica))
    total = g[metrica].sum()
    g["cuota %"] = (g[metrica] / total * 100).round(2)
    g["acumulado %"] = g["cuota %"].cumsum().round(2)
    # A: hasta el 80% acumulado. B: hasta el 95%. C: la cola.
    g["clase"] = np.where(g["acumulado %"] <= 80, "A",
                  np.where(g["acumulado %"] <= 95, "B", "C"))
    return g.round(2)


def patron_semanal(df, metrica="unidades", top=8):
    """Perfil por día de la semana de los productos principales."""
    orden = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    principales = (df.groupby("producto")[metrica].sum()
                     .sort_values(ascending=False).head(top).index)
    sub = df[df["producto"].isin(principales)]
    tabla = (sub.pivot_table(index="producto", columns="dia_semana",
                             values=metrica, aggfunc="mean")
                .reindex(columns=[d for d in orden if d in sub["dia_semana"].values]))
    return tabla.round(2)


def mezcla_por_canal(df, metrica="importe"):
    """Cómo cambia el mix de productos entre local y delivery."""
    t = df.pivot_table(index="producto", columns="canal",
                       values=metrica, aggfunc="sum", fill_value=0)
    for c in t.columns:
        t[f"{c} %"] = (t[c] / t[c].sum() * 100).round(2)
    return t.round(2)


def serie_producto(df, producto, metrica="unidades"):
    """
    Serie diaria de un producto, con los días sin venta como cero.

    Un día sin línea de venta significa cero unidades, no un hueco. Dejarlo
    como ausente sesgaría al alza cualquier media.
    """
    sub = df[df["producto"] == producto]
    if sub.empty:
        raise KeyError(f"producto sin ventas: {producto}")

    diario = sub.groupby("fecha")[metrica].sum()
    calendario = pd.date_range(df["fecha"].min(), df["fecha"].max(), freq="D")
    completo = diario.reindex(calendario, fill_value=0.0)

    salida = completo.rename_axis("fecha").rename("total").reset_index()
    salida["dia_semana"] = salida["fecha"].dt.day_name()
    # El negocio no abre en domingo: esos ceros son cierre, no demanda nula.
    salida = salida[salida["dia_semana"] != "Sunday"].reset_index(drop=True)
    salida["t"] = range(len(salida))
    return salida
