# -*- coding: utf-8 -*-
"""
Validación walk-forward (origen móvil) y métricas.

Por qué no un train_test_split aleatorio
---------------------------------------
Es el error más frecuente en series temporales. Repartir los días al azar
mete días futuros en el entrenamiento y días pasados en la prueba, así que
el modelo predice el martes habiendo visto el miércoles. El error de test
sale precioso y el modelo no sirve para nada, porque en producción el futuro
no está disponible.

Aquí cada predicción del día t usa exclusivamente datos hasta t-1, y el
modelo se reentrena en cada paso con la ventana expandida. Es más lento y es
la única forma de que el número reportado signifique algo.
"""

import numpy as np
import pandas as pd


def walk_forward(df, metodo, inicio=None):
    """
    Predice un paso adelante para cada día desde `inicio` hasta el final.

    `metodo` es un invocable (historico_df, fila_objetivo) -> float.
    Devuelve un DataFrame con fecha, día, valor real y predicción.
    """
    if inicio is None:
        # Se reserva el primer tercio como histórico mínimo para arrancar.
        inicio = max(30, len(df) // 3)

    filas = []
    for i in range(inicio, len(df)):
        historico = df.iloc[:i]          # estrictamente anterior a i
        objetivo = df.iloc[i]
        filas.append({
            "fecha": objetivo["fecha"],
            "dia_semana": objetivo["dia_semana"],
            "real": objetivo["total"],
            "pred": float(metodo(historico, objetivo)),
        })
    return pd.DataFrame(filas)


def metricas(real, pred):
    """MAE, RMSE y MAPE. El MAPE es interpretable pese al indexado."""
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = real - pred
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": float(np.mean(np.abs(err / real)) * 100),
    }


def tabla_comparativa(resultados, referencia):
    """
    Ordena los métodos y añade la mejora sobre la referencia.

    `resultados` es {nombre: DataFrame de walk_forward}.
    """
    filas = []
    base = metricas(resultados[referencia]["real"], resultados[referencia]["pred"])

    for nombre, res in resultados.items():
        m = metricas(res["real"], res["pred"])
        filas.append({
            "metodo": nombre,
            "MAE": round(m["MAE"], 2),
            "RMSE": round(m["RMSE"], 2),
            "MAPE %": round(m["MAPE"], 1),
            # Positivo significa mejor que la referencia.
            "mejora MAE %": round((base["MAE"] - m["MAE"]) / base["MAE"] * 100, 1),
        })

    return pd.DataFrame(filas).sort_values("MAE").reset_index(drop=True)


def error_por_dia(resultado):
    """Dónde falla el método: MAE desglosado por día de la semana."""
    r = resultado.copy()
    r["err_abs"] = (r["real"] - r["pred"]).abs()
    orden = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    g = r.groupby("dia_semana")["err_abs"].agg(["count", "mean"]).round(2)
    return g.reindex([d for d in orden if d in g.index])


def intervalo_bootstrap(real, pred, n=2000, semilla=42):
    """
    Intervalo de confianza del 95% para el MAE, por bootstrap.

    Con menos de 250 observaciones, un MAE puntual sin intervalo transmite
    una precisión que no existe.
    """
    rng = np.random.default_rng(semilla)
    err = np.abs(np.asarray(real, dtype=float) - np.asarray(pred, dtype=float))
    muestras = [rng.choice(err, size=len(err), replace=True).mean() for _ in range(n)]
    lo, hi = np.percentile(muestras, [2.5, 97.5])
    return float(lo), float(hi)
