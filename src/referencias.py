# -*- coding: utf-8 -*-
"""
Métodos de referencia.

Un modelo de series temporales no vale por su error absoluto, sino por
cuánto mejora a la alternativa trivial. Estas son las alternativas triviales,
y la que hay que batir es `estacional_ingenuo`: en una serie con patrón
semanal fuerte, "lo mismo que el mismo día de la semana pasada" suele ser
sorprendentemente difícil de superar.

Cada método recibe el histórico disponible (estrictamente anterior al día que
se predice) y devuelve un único número.
"""

SEMANA = 6  # aperturas por semana


def ingenuo(historico, fila):
    """El valor de la apertura anterior."""
    return historico["total"].iloc[-1]


def estacional_ingenuo(historico, fila):
    """El mismo día de la semana anterior. Referencia principal."""
    if len(historico) < SEMANA:
        return historico["total"].mean()
    return historico["total"].iloc[-SEMANA]


def media_movil(historico, fila, ventana=SEMANA):
    """Media de la última semana operativa."""
    return historico["total"].tail(ventana).mean()


def media_dia_semana(historico, fila):
    """Media histórica del mismo día de la semana."""
    mismos = historico[historico["dia_semana"] == fila["dia_semana"]]["total"]
    return mismos.mean() if len(mismos) else historico["total"].mean()


REFERENCIAS = {
    "Naive": ingenuo,
    "Seasonal naive": estacional_ingenuo,
    "Moving average (6d)": media_movil,
    "Day-of-week mean": media_dia_semana,
}

PRINCIPAL = "Seasonal naive"
