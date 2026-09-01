# -*- coding: utf-8 -*-
"""
Modelos, envueltos para encajar en la validación walk-forward.

Las variables se calculan una sola vez sobre la serie completa, lo cual es
seguro porque todas son retardos o medias cerradas en t-1: la fila del día t
no contiene ni un dato del día t. El reentrenamiento sí ocurre en cada paso,
usando solo las filas anteriores al día que se predice.
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class ModeloTemporal:
    """Adapta un estimador de scikit-learn a la interfaz (historico, fila)."""

    def __init__(self, estimador, tabla, columnas, minimo_entrenamiento=40):
        self.estimador = estimador
        # Indexado por fecha para localizar filas sin ambigüedad.
        self.tabla = tabla.set_index("fecha")
        self.columnas = columnas
        self.minimo = minimo_entrenamiento

    def __call__(self, historico, fila):
        fecha = fila["fecha"]
        if fecha not in self.tabla.index:
            # Sin retardos completos todavía: se cae a la media del histórico.
            return historico["total"].mean()

        previas = self.tabla.loc[self.tabla.index < fecha]
        if len(previas) < self.minimo:
            return historico["total"].mean()

        X = previas[self.columnas].to_numpy(dtype=float)
        y = previas["total"].to_numpy(dtype=float)
        x_obj = self.tabla.loc[[fecha], self.columnas].to_numpy(dtype=float)

        modelo = self._clonar()
        modelo.fit(X, y)
        return float(modelo.predict(x_obj)[0])

    def _clonar(self):
        from sklearn.base import clone
        return clone(self.estimador)


def construir(tabla, columnas, semilla=42):
    """Los modelos a comparar contra las referencias."""
    return {
        "Ridge + calendar": ModeloTemporal(
            make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            tabla, columnas,
        ),
        "Gradient boosting": ModeloTemporal(
            HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.06,
                max_depth=3,           # serie corta: árboles poco profundos
                min_samples_leaf=8,
                l2_regularization=1.0,
                random_state=semilla,
            ),
            tabla, columnas,
        ),
    }


def importancia_por_permutacion(tabla, columnas, semilla=42, repeticiones=20):
    """
    Qué variables sostienen realmente al modelo.

    Se entrena con el 80% inicial y se permuta sobre el 20% final, para que la
    importancia se mida sobre datos que el modelo no vio.
    """
    from sklearn.inspection import permutation_importance

    corte = int(len(tabla) * 0.8)
    entrena, prueba = tabla.iloc[:corte], tabla.iloc[corte:]

    modelo = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.06, max_depth=3,
        min_samples_leaf=8, l2_regularization=1.0, random_state=semilla,
    )
    modelo.fit(entrena[columnas].to_numpy(dtype=float),
               entrena["total"].to_numpy(dtype=float))

    r = permutation_importance(
        modelo,
        prueba[columnas].to_numpy(dtype=float),
        prueba["total"].to_numpy(dtype=float),
        n_repeats=repeticiones, random_state=semilla,
        scoring="neg_mean_absolute_error",
    )
    return (pd.DataFrame({"variable": columnas,
                          "importancia": r.importances_mean,
                          "desviacion": r.importances_std})
            .sort_values("importancia", ascending=False)
            .reset_index(drop=True))
