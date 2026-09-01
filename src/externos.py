# -*- coding: utf-8 -*-
"""
Fuentes externas: clima y calendario de festivos.

Qué está disponible cuándo
--------------------------
Esta distinción decide si el proyecto es honesto o no.

**El calendario de festivos se conoce con años de antelación.** Usarlo para
predecir mañana es legítimo en cualquier modo.

**El clima de mañana NO se conoce hoy.** Lo que existe es un pronóstico, que
trae su propio error. Por eso hay dos modos:

- `operativo`   : solo clima rezagado (el de ayer, ya observado). Es lo que de
                  verdad se podría usar en producción sin hacer trampa.
- `explicativo` : clima observado del propio día. No sirve para producción,
                  pero responde a otra pregunta legítima: ¿cuánta varianza
                  explica el clima, como techo de lo que un buen pronóstico
                  podría aportar?

Reportar solo el modo explicativo y llamarlo "predicción" es el error clásico.
Aquí se miden los dos y se etiquetan.
"""

from pathlib import Path

import pandas as pd

RUTA_CLIMA = Path(__file__).resolve().parent.parent / "data" / "clima_diario.csv"

# Umbral de lluvia relevante para el tráfico a pie. 5 mm en un día es
# claramente perceptible; por debajo suele ser llovizna intermitente.
LLUVIA_NOTABLE_MM = 5.0


def cargar_clima(ruta=RUTA_CLIMA):
    clima = pd.read_csv(ruta, parse_dates=["fecha"])
    clima["rango_termico"] = clima["temp_max"] - clima["temp_min"]
    clima["llovio"] = (clima["lluvia_mm"] > 0).astype(int)
    clima["lluvia_notable"] = (clima["lluvia_mm"] >= LLUVIA_NOTABLE_MM).astype(int)
    return clima


def cargar_festivos(fechas):
    """
    Indicadores de festivo para el rango dado, desde el paquete `holidays`.

    Se marca también la víspera y el día siguiente: en un negocio de comida,
    el día de antes de un festivo suele moverse tanto como el festivo mismo.
    """
    import holidays

    anios = sorted({f.year for f in fechas})
    sv = holidays.country_holidays("SV", years=anios)

    df = pd.DataFrame({"fecha": pd.to_datetime(sorted(set(fechas)))})
    dias = df["fecha"].dt.date

    df["es_festivo"] = [int(d in sv) for d in dias]
    df["vispera_festivo"] = [int((d + pd.Timedelta(days=1)).date() in sv) for d in df["fecha"]]
    df["post_festivo"] = [int((d - pd.Timedelta(days=1)).date() in sv) for d in df["fecha"]]
    df["nombre_festivo"] = [sv.get(d, "") for d in dias]
    return df


COLS_CLIMA_BASE = ["temp_max", "temp_min", "rango_termico",
                   "lluvia_mm", "horas_lluvia", "llovio", "lluvia_notable"]

COLS_FESTIVO = ["es_festivo", "vispera_festivo", "post_festivo"]


def unir(df, modo="operativo", ruta_clima=RUTA_CLIMA):
    """
    Añade clima y festivos a la serie de ventas.

    `modo`:
      - "ninguno"     : devuelve la serie intacta (modelo base, sin externos)
      - "operativo"   : festivos + clima de AYER
      - "explicativo" : festivos + clima de HOY (observado, no disponible en
                        producción; sirve para medir el techo explicativo)
    """
    if modo == "ninguno":
        return df.copy(), []

    salida = df.copy()
    columnas = []

    # --- Festivos: legítimos en ambos modos, se conocen de antemano ---------
    fest = cargar_festivos(salida["fecha"])
    salida = salida.merge(fest.drop(columns=["nombre_festivo"]), on="fecha", how="left")
    salida[COLS_FESTIVO] = salida[COLS_FESTIVO].fillna(0).astype(int)
    columnas += COLS_FESTIVO

    # --- Clima --------------------------------------------------------------
    clima = cargar_clima(ruta_clima)
    salida = salida.merge(clima, on="fecha", how="left")

    if modo == "explicativo":
        columnas += COLS_CLIMA_BASE
    elif modo == "operativo":
        # Se desplaza una apertura: la fila de hoy solo ve el clima de ayer.
        for c in COLS_CLIMA_BASE:
            salida[f"{c}_ayer"] = salida[c].shift(1)
            columnas.append(f"{c}_ayer")
        salida = salida.drop(columns=COLS_CLIMA_BASE)
    else:
        raise ValueError(f"modo desconocido: {modo}")

    return salida, columnas


def resumen_festivos(df):
    """Descriptivo: qué pasó los festivos que el negocio abrió."""
    fest = cargar_festivos(df["fecha"])
    unido = df.merge(fest, on="fecha", how="left")
    abiertos = unido[unido["es_festivo"] == 1]

    media_dia = df.groupby("dia_semana")["total"].mean()
    filas = []
    for _, r in abiertos.iterrows():
        esperado = media_dia[r["dia_semana"]]
        filas.append({
            "fecha": r["fecha"].date().isoformat(),
            "festivo": r["nombre_festivo"],
            "dia": r["dia_semana"],
            "ventas": round(r["total"], 1),
            "media del dia": round(esperado, 1),
            "desvio %": round((r["total"] - esperado) / esperado * 100, 1),
        })
    return pd.DataFrame(filas)


def cierres(df):
    """
    Días laborables sin registro: el negocio cerró. Se cruza con el calendario
    para ver cuántos cierres explica un festivo.
    """
    import holidays

    fechas = set(df["fecha"].dt.date)
    ini, fin = min(fechas), max(fechas)
    sv = holidays.country_holidays("SV", years=sorted({ini.year, fin.year}))

    esperados = pd.date_range(ini, fin, freq="D")
    filas = []
    for d in esperados:
        dia = d.date()
        if dia.weekday() == 6 or dia in fechas:      # domingo o día con datos
            continue
        filas.append({"fecha": dia.isoformat(),
                      "dia": d.day_name(),
                      "festivo": sv.get(dia, "")})
    return pd.DataFrame(filas)


# Comisiones documentadas del negocio, públicas en el seed de RestoPos:
# https://github.com/raulben6/restopos  (ComisionPedidosYa 0.24, IVA 0.13,
# FactorTarjeta 1.0275). Se aplican sobre el índice, que al ser una escala
# lineal deja los porcentajes intactos.
COMISION_DELIVERY = 0.24
IVA_SOBRE_COMISION = 0.13
COMISION_TARJETA = 0.0275


def con_margen(df):
    """Añade el ingreso neto de comisiones y el margen resultante."""
    d = df.copy()
    d["neto"] = (d["efectivo"]
                 + d["tarjeta"] * (1 - COMISION_TARJETA)
                 + d["delivery"] * (1 - COMISION_DELIVERY * (1 + IVA_SOBRE_COMISION)))
    d["margen"] = d["neto"] / d["total"]
    d["cuota_delivery"] = d["delivery"] / d["total"]
    return d


def efecto_lluvia(df, umbral=LLUVIA_NOTABLE_MM):
    """
    Compara días secos contra días de lluvia notable.

    La pregunta no es solo si llueve menos gente, sino si la que viene compra
    por otro canal. Devuelve la tabla comparativa y las pruebas t de Welch.
    """
    from scipy import stats

    d = con_margen(df)
    seco = d[d["lluvia_mm"] == 0]
    lluvia = d[d["lluvia_mm"] >= umbral]

    filas = []
    for col, etiqueta in [("total", "Total sales"),
                          ("delivery", "Delivery revenue"),
                          ("cuota_delivery", "Delivery share"),
                          ("neto", "Net of commissions"),
                          ("margen", "Margin")]:
        a, b = seco[col].mean(), lluvia[col].mean()
        t, p = stats.ttest_ind(seco[col].dropna(), lluvia[col].dropna(), equal_var=False)
        filas.append({
            "metric": etiqueta,
            f"dry (n={len(seco)})": round(a, 3),
            f"rain >= {umbral:g}mm (n={len(lluvia)})": round(b, 3),
            "change %": round((b - a) / a * 100, 1),
            "p-value": round(p, 4),
        })
    return pd.DataFrame(filas)
