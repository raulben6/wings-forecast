#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepara el dataset publicable a partir del Excel operativo del negocio.

Este script se ejecuta UNA VEZ, en local, contra un fichero que no forma
parte del repositorio. Su salida, `data/ventas_diarias.csv`, sí se publica.

Qué hace con las cifras
-----------------------
Las ventas son ingresos reales de un negocio identificable, así que no se
publican en su moneda. Se dividen por la media de la serie y se multiplican
por 100, de modo que la media pasa a valer 100 "unidades de venta".

Es una transformación lineal, y eso importa: la estacionalidad, la varianza
relativa, la mezcla de canales, las correlaciones y todas las métricas
porcentuales (MAPE, mejora sobre la referencia) quedan idénticas. Lo único
que se pierde es la escala absoluta, que es justamente lo que hay que
proteger. El factor divisor no se publica.

Uso
---
    python scripts/preparar_datos.py --excel "ruta/al/Wings Ingresos y Salidas.xlsx"
"""

import argparse
import csv
import datetime
from pathlib import Path

import openpyxl

# Índices de columna en la hoja original.
COL_FECHA = 1
COL_EFECTIVO = 2
COL_TARJETA = 3
COL_DELIVERY = 4


def _num(v):
    """El Excel deja celdas vacías donde no hubo venta por ese canal."""
    return float(v) if isinstance(v, (int, float)) else 0.0


def leer_excel(ruta, hoja="Data_cruda"):
    wb = openpyxl.load_workbook(ruta, data_only=True)
    ws = wb[hoja]
    filas = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        fecha = r[COL_FECHA]
        if not isinstance(fecha, datetime.datetime):
            continue
        filas.append({
            "fecha": fecha.date(),
            "efectivo": _num(r[COL_EFECTIVO]),
            "tarjeta": _num(r[COL_TARJETA]),
            "delivery": _num(r[COL_DELIVERY]),
        })
    filas.sort(key=lambda f: f["fecha"])
    return filas


def indexar(filas):
    """Escala los tres canales por un único factor: la media del total."""
    totales = [f["efectivo"] + f["tarjeta"] + f["delivery"] for f in filas]
    factor = sum(totales) / len(totales) / 100.0

    salida = []
    for f, t in zip(filas, totales):
        salida.append({
            "fecha": f["fecha"].isoformat(),
            "dia_semana": f["fecha"].strftime("%A"),
            "efectivo": round(f["efectivo"] / factor, 2),
            "tarjeta": round(f["tarjeta"] / factor, 2),
            "delivery": round(f["delivery"] / factor, 2),
            "total": round(t / factor, 2),
        })
    return salida


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--excel", required=True, help="Excel operativo (no versionado)")
    ap.add_argument("--salida", default="data/ventas_diarias.csv")
    args = ap.parse_args()

    filas = leer_excel(args.excel)
    if not filas:
        raise SystemExit("No se encontró ninguna fila con fecha válida.")

    datos = indexar(filas)
    destino = Path(args.salida)
    destino.parent.mkdir(parents=True, exist_ok=True)

    with destino.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(datos[0].keys()))
        w.writeheader()
        w.writerows(datos)

    print(f"{len(datos)} días escritos en {destino}")
    print(f"rango: {datos[0]['fecha']} a {datos[-1]['fecha']}")
    print("media del total (por construcción): "
          f"{sum(d['total'] for d in datos) / len(datos):.2f}")


if __name__ == "__main__":
    main()
