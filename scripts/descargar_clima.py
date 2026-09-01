#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descarga el histórico meteorológico diario de San Salvador.

Fuente: Open-Meteo Historical Weather API (reanálisis ERA5). Es gratuita,
no pide clave y permite uso no comercial con atribución.
https://open-meteo.com/en/docs/historical-weather-api

La salida, `data/clima_diario.csv`, se versiona en el repositorio. Así el
análisis se reproduce sin red y sin depender de que un servicio externo siga
respondiendo dentro de dos años.

    python scripts/descargar_clima.py
    python scripts/descargar_clima.py --desde 2025-10-01 --hasta 2026-06-30
"""

import argparse
import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path

# San Salvador, El Salvador.
LATITUD = 13.6929
LONGITUD = -89.2182
ZONA = "America/El_Salvador"

API = "https://archive-api.open-meteo.com/v1/archive"

VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "precipitation_hours",
]

# Nombres en español, coherentes con el resto del proyecto.
RENOMBRAR = {
    "time": "fecha",
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "precipitation_sum": "lluvia_mm",
    "precipitation_hours": "horas_lluvia",
}


def descargar(desde, hasta):
    consulta = urllib.parse.urlencode({
        "latitude": LATITUD,
        "longitude": LONGITUD,
        "start_date": desde,
        "end_date": hasta,
        "daily": ",".join(VARIABLES),
        "timezone": ZONA,
    })
    with urllib.request.urlopen(f"{API}?{consulta}", timeout=60) as r:
        datos = json.loads(r.read().decode("utf-8"))

    diario = datos["daily"]
    filas = []
    for i, fecha in enumerate(diario["time"]):
        fila = {"fecha": fecha}
        for var in VARIABLES:
            valor = diario[var][i]
            fila[RENOMBRAR[var]] = "" if valor is None else valor
        filas.append(fila)
    return filas


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--desde", default="2025-10-01")
    ap.add_argument("--hasta", default="2026-06-30")
    ap.add_argument("--salida", default="data/clima_diario.csv")
    args = ap.parse_args()

    filas = descargar(args.desde, args.hasta)
    destino = Path(args.salida)
    destino.parent.mkdir(parents=True, exist_ok=True)

    with destino.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["fecha", "temp_max", "temp_min",
                                           "lluvia_mm", "horas_lluvia"])
        w.writeheader()
        w.writerows(filas)

    lluvia = [float(f["lluvia_mm"]) for f in filas if f["lluvia_mm"] != ""]
    con_lluvia = sum(1 for v in lluvia if v > 0)
    print(f"{len(filas)} días escritos en {destino}")
    print(f"días con lluvia: {con_lluvia} ({con_lluvia / len(lluvia):.0%})")
    print(f"lluvia máxima: {max(lluvia):.1f} mm")


if __name__ == "__main__":
    main()
