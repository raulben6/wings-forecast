#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exporta la demanda por producto y día desde el POS.

Lee de la base de datos de RestoPos (https://github.com/raulben6/restopos),
que es el sistema de punto de venta que opera el negocio.

Garantías de este script
------------------------
- **Solo lectura.** Ejecuta una única sentencia `SELECT`. No hay `INSERT`,
  `UPDATE`, `DELETE`, `ALTER` ni `CREATE` en ninguna parte del fichero.
- **Agrega en origen.** La base de datos devuelve producto × día ya sumado.
  Nunca se descarga la tabla de órdenes fila a fila.
- **No toca datos personales.** No lee `users`, ni `operator_id`, ni las notas
  de cliente, ni el motivo de cancelación, ni el método de pago individual.
- **Indexa los importes**, igual que `preparar_datos.py`, para no publicar la
  facturación real de un negocio identificable.

La conexión se toma de `DATABASE_PUBLIC_URL` o `DATABASE_URL`. El valor nunca
se imprime.

    railway run --service Postgres python scripts/exportar_pos.py
    python scripts/exportar_pos.py --sin-indexar --salida /ruta/local.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# Una sola consulta, agregada, sin datos personales.
CONSULTA = """
SELECT
    (o.created_at AT TIME ZONE 'America/El_Salvador')::date  AS fecha,
    oi.combo_name                                            AS producto,
    COALESCE(c.category, 'sin categoria')                    AS categoria,
    o.channel                                                AS canal,
    SUM(oi.qty)::int                                         AS unidades,
    SUM(oi.qty * (oi.combo_price + COALESCE(v.extra, 0)))::numeric AS importe
FROM order_items oi
JOIN orders o        ON o.id = oi.order_id
LEFT JOIN combos c   ON c.id = oi.combo_id
LEFT JOIN (
    SELECT item_id, SUM(extra_price) AS extra
    FROM order_item_variants
    GROUP BY item_id
) v ON v.item_id = oi.id
WHERE o.status IS DISTINCT FROM 'Cancelado'
GROUP BY 1, 2, 3, 4
ORDER BY 1, 2
"""

CABECERA = ["fecha", "producto", "categoria", "canal", "unidades", "importe"]


def obtener_filas(url):
    import psycopg

    with psycopg.connect(url, connect_timeout=30) as conexion:
        # Sesión de solo lectura: si algo intentara escribir, la base lo rechaza.
        conexion.read_only = True
        with conexion.cursor() as cur:
            cur.execute(CONSULTA)
            columnas = [d.name for d in cur.description]
            return [dict(zip(columnas, fila)) for fila in cur.fetchall()]


def indexar(filas):
    """
    Escala unidades e importes para no revelar el volumen ni la facturación.

    Cada magnitud se divide por su media diaria y se multiplica por 100. Es
    lineal, así que participaciones, estacionalidad y errores porcentuales
    quedan intactos. Los divisores no se publican.
    """
    dias = len({f["fecha"] for f in filas})
    if not dias:
        return filas

    f_unid = sum(f["unidades"] for f in filas) / dias / 100.0
    f_imp = sum(float(f["importe"]) for f in filas) / dias / 100.0

    salida = []
    for f in filas:
        salida.append({
            **f,
            "unidades": round(f["unidades"] / f_unid, 3) if f_unid else 0.0,
            "importe": round(float(f["importe"]) / f_imp, 3) if f_imp else 0.0,
        })
    return salida


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salida", default="data/ventas_producto.csv")
    ap.add_argument("--sin-indexar", action="store_true",
                    help="conserva unidades e importes reales (uso local, nunca al repo)")
    ap.add_argument("--minimo-dias", type=int, default=60,
                    help="avisa si el histórico es demasiado corto para modelar")
    args = ap.parse_args()

    url = os.environ.get("DATABASE_PUBLIC_URL") or os.environ.get("DATABASE_URL")
    if not url:
        sys.exit("No hay DATABASE_PUBLIC_URL ni DATABASE_URL en el entorno.\n"
                 "Sugerencia: railway run --service Postgres python scripts/exportar_pos.py")

    filas = obtener_filas(url)
    if not filas:
        sys.exit("El POS no devolvió ninguna línea de venta.")

    dias = len({f["fecha"] for f in filas})
    productos = len({f["producto"] for f in filas})
    unidades = sum(f["unidades"] for f in filas)

    print(f"{len(filas)} combinaciones producto/día")
    print(f"{productos} productos distintos · {dias} días con ventas · {unidades} unidades")
    print(f"rango: {min(f['fecha'] for f in filas)} a {max(f['fecha'] for f in filas)}")

    if dias < args.minimo_dias:
        print(f"\n  AVISO: {dias} días de historial. Por debajo de ~{args.minimo_dias} no hay "
              "base para modelar demanda por producto; el análisis lo indicará.")

    datos = filas if args.sin_indexar else indexar(filas)

    destino = Path(args.salida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CABECERA)
        w.writeheader()
        w.writerows(datos)

    print(f"\nescrito en {destino}"
          + ("  (SIN indexar: no lo subas al repositorio)" if args.sin_indexar else ""))


if __name__ == "__main__":
    main()
