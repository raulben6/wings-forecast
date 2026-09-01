# -*- coding: utf-8 -*-
"""
Figuras del informe.

La paleta es una secuencia categórica validada para daltonismo: los tonos se
asignan en orden fijo por identidad de la serie, nunca por su ranking, y no se
recorren cíclicamente. Un eje por gráfico, nunca dos escalas. Marcas finas,
rejilla discreta, y etiquetas directas solo donde aportan.
"""

import matplotlib
matplotlib.use("Agg")                     # sin ventana: se ejecuta en CI y en scripts
import matplotlib.pyplot as plt
import numpy as np

# --- Paleta -----------------------------------------------------------------
SUPERFICIE = "#fcfcfb"
TEXTO = "#0b0b0b"
TEXTO_SEC = "#52514e"
REJILLA = "#e4e3e0"

SERIE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # azul, naranja, aqua, amarillo

DIAS_ES = {"Monday": "Lun", "Tuesday": "Mar", "Wednesday": "Mié",
           "Thursday": "Jue", "Friday": "Vie", "Saturday": "Sáb"}


def _base(ancho=9, alto=4.2):
    fig, ax = plt.subplots(figsize=(ancho, alto), dpi=150)
    fig.patch.set_facecolor(SUPERFICIE)
    ax.set_facecolor(SUPERFICIE)
    ax.grid(True, color=REJILLA, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(REJILLA)
    ax.tick_params(colors=TEXTO_SEC, labelsize=9, length=0)
    return fig, ax


def _titulo(ax, titulo, subtitulo=None):
    ax.set_title(titulo, color=TEXTO, fontsize=13, fontweight="600",
                 loc="left", pad=18 if subtitulo else 10)
    if subtitulo:
        ax.text(0, 1.02, subtitulo, transform=ax.transAxes,
                color=TEXTO_SEC, fontsize=10, va="bottom")


def _guardar(fig, ruta):
    fig.tight_layout()
    fig.savefig(ruta, facecolor=SUPERFICIE, bbox_inches="tight")
    plt.close(fig)
    return ruta


def serie_temporal(df, ruta, ventana=6):
    """Ventas diarias con la media móvil de una semana operativa encima."""
    fig, ax = _base(10, 4)
    ax.plot(df["fecha"], df["total"], color=SERIE[0], linewidth=1.0,
            alpha=0.55, label="Daily sales", zorder=2)
    ax.plot(df["fecha"], df["total"].rolling(ventana).mean(), color=SERIE[1],
            linewidth=2.0, label=f"{ventana}-day moving average", zorder=3)
    _titulo(ax, "Daily sales, indexed to mean = 100",
            f"{len(df)} trading days · {df['fecha'].min():%b %Y} to {df['fecha'].max():%b %Y}")
    ax.set_ylabel("Sales index", color=TEXTO_SEC, fontsize=10)
    leg = ax.legend(frameon=False, loc="upper left", fontsize=9,
                    labelcolor=TEXTO_SEC, ncol=2)
    leg.set_zorder(5)
    return _guardar(fig, ruta)


def distribucion_dia_semana(df, ruta, orden=None):
    """Caja y bigotes por día: enseña el nivel y la dispersión a la vez."""
    orden = orden or ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    datos = [df.loc[df["dia_semana"] == d, "total"].to_numpy() for d in orden]

    fig, ax = _base(8, 4.2)
    bp = ax.boxplot(datos, patch_artist=True, widths=0.55,
                    medianprops=dict(color=SUPERFICIE, linewidth=2),
                    whiskerprops=dict(color=TEXTO_SEC, linewidth=1),
                    capprops=dict(color=TEXTO_SEC, linewidth=1),
                    flierprops=dict(marker="o", markersize=4,
                                    markerfacecolor=TEXTO_SEC,
                                    markeredgecolor="none", alpha=0.45))
    for caja in bp["boxes"]:
        caja.set(facecolor=SERIE[0], edgecolor="none", alpha=0.9)

    ax.set_xticklabels([DIAS_ES[d] for d in orden])
    medias = [np.mean(d) for d in datos]
    for i, m in enumerate(medias, start=1):
        ax.text(i, m, f"{m:.0f}", ha="center", va="center",
                color=SUPERFICIE, fontsize=9, fontweight="700", zorder=5)
    _titulo(ax, "Sales by day of week",
            "Box shows the spread; the number is the mean. Closed on Sundays.")
    ax.set_ylabel("Sales index", color=TEXTO_SEC, fontsize=10)
    return _guardar(fig, ruta)


def mezcla_canales(df, ruta, ventana=12):
    """Área apilada: cómo se reparte la venta entre canales."""
    fig, ax = _base(10, 3.8)
    suave = df[["efectivo", "tarjeta", "delivery"]].rolling(ventana).mean()
    ax.stackplot(df["fecha"], suave["efectivo"], suave["tarjeta"], suave["delivery"],
                 colors=SERIE[:3], labels=["Cash", "Card", "Delivery"],
                 edgecolor=SUPERFICIE, linewidth=2, zorder=2)
    _titulo(ax, "Revenue mix by channel",
            f"{ventana}-day moving average, smoothed to show the trend")
    ax.set_ylabel("Sales index", color=TEXTO_SEC, fontsize=10)
    ax.legend(frameon=False, loc="upper left", fontsize=9,
              labelcolor=TEXTO_SEC, ncol=3)
    return _guardar(fig, ruta)


def comparacion_modelos(tabla, ruta, referencia_mae=None):
    """
    Barras horizontales de MAE, de mejor a peor.

    Una sola serie, un solo color: el ranking ya lo comunica el orden. La
    referencia se marca con una línea, no repintando barras.
    """
    t = tabla.sort_values("MAE", ascending=False)
    fig, ax = _base(8.5, max(3.2, 0.55 * len(t) + 1.4))

    y = np.arange(len(t))
    ax.barh(y, t["MAE"], height=0.6, color=SERIE[0], zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(t["metodo"], fontsize=10, color=TEXTO)
    ax.grid(axis="y", visible=False)

    for i, v in enumerate(t["MAE"]):
        ax.text(v + max(t["MAE"]) * 0.015, i, f"{v:.1f}", va="center",
                fontsize=9, color=TEXTO_SEC)

    if referencia_mae is not None:
        ax.axvline(referencia_mae, color=SERIE[1], linewidth=2,
                   linestyle="--", zorder=3)
        ax.text(referencia_mae, len(t) - 0.35, "  baseline", color=SERIE[1],
                fontsize=9, va="top", fontweight="600")

    _titulo(ax, "Forecast error by method",
            "Mean absolute error, walk-forward validation. Lower is better.")
    ax.set_xlabel("MAE (sales index points)", color=TEXTO_SEC, fontsize=10)
    return _guardar(fig, ruta)


def error_por_dia(serie, ruta, titulo="Where the model misses"):
    """MAE desglosado por día de la semana."""
    fig, ax = _base(7.5, 3.6)
    etiquetas = [DIAS_ES.get(d, d) for d in serie.index]
    ax.bar(etiquetas, serie["mean"], width=0.6, color=SERIE[0], zorder=2)
    for i, v in enumerate(serie["mean"]):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom",
                fontsize=9, color=TEXTO_SEC)
    _titulo(ax, titulo, "Mean absolute error by day of week")
    ax.set_ylabel("MAE", color=TEXTO_SEC, fontsize=10)
    return _guardar(fig, ruta)


def real_vs_prediccion(resultado, ruta, nombre):
    """Serie real contra la predicción en la ventana de validación."""
    fig, ax = _base(10, 4)
    ax.plot(resultado["fecha"], resultado["real"], color=SERIE[0],
            linewidth=1.6, label="Actual", zorder=3)
    ax.plot(resultado["fecha"], resultado["pred"], color=SERIE[1],
            linewidth=1.6, linestyle="--", label=f"Predicted ({nombre})", zorder=4)
    _titulo(ax, "Actual vs predicted",
            "Walk-forward: every point was forecast using only earlier days")
    ax.set_ylabel("Sales index", color=TEXTO_SEC, fontsize=10)
    ax.legend(frameon=False, loc="upper left", fontsize=9,
              labelcolor=TEXTO_SEC, ncol=2)
    return _guardar(fig, ruta)


def importancia(tabla, ruta, top=12):
    """Importancia por permutación, medida fuera de muestra."""
    t = tabla.head(top).sort_values("importancia")
    fig, ax = _base(8, max(3.2, 0.42 * len(t) + 1.4))
    y = np.arange(len(t))
    ax.barh(y, t["importancia"], height=0.6, color=SERIE[0],
            xerr=t["desviacion"], error_kw=dict(ecolor=TEXTO_SEC, lw=1, capsize=2),
            zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(t["variable"], fontsize=9, color=TEXTO)
    ax.grid(axis="y", visible=False)
    _titulo(ax, "Which variables carry the model",
            "Permutation importance on held-out days, MAE loss")
    ax.set_xlabel("Increase in MAE when shuffled", color=TEXTO_SEC, fontsize=10)
    return _guardar(fig, ruta)


def lluvia_vs_ventas(df, ruta):
    """
    Ventas en días secos frente a días de lluvia.

    Comparación cruda, sin controlar por día de la semana: eso lo hace el
    modelo. Aquí solo se mira si a simple vista hay efecto.
    """
    seco = df.loc[df["lluvia_mm"] == 0, "total"]
    ligera = df.loc[(df["lluvia_mm"] > 0) & (df["lluvia_mm"] < 5), "total"]
    fuerte = df.loc[df["lluvia_mm"] >= 5, "total"]

    grupos = [seco, ligera, fuerte]
    etiquetas = [f"Dry\nn={len(seco)}", f"Light rain\nn={len(ligera)}",
                 f"Rain >= 5mm\nn={len(fuerte)}"]

    fig, ax = _base(7.5, 4)
    bp = ax.boxplot(grupos, patch_artist=True, widths=0.5,
                    medianprops=dict(color=SUPERFICIE, linewidth=2),
                    whiskerprops=dict(color=TEXTO_SEC, linewidth=1),
                    capprops=dict(color=TEXTO_SEC, linewidth=1),
                    flierprops=dict(marker="o", markersize=4,
                                    markerfacecolor=TEXTO_SEC,
                                    markeredgecolor="none", alpha=0.45))
    for caja in bp["boxes"]:
        caja.set(facecolor=SERIE[0], edgecolor="none", alpha=0.9)

    ax.set_xticklabels(etiquetas)
    for i, g in enumerate(grupos, start=1):
        if len(g):
            ax.text(i, g.mean(), f"{g.mean():.0f}", ha="center", va="center",
                    color=SUPERFICIE, fontsize=9, fontweight="700", zorder=5)

    _titulo(ax, "Sales on dry days vs rainy days",
            "Not controlled for weekday; the number is the mean")
    ax.set_ylabel("Sales index", color=TEXTO_SEC, fontsize=10)
    return _guardar(fig, ruta)


def aporte_externos(por_modo, etiquetas_modo, ruta):
    """MAE de cada modelo con y sin las fuentes externas."""
    modos = [m for m in etiquetas_modo if m in por_modo]
    nombres = sorted({n for d in por_modo.values() for n in d})

    fig, ax = _base(8.5, 4)
    ancho = 0.8 / max(len(nombres), 1)
    x = np.arange(len(modos))

    for j, nombre in enumerate(nombres):
        valores = [por_modo[m].get(nombre, np.nan) for m in modos]
        pos = x + j * ancho - 0.4 + ancho / 2
        ax.bar(pos, valores, width=ancho * 0.88, color=SERIE[j % len(SERIE)],
               label=nombre, zorder=2)
        for p, v in zip(pos, valores):
            if not np.isnan(v):
                ax.text(p, v, f"{v:.1f}", ha="center", va="bottom",
                        fontsize=8.5, color=TEXTO_SEC)

    ax.set_xticks(x)
    ax.set_xticklabels([etiquetas_modo[m] for m in modos], fontsize=9)
    ax.set_ylim(0, max(v for d in por_modo.values() for v in d.values()) * 1.18)
    _titulo(ax, "Do holidays and weather help?",
            "Mean absolute error by feature set. Lower is better.")
    ax.set_ylabel("MAE", color=TEXTO_SEC, fontsize=10)
    ax.legend(frameon=False, loc="upper right", fontsize=9,
              labelcolor=TEXTO_SEC, ncol=2)
    return _guardar(fig, ruta)


def desplazamiento_canal(df, ruta, umbral=5.0):
    """
    Cuota del canal de delivery en días secos y lluviosos.

    Es el gráfico que explica por qué el clima no mejora la predicción del
    total: la lluvia no quita demanda, la cambia de canal.
    """
    cuota = (df["delivery"] / df["total"]) * 100
    seco = cuota[df["lluvia_mm"] == 0]
    ligera = cuota[(df["lluvia_mm"] > 0) & (df["lluvia_mm"] < umbral)]
    fuerte = cuota[df["lluvia_mm"] >= umbral]

    grupos = [seco, ligera, fuerte]
    etiquetas = [f"Dry\nn={len(seco)}", f"Light rain\nn={len(ligera)}",
                 f"Rain >= {umbral:g}mm\nn={len(fuerte)}"]

    fig, ax = _base(7.5, 4)
    bp = ax.boxplot(grupos, patch_artist=True, widths=0.5,
                    medianprops=dict(color=SUPERFICIE, linewidth=2),
                    whiskerprops=dict(color=TEXTO_SEC, linewidth=1),
                    capprops=dict(color=TEXTO_SEC, linewidth=1),
                    flierprops=dict(marker="o", markersize=4,
                                    markerfacecolor=TEXTO_SEC,
                                    markeredgecolor="none", alpha=0.45))
    for caja in bp["boxes"]:
        caja.set(facecolor=SERIE[2], edgecolor="none", alpha=0.9)

    ax.set_xticklabels(etiquetas)
    for i, g in enumerate(grupos, start=1):
        if len(g):
            ax.text(i, g.mean(), f"{g.mean():.0f}%", ha="center", va="center",
                    color=SUPERFICIE, fontsize=9, fontweight="700", zorder=5)

    _titulo(ax, "Rain moves demand to delivery",
            "Delivery share of daily revenue. The number is the mean.")
    ax.set_ylabel("Delivery share (%)", color=TEXTO_SEC, fontsize=10)
    return _guardar(fig, ruta)
