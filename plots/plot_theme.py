"""Tema de gráficos compartido: un solo lugar para tipografía, grilla y paleta.

Convenciones estándar de figura científica, no de filmina: Helvetica (con
respaldo a Arial y DejaVu Sans, que es lo que hay en Linux), cuerpo de 10 pt,
jerarquía título > eje > ticks, grilla gris fina detrás del dato, y sin adornos.
Todo lo que dibuja — ``plots/`` y ``scripts/eda_*.py`` — pasa por acá.

Uso:
    from plots.plot_theme import apply_theme, ACCENT, BASELINE, SERIES, save
    apply_theme()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color=ACCENT, label="Modelo (AUC = 0.955)")
    ax.plot([0, 1], [0, 1], color=BASELINE, linestyle=DASH, label="Azar")
    save(fig, "output/roc_test.jpg")
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

# ── Paleta ────────────────────────────────────────────────────────────────────
ACCENT   = "#2563eb"   # serie principal: el modelo propuesto
DEEP     = "#1d4ed8"   # variante oscura del acento (anotaciones sobre azul claro)
LIGHT    = "#eef2fd"   # relleno suave: banda de confianza, área bajo la curva
INK      = "#1a1a1a"   # títulos y etiquetas de eje
BODY     = "#404040"   # ticks, texto secundario
MUTED    = "#737373"   # anotaciones y etiquetas menores
BASELINE = "#999999"   # baselines y series de referencia (siempre punteadas)
BORDER   = "#cccccc"   # grilla y spines
SURFACE  = "#ffffff"   # fondo
POS      = "#0f766e"   # mejora
NEG      = "#b42318"   # degradación

# Multi-serie: análogos al azul, en este orden. Nunca más de cuatro.
SERIES = [ACCENT, "#0ea5e9", "#14b8a6", MUTED]

# ── Extensiones fuera del deck ────────────────────────────────────────────────
# La regla de "nunca más de cuatro" vale para una filmina. Las ablaciones tienen
# tantas curvas como experimentos declare el config (hoy cinco), así que las que
# pasan de cuatro salen de acá y no del ciclo de colores por defecto de mpl.
SERIES_EXTENDED = [
	ACCENT,
	"#0ea5e9",   # sky
	"#14b8a6",   # teal
	"#7c3aed",   # violeta
	"#f59e0b",   # ámbar
	"#0f766e",   # verde profundo
	"#e11d48",   # rosa
	MUTED,
]

# Clases del target: comprado vs. no comprado. El positivo es la clase rara
# (BTR base 0.13), así que se lleva el color con peso y el negativo el gris.
POSITIVE = POS         # bought = 1
NEGATIVE = BASELINE    # bought = 0
POSITIVE_LIGHT = "#d7f0ec"
NEGATIVE_LIGHT = "#edeff4"

# Splits, cuando hay que distinguir train/val/test en una misma figura.
SPLIT_COLORS = {"train": ACCENT, "validation": "#0ea5e9", "test": "#14b8a6"}

DASH = (0, (4, 3))   # patrón único para baselines y líneas de referencia

# Tipografía: la de un paper, no la de una marca. Helvetica y Arial están en
# macOS y Windows; DejaVu Sans es el default de matplotlib y cierra el respaldo
# en Linux, así que la figura sale igual en cualquier máquina del equipo.
FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]

# El título vive en la filmina, no en el ax. Poner True para verlos al iterar
# en local sin tener que abrir el deck.
SHOW_AXES_TITLES = False


def series_colors(count):
	"""Colores para ``count`` series, en el orden en que se dibujan.

	Hasta cuatro respeta SERIES (la regla del deck); de ahí en más cae en
	SERIES_EXTENDED y, si aún así faltan, cicla.
	"""
	palette = SERIES if count <= len(SERIES) else SERIES_EXTENDED
	return [palette[index % len(palette)] for index in range(count)]


def apply_theme(font=None, base=10):
	"""Cuerpo de 10 pt a 200 dpi: el tamaño de una figura de paper.

	La escala es la estándar y va toda derivada de ``base``: título un punto
	arriba, ejes en ``base``, ticks y leyenda un punto abajo. Los scripts de EDA
	fijan algunos tamaños a mano (8-13 pt) y encajan en esta escala; con el
	cuerpo en 17 pt que usaba el deck quedaban al revés, con el título más chico
	que la etiqueta del eje.
	"""
	mpl.rcParams.update({
		"figure.facecolor": SURFACE,
		"axes.facecolor": SURFACE,
		"savefig.facecolor": SURFACE,
		"savefig.bbox": "tight",
		"savefig.pad_inches": 0.05,
		"figure.dpi": 200,
		"savefig.dpi": 200,

		"font.family": "sans-serif",
		"font.sans-serif": ([font] if font else []) + FONT_STACK,
		# Helvetica no trae flechas ni símbolos de conjuntos. Los pocos que
		# aparecen en los títulos van como mathtext ($\\rightarrow$, $\\cap$), que
		# los saca de la fuente matemática; "regular" evita que salgan en itálica.
		"mathtext.default": "regular",

		"font.size": base,
		"axes.titlesize": base + 1,
		"axes.labelsize": base,
		"xtick.labelsize": base - 1,
		"ytick.labelsize": base - 1,
		"legend.fontsize": base - 1,
		"figure.titlesize": base + 2,

		"text.color": INK,
		"axes.labelcolor": INK,
		"axes.titlecolor": INK,
		"axes.edgecolor": BORDER,
		"xtick.color": BORDER,
		"ytick.color": BORDER,
		"xtick.labelcolor": BODY,
		"ytick.labelcolor": BODY,

		"axes.spines.top": False,
		"axes.spines.right": False,
		"axes.linewidth": 0.8,
		"xtick.direction": "out",
		"ytick.direction": "out",
		"xtick.major.size": 3.5,
		"ytick.major.size": 3.5,
		"xtick.major.width": 0.8,
		"ytick.major.width": 0.8,
		"xtick.major.pad": 3,
		"ytick.major.pad": 3,

		"axes.grid": True,
		"axes.grid.axis": "y",
		"axes.axisbelow": True,
		"grid.color": BORDER,
		"grid.linewidth": 0.6,
		"grid.alpha": 0.7,

		"lines.linewidth": 1.5,
		"lines.markersize": 4,
		"patch.linewidth": 0,

		# Marco tenue: la leyenda va adentro del área de datos y sin caja se
		# mezcla con la grilla. Esquinas rectas, no redondeadas.
		"legend.frameon": True,
		"legend.fancybox": False,
		"legend.framealpha": 0.9,
		"legend.facecolor": SURFACE,
		"legend.edgecolor": BORDER,
		"legend.borderpad": 0.5,
		"legend.handlelength": 1.8,
		"legend.labelcolor": INK,

		"axes.prop_cycle": mpl.cycler(color=SERIES),
	})


def legend_top_left(axes, ncols=None, gap=6.0):
	"""Leyenda afuera del área de datos, arriba y alineada al borde izquierdo.

	Adentro del cuadro la leyenda o se apoya sobre las curvas o obliga a buscar
	la única esquina libre, que cambia de figura en figura. Arriba a la izquierda
	está siempre en el mismo lugar y se lee antes que el gráfico, que es el orden
	en que uno mira. El marco sobra porque ya no hay nada de qué separarla.

	``gap`` va en puntos, no en fracción del ax, y esa es la parte que importa:
	el llamador corre ``tight_layout()`` después de esto y el ax cambia de alto,
	así que cualquier distancia medida como fracción deja de valer y la leyenda
	se termina comiendo el título. En puntos, leyenda y título se mueven con el
	borde del ax y la separación queda igual.
	"""
	handles, labels = axes.get_legend_handles_labels()
	if not handles:
		return None
	figure = axes.figure
	anchor = axes.transAxes + ScaledTranslation(0, gap / 72, figure.dpi_scale_trans)
	legend = axes.legend(
		handles, labels,
		loc="lower left",
		bbox_to_anchor=(0.0, 1.0),
		bbox_transform=anchor,
		ncols=ncols if ncols else len(handles),
		frameon=False,
		borderaxespad=0.0,
	)
	if axes.get_title():
		# El título va arriba de todo. ``pad`` también es en puntos, y es el
		# mecanismo que matplotlib respeta cuando reacomoda títulos en cada draw.
		figure.canvas.draw()
		renderer = figure.canvas.get_renderer()
		height = legend.get_window_extent(renderer).height * 72 / figure.dpi
		axes.set_title(
			axes.get_title(),
			pad=2 * gap + height,
			fontdict={
				"fontsize": axes.title.get_fontsize(),
				"fontweight": axes.title.get_fontweight(),
				"color": axes.title.get_color(),
			},
		)
	return legend


def set_title(axes, title):
	"""Título sólo si SHOW_AXES_TITLES; la versión del deck va sin él."""
	if SHOW_AXES_TITLES:
		axes.set_title(title)


def save(fig, path):
	"""El título vive en la filmina, no en el ax: no usar ax.set_title()."""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	# El default de jpg submuestrea el croma y deja halo alrededor de una línea
	# fina sobre blanco. Con subsampling=0 el trazo sale limpio.
	pil_kwargs = (
		{"quality": 95, "subsampling": 0} if path.suffix.lower() in (".jpg", ".jpeg") else {}
	)
	fig.savefig(path, pil_kwargs=pil_kwargs)
	plt.close(fig)
	return path
