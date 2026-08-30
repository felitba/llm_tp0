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

# ── Paleta: Okabe-Ito ─────────────────────────────────────────────────────────
# El estándar para series categóricas en publicación científica, elegido porque
# se distingue bajo los tres tipos de daltonismo. No es tab10: tab10 falla, su
# verde y su naranja quedan a ΔE 0.7 bajo protanopia, o sea idénticos para quien
# no distingue el rojo. Estos seis pasan las seis validaciones sobre blanco.
BLUE       = "#0072B2"
ORANGE     = "#E69F00"
GREEN      = "#009E73"
VERMILLION = "#D55E00"
SKY        = "#56B4E9"
PURPLE     = "#CC79A7"

ACCENT   = BLUE        # serie principal: el modelo propuesto
DEEP     = "#004b75"   # variante oscura del acento (anotaciones sobre azul claro)
LIGHT    = "#e0eef6"   # relleno suave: banda de confianza, área bajo la curva
INK      = "#1a1a1a"   # títulos y etiquetas de eje
BODY     = "#404040"   # ticks, texto secundario
MUTED    = "#737373"   # anotaciones y etiquetas menores
BASELINE = "#999999"   # baselines y series de referencia (siempre punteadas)
BORDER   = "#cccccc"   # grilla y spines
SURFACE  = "#ffffff"   # fondo
POS      = GREEN       # mejora
NEG      = VERMILLION  # degradación

# Orden fijo de asignación. El color sigue a la serie, nunca a su puesto en el
# ranking: si un filtro saca una curva, las que quedan conservan su color.
SERIES = [BLUE, ORANGE, GREEN, VERMILLION]
SERIES_EXTENDED = [BLUE, ORANGE, GREEN, VERMILLION, SKY, PURPLE]

# Clases del target: comprado vs. no comprado. El positivo es la clase rara
# (BTR base 0.13), así que se lleva el color con peso y el negativo el gris.
POSITIVE = POS         # bought = 1
NEGATIVE = BASELINE    # bought = 0
POSITIVE_LIGHT = "#d6efe7"
NEGATIVE_LIGHT = "#ededed"

# Splits, cuando hay que distinguir train/val/test en una misma figura.
# Azul y naranja son los dos primeros de la paleta y el par seguro clásico: train
# y validation se dibujan como dos líneas finas superpuestas y cualquier par de
# azules vecinos se lee como un solo color a ese grosor.
SPLIT_COLORS = {"train": BLUE, "validation": ORANGE, "test": GREEN}

DASH = (0, (4, 3))   # patrón único para baselines y líneas de referencia

# Tipografía: la de un paper, no la de una marca. Helvetica y Arial están en
# macOS y Windows; DejaVu Sans es el default de matplotlib y cierra el respaldo
# en Linux, así que la figura sale igual en cualquier máquina del equipo.
FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]

# Cada figura se explica sola: título arriba e hiperparámetros abajo del título.
SHOW_AXES_TITLES = True


def series_colors(count):
	"""Colores para ``count`` series, en el orden en que se dibujan.

	Hasta cuatro respeta SERIES; de ahí en más cae en SERIES_EXTENDED y, si aún
	así faltan, cicla. Pasadas las seis ranuras el color repetido deja de
	identificar: usar ``series_styles`` en ese caso.
	"""
	palette = SERIES if count <= len(SERIES) else SERIES_EXTENDED
	return [palette[index % len(palette)] for index in range(count)]


def series_styles(count):
	"""``(color, linestyle)`` por serie, para cuando son más que la paleta.

	Un séptimo color no existe: sería un tono generado, indistinguible de alguno
	de los seis bajo daltonismo. Lo que se hace es reciclar el color y cambiar el
	trazo, así la identidad la cargan dos canales y no uno.
	"""
	palette = SERIES if count <= len(SERIES) else SERIES_EXTENDED
	strokes = ["-", (0, (5, 2)), (0, (1, 1.6)), (0, (7, 2, 1.5, 2))]
	return [
		(palette[index % len(palette)], strokes[(index // len(palette)) % len(strokes)])
		for index in range(count)
	]


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


def legend_top_left(axes, ncols=None, gap=6.0, subtitle=None):
	"""Encabezado del ax, de arriba a abajo: título, subtítulo, leyenda, datos.

	La leyenda va afuera del área de datos: adentro se apoya sobre las curvas o
	obliga a buscar la única esquina libre, que cambia de figura en figura.

	``subtitle`` es la línea de hiperparámetros: qué corrida es ésta, para que la
	figura se explique sin volver al config.

	Todo se apila en puntos, no en fracción del ax, y esa es la parte que
	importa: el llamador corre ``tight_layout()`` después de esto y el ax cambia
	de alto, así que cualquier distancia medida como fracción deja de valer y la
	leyenda se termina comiendo el título.
	"""
	figure = axes.figure
	handles, labels = axes.get_legend_handles_labels()
	offset = gap
	legend = None

	if handles:
		legend = axes.legend(
			handles, labels,
			loc="lower left",
			bbox_to_anchor=(0.0, 1.0),
			bbox_transform=_points_above(axes, offset),
			ncols=ncols if ncols else len(handles),
			frameon=False,
			borderaxespad=0.0,
		)
		offset += _height_in_points(figure, legend) + gap

	if subtitle:
		text = axes.text(
			0.5, 1.0, subtitle,
			transform=_points_above(axes, offset),
			ha="center", va="bottom",
			fontsize=mpl.rcParams["font.size"] - 1, color=MUTED,
		)
		offset += _height_in_points(figure, text) + gap

	if axes.get_title():
		# ``pad`` también es en puntos, y es el mecanismo que matplotlib respeta
		# cuando reacomoda títulos en cada draw; ``title.set_y()`` se pisa solo.
		axes.set_title(
			axes.get_title(),
			pad=offset,
			fontdict={
				"fontsize": axes.title.get_fontsize(),
				"fontweight": axes.title.get_fontweight(),
				"color": axes.title.get_color(),
			},
		)
	return legend


def _points_above(axes, points):
	"""Transform anclado al borde superior del ax, corrido ``points`` hacia arriba."""
	return axes.transAxes + ScaledTranslation(
		0, points / 72, axes.figure.dpi_scale_trans
	)


def _height_in_points(figure, artist):
	"""Alto real del artista ya compuesto, en puntos.

	Medido y no estimado: una leyenda de dos columnas o una etiqueta de dos
	renglones desarma cualquier número fijo.
	"""
	figure.canvas.draw()
	renderer = figure.canvas.get_renderer()
	return artist.get_window_extent(renderer).height * 72 / figure.dpi


def set_title(axes, title):
	"""Título sólo si SHOW_AXES_TITLES."""
	if SHOW_AXES_TITLES and title:
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