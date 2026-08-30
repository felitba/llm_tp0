"""Figuras del paso 2 del guion: características y calidad de los datos.

Produce las tres figuras que le faltaban a la sección "definición del problema":

    13_valores_posibles.jpg   cardinalidad de las categóricas y rango de las numéricas
    14_distribuciones.jpg     distribución del target y de las dos numéricas que usa el modelo
    15_baseline_title_tag.jpg qué agrega el Transformer sobre una tabla de lookup

Además imprime por consola cada número que se cita en `FILMINAS.md` F2 y F4.

Sobre pandas: `scripts/eda_columns.py` usa la stdlib a propósito, porque es el script que
denuncia el bug de `"None"` → `NaN` y sería absurdo reintroducirlo ahí. Acá sí usamos
pandas, pero con `keep_default_na=False`, que es la misma lectura que hace el pipeline
en `dataset/preprocess_dataset.get_raw_dataset`.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# `python scripts/eda_dataset.py` deja scripts/ en el path, no la raíz del repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plots.plot_theme import (  # noqa: E402
	ACCENT,
	BASELINE,
	BODY,
	MUTED,
	NEGATIVE,
	POSITIVE,
	SERIES,
	apply_theme,
	save,
	set_title,
)

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset" / "supermarket_products.csv"
OUTPUT_DIR = ROOT / "output" / "eda"
EXPERIMENTS = ROOT / "output" / "experiments"

CATEGORICAS = [
	"ingredients", "package_size", "title_tag", "brand",
	"category", "country_of_origin", "allergens", "unit_of_measure", "storage_type",
]
NUMERICAS = ["price", "nutrition_score", "net_weight_oz", "filter_price_max", "filter_price_min"]

# Las dos que el modelo efectivamente recibe como tokens numéricos.
NUMERICAS_DEL_MODELO = ["price", "nutrition_score"]


def cargar() -> pd.DataFrame:
	"""Misma lectura que el pipeline: 'None' es una categoría de allergens, no un nulo.

	`title_tag` y `product_name` no están en el CSV: los deriva `process_title_column`.
	Lo importamos en vez de reimplementarlo para que la cardinalidad que reporta esta
	figura sea exactamente la que ve el modelo.
	"""
	from dataset.preprocess_dataset import process_title_column

	return process_title_column(pd.read_csv(DATASET, keep_default_na=False))


# ── Figura 13: valores posibles ───────────────────────────────────────────────

def figura_valores_posibles(df: pd.DataFrame) -> Path:
	cardinalidades = {columna: df[columna].nunique() for columna in CATEGORICAS}
	figura, (izq, der) = plt.subplots(1, 2, figsize=(15, 6.5))

	nombres = list(cardinalidades)
	valores = [cardinalidades[n] for n in nombres]
	# title_tag en acento: es la columna que después explica casi todo el target.
	colores = [ACCENT if n == "title_tag" else BASELINE for n in nombres]
	barras = izq.barh(nombres, valores, color=colores, height=0.62)
	izq.bar_label(barras, padding=6, color=BODY, fontsize=14)
	izq.set_xlim(0, max(valores) * 1.18)
	izq.set_xlabel("valores distintos")
	izq.grid(axis="x")
	izq.grid(axis="y", visible=False)
	set_title(izq, "Cardinalidad de las categóricas")

	# Cada numérica normalizada a su propio rango: la barra completa es el rango y
	# el punto marca dónde cae la mediana dentro de él. Las escalas reales difieren
	# ~100x, así que un eje compartido no diría nada.
	der.grid(axis="x")
	der.grid(axis="y", visible=False)
	for fila, columna in enumerate(NUMERICAS):
		serie = pd.to_numeric(df[columna])
		minimo, maximo, mediana = serie.min(), serie.max(), serie.median()
		posicion = (mediana - minimo) / (maximo - minimo)
		destacada = columna in NUMERICAS_DEL_MODELO
		color = ACCENT if destacada else BASELINE
		der.barh(fila, 1.0, color=color, alpha=0.22 if destacada else 0.35, height=0.5)
		der.plot([posicion], [fila], "o", color=color, markersize=11, zorder=3)
		der.text(-0.03, fila, f"{minimo:,.2f}", ha="right", va="center", color=MUTED, fontsize=13)
		der.text(1.03, fila, f"{maximo:,.2f}", ha="left", va="center", color=MUTED, fontsize=13)
		der.text(posicion, fila + 0.34, f"med {mediana:,.2f}", ha="center", va="bottom",
		         color=BODY, fontsize=12)
	der.set_yticks(range(len(NUMERICAS)))
	der.set_yticklabels(NUMERICAS)
	der.set_xticks([])
	der.set_xlim(-0.22, 1.22)
	der.set_ylim(-0.7, len(NUMERICAS) - 0.3)
	der.invert_yaxis()
	der.set_xlabel("rango propio de cada columna (mín $\\rightarrow$ máx)")
	set_title(der, "Rango de las numéricas")

	figura.tight_layout()
	return save(figura, OUTPUT_DIR / "13_valores_posibles.jpg")


# ── Figura 14: distribuciones ─────────────────────────────────────────────────

def figura_distribuciones(df: pd.DataFrame) -> Path:
	figura, ejes = plt.subplots(2, 2, figsize=(14, 9))
	(target, por_query), (precio, nutricion) = ejes

	comprados = int(df["bought"].sum())
	no_comprados = len(df) - comprados
	barras = target.bar(["no comprado", "comprado"], [no_comprados, comprados],
	                    color=[NEGATIVE, POSITIVE], width=0.55)
	target.bar_label(barras, fmt="{:,.0f}", padding=6, color=BODY, fontsize=14)
	target.set_ylim(0, no_comprados * 1.18)
	target.set_ylabel("filas")
	target.annotate(f"BTR base = {comprados / len(df):.4f}",
	                xy=(0.5, 0.82), xycoords="axes fraction", ha="center",
	                color=ACCENT, fontsize=15)
	set_title(target, "Distribución del target")

	conteo = df.groupby("query_id")["bought"].sum().value_counts().sort_index()
	# El estrato 3+ del split junta 3 y 4 porque sólo hay 5 queries con 4.
	colores = [ACCENT if indice >= 3 else BASELINE for indice in conteo.index]
	barras = por_query.bar([str(i) for i in conteo.index], conteo.values, color=colores, width=0.6)
	por_query.bar_label(barras, fmt="{:,.0f}", padding=6, color=BODY, fontsize=14)
	por_query.set_ylim(0, conteo.max() * 1.18)
	por_query.set_xlabel("productos comprados en la query")
	por_query.set_ylabel("queries")
	por_query.annotate("estrato 3+ del split:\nsólo 5 queries llegan a 4",
	                   xy=(0.62, 0.62), xycoords="axes fraction", color=BODY, fontsize=13)
	set_title(por_query, "Positivos por query")

	for eje, columna, titulo in ((precio, "price", "price"),
	                             (nutricion, "nutrition_score", "nutrition_score")):
		serie = pd.to_numeric(df[columna])
		eje.hist(serie, bins=40, color=ACCENT, alpha=0.75)
		eje.axvline(serie.median(), color=BODY, linestyle=(0, (4, 4)), linewidth=2)
		eje.annotate(f"mediana {serie.median():,.2f}\nmedia {serie.mean():,.2f}",
		             xy=(0.62, 0.78), xycoords="axes fraction", color=BODY, fontsize=13)
		eje.set_xlabel(titulo)
		eje.set_ylabel("filas")
		set_title(eje, f"Distribución de {titulo}")

	figura.tight_layout()
	return save(figura, OUTPUT_DIR / "14_distribuciones.jpg")


# ── Figura 15: el baseline que hay que vencer ─────────────────────────────────

def metricas_del_transformer() -> tuple[str, float, float]:
	"""El mejor resultado realmente entrenado, para no hardcodear números en la figura.

	Descarta corridas de una sola época: los smoke tests (`--epochs 1`) quedan
	guardados igual que cualquier otra corrida, y compararlos contra el lookup daría
	una diferencia negativa que no dice nada sobre la arquitectura.
	"""
	candidatos: list[tuple[str, float, float]] = []

	for archivo in sorted(EXPERIMENTS.glob("*/run.json")):
		datos = json.loads(archivo.read_text(encoding="utf-8"))
		epocas = (datos.get("summary") or {}).get("epochs") or len(datos.get("epoch_metrics") or [])
		prueba = datos.get("test") or {}
		roc, pr = prueba.get("roc_auc"), prueba.get("pr_auc")
		if int(epocas or 0) > 1 and roc is not None and pr is not None:
			candidatos.append((archivo.parent.name, float(roc), float(pr)))

	for resumen in sorted(EXPERIMENTS.glob("summary*.csv")):
		with resumen.open(encoding="utf-8") as archivo:
			for fila in csv.DictReader(archivo):
				if int(float(fila.get("epochs", 0))) > 1:
					candidatos.append(
						(fila["name"], float(fila["test_roc_auc"]), float(fila["test_pr_auc"]))
					)

	if not candidatos:
		raise SystemExit(
			"No hay ninguna corrida de más de una época guardada.\n"
			"Corré `python main.py` (sin --epochs 1) antes que este script."
		)
	return max(candidatos, key=lambda fila: fila[2])   # el de mejor PR-AUC


def baseline_title_tag() -> tuple[np.ndarray, np.ndarray, float]:
	"""BTR promedio por title_tag aprendido en train y aplicado a test."""
	from config.config import load_config
	from dataset.preprocess_dataset import get_data_processed

	splits = get_data_processed(load_config())
	train, test = splits["train"], splits["test"]
	media_por_tag = train.groupby("title_tag")["bought"].mean()
	scores = test["title_tag"].map(media_por_tag).fillna(train["bought"].mean())
	etiquetas = test["bought"].astype(int).to_numpy()
	return etiquetas, scores.to_numpy(dtype=float), float(etiquetas.mean())


def figura_baseline(nombre_modelo: str, roc_modelo: float, pr_modelo: float,
                    roc_lookup: float, pr_lookup: float, btr_test: float) -> Path:
	from plots.pr_auc import pr_auc_score  # noqa: F401  (import tardío: evita ciclos)

	figura, (izq, der) = plt.subplots(1, 2, figsize=(14, 6))
	etiquetas = ["azar", "title_tag\n(lookup)", "Transformer"]
	colores = [BASELINE, SERIES[1], ACCENT]

	for eje, valores, titulo in (
		(izq, [0.5, roc_lookup, roc_modelo], "ROC-AUC"),
		(der, [btr_test, pr_lookup, pr_modelo], "PR-AUC"),
	):
		barras = eje.bar(etiquetas, valores, color=colores, width=0.58)
		eje.bar_label(barras, fmt="%.4f", padding=6, color=BODY, fontsize=14)
		eje.set_ylim(0, 1.12)
		eje.set_ylabel(titulo)
		# La distancia que realmente compra la arquitectura.
		delta = valores[2] - valores[1]
		eje.annotate(f"{delta:+.4f}", xy=(2, valores[2]), xytext=(1.5, valores[2] + 0.13),
		             ha="center", color=ACCENT, fontsize=15,
		             arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8))
		set_title(eje, titulo)

	figura.tight_layout()
	return save(figura, OUTPUT_DIR / "15_baseline_title_tag.jpg")


def main() -> None:
	apply_theme()
	df = cargar()

	print("=" * 72)
	print("VALORES POSIBLES")
	for columna in CATEGORICAS:
		print(f"  {columna:20} {df[columna].nunique():>4} valores")
	for columna in NUMERICAS:
		serie = pd.to_numeric(df[columna])
		print(f"  {columna:20} min={serie.min():>8.2f} max={serie.max():>8.2f} "
		      f"mediana={serie.median():>8.2f} media={serie.mean():>8.2f}")

	print("\nCALIDAD DE DATOS")
	# Sobre las 22 columnas del CSV: product_name y title_tag las derivamos nosotros
	# y contarlas acá inflaría el número que después se cita en la filmina.
	crudo = df.drop(columns=["product_name", "title_tag"])
	print(f"  celdas vacías: {int(crudo.isna().sum().sum())} en {crudo.shape[1]} columnas")
	print(f"  filas duplicadas: {int(crudo.duplicated().sum())}")
	print(f"  duplicadas ignorando timestamp y query_id: "
	      f"{int(crudo.drop(columns=['timestamp', 'query_id']).duplicated().sum())}")
	dentro = ((pd.to_numeric(df["price"]) >= pd.to_numeric(df["filter_price_min"]))
	          & (pd.to_numeric(df["price"]) <= pd.to_numeric(df["filter_price_max"]))).sum()
	print(f"  price dentro del rango de filtros: {dentro}/{len(df)}")

	print("\nDISTRIBUCIÓN")
	print(f"  BTR base: {df['bought'].mean():.4f} ({int(df['bought'].sum())} de {len(df)})")
	print(f"  positivos por query: {df.groupby('query_id')['bought'].sum().value_counts().sort_index().to_dict()}")

	from plots.pr_auc import pr_auc_score
	from plots.roc_auc import roc_auc_score

	etiquetas, scores, btr_test = baseline_title_tag()
	roc_lookup = roc_auc_score(etiquetas, scores)
	pr_lookup = pr_auc_score(etiquetas, scores)
	nombre, roc_modelo, pr_modelo = metricas_del_transformer()

	print("\nBASELINE vs TRANSFORMER (test)")
	print(f"  {'azar':32} ROC-AUC {0.5:.4f}   PR-AUC {btr_test:.4f}")
	print(f"  {'title_tag (lookup)':32} ROC-AUC {roc_lookup:.4f}   PR-AUC {pr_lookup:.4f}")
	print(f"  {nombre:32} ROC-AUC {roc_modelo:.4f}   PR-AUC {pr_modelo:.4f}")
	print(f"  {'lo que agrega la arquitectura':32} "
	      f"ROC-AUC +{roc_modelo - roc_lookup:.4f}   PR-AUC +{pr_modelo - pr_lookup:.4f}")
	print(f"  valores distintos del score trivial: {len(set(scores))}")

	print("\nFIGURAS")
	for ruta in (
		figura_valores_posibles(df),
		figura_distribuciones(df),
		figura_baseline(nombre, roc_modelo, pr_modelo, roc_lookup, pr_lookup, btr_test),
	):
		print(f"  {ruta.relative_to(ROOT)}")


if __name__ == "__main__":
	main()
