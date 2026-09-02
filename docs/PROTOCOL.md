# Protocolo de seleccion de modelo y evaluacion

**Congelado el 2026-09-01, antes de la evaluacion final sobre test.**

Este archivo existe para que "la regla se fijo usando validacion" sea
verificable por el timestamp del commit y no una afirmacion a posteriori.
Cualquier cambio a este protocolo debe ser un commit propio, con fecha, y
anterior a la corrida de test que reporte.

## 1. Regla de detencion del entrenamiento

Presupuesto fijo de **50 epocas, siempre completas**. Sin early stopping:
`early_stopping_patience: 0`.

No hay `patience` ni `min_delta` que justificar porque no hay detencion
temprana. Esto es deliberado: la literatura no fija un valor para ninguno de los
dos (Prechelt 1998 ajusta sus umbrales empiricamente; Goodfellow et al. 2016
presentan `p` como hiperparametro libre del Algoritmo 7.1). Correr el
presupuesto completo ademas garantiza la misma grilla de epocas para todos los
brazos, que es lo que hace legitima la comparacion.

## 2. Regla de seleccion de checkpoint

**Checkpoint de minima `val_loss`**, elegido post-hoc entre las 50 epocas.

`early_stopping_metric: "val_loss"` con `early_stopping_patience: 0`: se entrenan
siempre las 50 epocas y al final se restaura el checkpoint de menor perdida de
validacion. No hay corte anticipado, asi que la seleccion mira todas las epocas.

Es la regla convencional: tanto Prechelt (1998) como Goodfellow et al. (2016,
Algoritmo 7.1) definen la detencion sobre el **error de validacion**, no sobre la
metrica de la tarea. Apicella et al. (2026), el estudio mas directo sobre esta
eleccion, encuentra que los criterios basados en perdida son mas estables que los
basados en la metrica.

**Selecciona la perdida, evalua la metrica.** Son trabajos distintos y esta bien
que sean cantidades distintas: la perdida de validacion dice *cuando* el modelo
empieza a sobreajustar, y PR-AUC / ROC-AUC dicen *que tan util* es el clasificador
resultante. La consigna pide exactamente eso: "evaluar la performance con PR-AUC,
ROC-AUC y metricas propias de modelos teniendo en cuenta overfitting y
underfitting". Con esta regla la curva de perdida que se muestra en el informe y
el checkpoint que se reporta cuentan **la misma historia**:

```
train loss  ↓
val loss    ↓
              minimo  ← checkpoint seleccionado
val loss    ↑         → sobreajuste
```

Si se seleccionara por maxima val_pr_auc, la figura de perdida ilustraria una cosa
y el checkpoint vendria de otra.

### Alternativas evaluadas

Se comparo con maxima `val_pr_auc` en un bake-off hecho **solo con validacion**
(las queries de validacion partidas al medio 200 veces; cada regla elige con una
mitad y se mide en la otra). Las dos reglas son **indistinguibles**: minima
val_loss da 0.7459 de AP media contra 0.7495 de maxima val_pr_auc, pero gana en
posicion media (5.80 contra 6.20). Ninguna domina y la diferencia esta dentro del
ruido, asi que la eleccion se hace por defendibilidad, no por performance.

Tambien se probaron el promedio de probabilidades entre epocas (0.7585, la mejor
de todas pero no significativa: p=0.132 a nivel brazo) y el suavizado de la
metrica. Se descartan por desproporcionados para lo que pide la consigna: obligan
a explicar ruido de seleccion y sesgo del maximo en una presentacion de 25-30
minutos cuyo tema es la arquitectura Transformer. `scripts/reselect.py` conserva
las reglas (`val_pr_auc`, `val_pr_auc_smooth`, `ensemble`) para reexaminarlas sin
reentrenar si hiciera falta.

### Lo unico que hay que vigilar

La minima val_loss puede caer muy temprano. En la tanda anterior cayo en la epoca
3 en un brazo. No cambia la regla, pero **si algun brazo selecciona una epoca muy
temprana hay que decirlo** en el informe, porque ahi el brazo no esta compitiendo
en igualdad de condiciones. Como cada corrida guarda las 50 epocas, esto se
verifica despues sin reentrenar.

## 3. Promediado

**No se promedia nada.** Cada corrida reporta un unico checkpoint.

En particular **no se promedia entre semillas**: promediar las probabilidades de
las 3 semillas construiria un deep ensemble (Lakshminarayanan et al. 2017), que es
un modelo distinto y mas fuerte que cualquiera de sus miembros, y ocultaria
exactamente la varianza que hay que reportar. Las semillas son replicas.

## 4. Semillas y comparacion entre configuraciones

**Tres semillas (42, 7, 1234) para toda configuracion que se compare o se
reporte.** Una sola semilla queda reservada para pruebas exploratorias que no
sostienen ninguna conclusion de la ablacion; si un numero entra en una figura o en
una afirmacion del informe, tiene tres semillas detras.

El procedimiento, por configuracion:

1. Entrenar de forma independiente con las semillas `42`, `7` y `1234`, sobre la
   misma particion train/validation/test (`split_seed` 42 fijo, asi lo unico que
   cambia entre las tres corridas es la inicializacion).
2. Para cada semilla, quedarse con el **unico checkpoint de minima perdida de
   validacion** (seccion 2).
3. Registrar la PR-AUC y la ROC-AUC de **validacion** de ese checkpoint.
4. Comparar configuraciones por la **media de las tres semillas**, mostrando
   siempre los tres valores individuales y el rango.
5. **No elegir la mejor semilla.** Quedarse con la mejor de tres es una seleccion
   sobre ruido, y peor que la de epoca: son tres muestras independientes en vez de
   50 correlacionadas.
6. Recien despues de congelar las decisiones de configuracion, evaluar sobre test
   el checkpoint seleccionado de **cada una de las tres semillas**.
7. Reportar PR-AUC y ROC-AUC de test como **media de las tres semillas**, junto a
   los valores individuales o el rango.

`scripts/reselect.py` imprime la tabla del punto 4 (una fila por configuracion,
media y rango sobre validacion) sin necesidad de `--final`, asi la ablacion se
decide con test cerrado. Con `--apply --final` imprime la del punto 7 y la guarda
en `resultados_finales.csv`.

Los intervalos de confianza del 95% salen de un **bootstrap a nivel de consulta**
(2000 remuestreos de consultas completas, `metrics/uncertainty.py`). Las filas de
una misma `query_id` son los productos de una sola busqueda y son dependientes,
asi que remuestrear filas subestimaria el intervalo (Field & Welsh 2007). El
intervalo se calcula sobre la semilla mediana: pertenece a un modelo, no a la
media de tres.

**Rango e intervalo miden cosas distintas y no se suman.** El rango entre semillas
es varianza de optimizacion (que habria pasado con otra inicializacion); el
intervalo bootstrap es varianza de muestreo (que habria pasado con otras 1009
filas de test). En este dataset el segundo es unas cinco veces mas grande.

**Diferencias entre configuraciones menores que el intervalo se reportan como
empates.** Ademas de la media se reporta **la epoca elegida por cada semilla**,
que es la limitacion de la seccion 2.

### Parrafo de metodologia para el informe

> Cada configuracion se entrena durante 50 epocas bajo tres semillas (42, 7 y
> 1234) sobre la misma particion train/validation/test. Para cada corrida se
> conserva el checkpoint de minima perdida de validacion. Las configuraciones se
> comparan utilizando PR-AUC y ROC-AUC de validacion del checkpoint seleccionado,
> reportando media y rango entre semillas. La perdida y las curvas
> train/validation se utilizan ademas para analizar overfitting y underfitting.
> Una vez elegida la configuracion final, los tres checkpoints correspondientes a
> sus tres semillas se evaluan sobre test y se reporta la media y el rango de
> PR-AUC y ROC-AUC. Test no participa en la seleccion de configuraciones. No se
> promedian pesos ni predicciones entre semillas.

## 5. Evaluacion sobre test

Test se usa **una sola vez, al final**, para reportar PR-AUC y ROC-AUC del modelo
ya elegido. No se usa para comparar brazos ni para encadenar los pasos 01-06.

- `scripts/reselect.py` requiere `--final` para leer las predicciones de test.
- Se declara en el informe que durante el diseno del protocolo se inspecciono
  test PR-AUC por epoca de forma diagnostica, y que la regla final se fijo solo
  con validacion.
- `error_patterns.txt` y `error_patterns.jpg` (analisis de errores sobre test) son
  parte del reporte final, no una herramienta para iterar el modelo.

### El flujo completo

```
TRAIN          aprende los pesos          x3 semillas (42, 7, 1234)
   |
VALIDATION     1. elige el checkpoint de minima val_loss, por semilla
               2. compara configuraciones por la MEDIA de las 3 semillas
                  (mostrando siempre los 3 valores y el rango)
               3. observa sobreajuste en las curvas train/val
   |
   |           congelar las decisiones de configuracion
   |
TEST           una sola vez: los 3 checkpoints elegidos, uno por semilla
                  -> PR-AUC y ROC-AUC, media + rango entre semillas
```

## 6. Metrica

**PR-AUC estimada como Average Precision sin interpolar**, calculada por
`sklearn.metrics.average_precision_score` a traves de `plots/pr_auc.pr_auc_score`.
ROC-AUC por `sklearn.metrics.roc_auc_score` a traves de `plots/roc_auc.roc_auc_score`.
Las dos delegan en sklearn para que el numero reportado sea el de la
implementacion de referencia y no haya que defenderlo.

PR-AUC es la cantidad que pide la consigna; Average Precision es el estimador de
esa cantidad. El trapecio que se usaba antes interpola linealmente en el espacio
PR, lo cual Davis & Goadrich (2006) muestran que no es alcanzable, y Boyd et al.
(2013) no lo recomiendan. El error no es un offset constante: crece cuando la
distribucion de scores es gruesa, asi que era distinto por brazo y rompia la
comparacion.

- Misma funcion en validacion y test.
- Se reporta contra la tasa base de positivos (0.126 en test) (Saito &
  Rehmsmeier 2015).
- **ROC-AUC**: el estimador trapezoidal si es correcto en espacio ROC (ahi la
  interpolacion lineal es alcanzable), pero la implementacion propia ordenaba con
  `np.argsort`, que no es estable, y barajaba los puntos con igual FPR (los
  tramos verticales de la curva): hasta 0.023 de diferencia contra sklearn en
  entradas chicas. Corregido con `np.lexsort` en `area_under_curve`, y el escalar
  reportado viene de sklearn de todas formas.

## 7. Desempate entre configuraciones (agregado 2026-09-01, antes de la lectura de test)

Este agregado no modifica ninguna regla de las secciones 1-6; formaliza el criterio de
desempate que las tablas del informe ya venian usando, y se congela junto con el resto
**antes** de la unica lectura de test.

El intervalo de la seccion 4 es marginal: acota cuanto se moveria UN brazo con otra
muestra de consultas. Para comparar dos brazos la cantidad correcta es la DIFERENCIA, y
su intervalo sale del **bootstrap pareado por consulta**
(`scripts/paired_bootstrap.py`): en cada uno de los 2000 remuestreos se toman las mismas
consultas para los dos brazos y se mide la diferencia de AP, promediando las semillas de
cada brazo dentro del remuestreo. La varianza compartida de la muestra se cancela en la
resta, que es exactamente la reduccion de varianza que compra el apareamiento.

- **IC 95% de la diferencia que contiene 0 -> empate**, sin importar P(delta>0), que se
  reporta aparte como descripcion (un empate con signo consistente no es lo mismo que una
  moneda al aire, pero tampoco es una diferencia).
- Entre empatados se elige por **estabilidad entre semillas, costo y simplicidad**, y la
  eleccion se declara como tal en el informe: nunca por el tercer decimal de la media.
- Validacion solamente, como todo lo que decide entre brazos.

## Comando de reporte

```
# 1. entrenar y comparar los brazos sobre VALIDACION (test queda cerrado)
python main.py --config config/01_entrada.json
python scripts/reselect.py --config config/01_entrada.json

# 2. una sola vez, cuando el modelo ya esta elegido
python scripts/reselect.py --config config/01_entrada.json --apply --final
```
