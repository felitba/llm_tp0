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

**Checkpoint de maxima `val_pr_auc`**, elegido post-hoc entre las 50 epocas.

`early_stopping_metric: "val_pr_auc"` con `early_stopping_patience: 0`: se
entrenan siempre las 50 epocas y recien al final se restaura el mejor checkpoint.
No hay corte anticipado, asi que la seleccion mira todas las epocas y no solo las
anteriores a un corte (Apicella et al. 2026 encuentran que la seleccion post-hoc
es al menos tan buena como el early stopping con patience, y mas estable).

Se elige por `val_pr_auc` y no por `val_loss` porque es la metrica que se
reporta. La entropia cruzada es una regla de puntuacion propia: se descompone en
calibracion mas ordenamiento, y penaliza la calibracion. La AP es invariante a
cualquier transformacion monotona de los scores, asi que depende solo del
ordenamiento. Minimizar `val_loss` optimiza una propiedad que no se reporta; en
la practica su minimo llega temprano y despues sube mientras el ordenamiento no
empeora, que es el fenomeno documentado por Guo et al. (2017) ("neural networks
can overfit to NLL without overfitting to the 0/1 loss") y explicado por Soudry
et al. (2018) via el crecimiento de la norma de los logits.

`val_loss` se sigue calculando y graficando como **diagnostico de sobreajuste**,
que es lo que pide la consigna, pero no selecciona.

### Limitacion medida (va explicita en el informe)

Con 131 positivos de validacion, el error estandar bootstrap por consulta de la
AP es **0.032-0.050**, y la variacion de la AP entre epocas es **0.007-0.029**.
El ruido de medicion es mayor que la diferencia entre epocas, asi que la epoca
elegida no es una propiedad estable del modelo: en `s1b_all_text_5col` el argmax
cayo en la epoca **5, 5 y 47** segun la semilla.

De ahi dos reglas de reporte:

1. **El numero que se reporta es el de TEST.** Test no participa de la
   seleccion, asi que no queda sesgado por ella. La AP de *validacion* en la
   epoca elegida **si** esta sesgada al alza, por ser el maximo de 50
   estimaciones ruidosas (Jensen & Cohen 2000; Cawley & Talbot 2010), y por eso
   no se presenta como estimacion insesgada de nada.
2. **Se reporta la epoca elegida por semilla**, junto con el intervalo de
   confianza bootstrap por consulta. Diferencias entre brazos menores que ese
   intervalo se informan como empates.

### Alternativa evaluada y descartada

`scripts/reselect.py` conserva la regla `ensemble` (promedio de las
probabilidades de las epocas posteriores al primer 20% del presupuesto), que no
elige ninguna epoca y por lo tanto no puede elegir ruido. En un bake-off hecho
**solo con validacion** (mitades por consulta, 200 particiones, seleccionar en
una mitad y medir en la otra) dio 0.7585 contra 0.7495 del argmax: unos 0.009 de
AP a favor del promedio, diferencia **no significativa** a nivel brazo
(p=0.132). Se opta por el checkpoint unico por simplicidad de reporte, con la
diferencia documentada aca. La regla sigue disponible sin reentrenar, porque
cada corrida guarda las probabilidades de todas las epocas.

Lo que **no** se usa: SWA ni Snapshot Ensembles. SWA promedia pesos bajo LR
ciclico o constante alto sobre SGD y necesita recalcular BatchNorm; Snapshot
depende de annealing coseno ciclico y su propia ablacion NoCycle muestra que sin
el ciclo no funciona. Entrenamos con AdamW a LR constante 3e-4 sin scheduler, asi
que ninguna de las dos aplica.

## 3. Promediado

**No se promedia nada.** Cada corrida reporta un unico checkpoint.

En particular **no se promedia entre semillas**: promediar las probabilidades de
las 3 semillas construiria un deep ensemble (Lakshminarayanan et al. 2017), que
es un modelo distinto y mas fuerte que cualquiera de sus miembros, y ocultaria
exactamente la varianza que hay que reportar. Las semillas son replicas, no
componentes de un modelo.

## 4. Comparacion entre configuraciones

Tres semillas de inicializacion (42, 7, 1234) por brazo, con el split fijo. Se
reporta **media y rango** entre semillas.

Los intervalos de confianza del 95% salen de un **bootstrap a nivel de consulta**
(2000 remuestreos de consultas completas, `query_bootstrap_ci` en
`scripts/reselect.py`). Las filas de una misma `query_id` son los productos de
una sola busqueda y son dependientes, asi que remuestrear filas subestimaria el
intervalo (Field & Welsh 2007).

**Diferencias entre brazos menores que el intervalo se reportan como empates.**
Con SE ~0.04 el intervalo tiene un semiancho de ~0.075, asi que los brazos del
grupo puntero caen todos dentro del mismo intervalo y se informan como un unico
resultado. Ademas de la media entre semillas se reporta **la epoca elegida por
cada semilla**, que es la limitacion de la seccion 2.

## 5. Evaluacion sobre test

- Una sola vez por brazo, al final, despues de que este archivo este commiteado.
- `scripts/reselect.py` requiere `--final` para leer las predicciones de test.
- Se declara en el informe que durante el diseno del protocolo se inspecciono
  test PR-AUC por epoca de forma diagnostica, y que la regla final se fijo solo
  con validacion.

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

## Comando de reporte

```
python main.py --config config/01_entrada.json
python scripts/reselect.py --config config/01_entrada.json --rule val_pr_auc --apply --final
```
