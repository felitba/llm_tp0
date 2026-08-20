# Arquitectura propuesta para predecir BTR

## 1. Idea general

El problema combina distintos tipos de datos:

- texto: `product_name`, `comments`, `ingredients`
- categoricas: `category`, `brand`, `storage_type`, `allergens`, `country_of_origin`
- numericas: `price`, `filter_price_min`, `filter_price_max`, `net_weight_oz`, `nutrition_score`
- target: `bought`

La propuesta es convertir cada dato de entrada a un vector de la misma dimension `d_model`. Luego, esos vectores se apilan como una secuencia y se pasan a un Transformer Encoder.

```text
fila del dataset
    |
    |- texto -> tokenizer acotado -> token ids -> learned embedding
    |- categoricas -> categorical ids -> learned embeddings por columna
    `- numericas -> normalizacion -> linear projection
    |
    v
secuencia de vectores [B, seq_len, d_model]
    |
    v
Transformer Encoder
    |
    v
representacion contextual de la fila
    |
    v
classification head
    |
    v
logit -> sigmoid -> P(bought = 1)
```

## 2. Tokenizacion

La tokenizacion solo es necesaria para columnas de texto libre.

Columnas candidatas:

- `product_name`
- `comments`
- `ingredients`

Como el dominio del problema es acotado, no conviene arrancar con un tokenizer general gigante como GPT-2. Es mas razonable usar un tokenizer propio simple:

```text
"Family Pack Blueberry Muffins"
    -> ["family", "pack", "blueberry", "muffins"]
    -> [12, 48, 103, 221]
```

El vocabulario debe construirse solo con el train set para evitar data leakage.

Configuracion inicial sugerida:

```text
max_vocab_size = 2000 o 5000
min_freq = 2
max_product_name_len = 10
max_ingredients_len = 20
max_comments_len = 5
```

Tokens especiales:

```text
PAD = 0
UNK = 1
```

### Justificacion del tokenizer acotado

Para este problema se recomienda usar un tokenizer customizado y acotado al dataset, en lugar de un tokenizer general como GPT-2.

Motivos:

- El vocabulario del problema es chico y especifico: productos, marcas, ingredientes, categorias y comentarios comerciales.
- Un tokenizer general trae un vocabulario muy grande, por ejemplo GPT-2 usa mas de 50.000 tokens.
- Un vocabulario grande aumenta el tamano de la matriz de embeddings sin necesariamente aportar informacion util para este dominio.
- Un tokenizer propio permite controlar mejor el preprocesamiento y justificarlo en el EDA.
- Reduce costo computacional y riesgo de overfitting.

Ejemplo de diferencia:

```text
GPT-2 tokenizer:
vocab_size aproximado = 50257

Tokenizer custom:
vocab_size sugerido = 2000 a 5000
```

La matriz de embeddings depende de:

```text
vocab_size x d_model
```

Por ejemplo:

```text
GPT-2: 50257 x 64 = 3.216.448 parametros
Custom: 3000 x 64 = 192.000 parametros
```

Para una primera version del TP, el tokenizer custom es mas simple, mas liviano y mas coherente con el dominio del dataset.

## 3. Learned embeddings

Un learned embedding es una tabla entrenable que convierte IDs en vectores.

Ejemplo para `category`:

```text
0 -> Dairy
1 -> Frozen
2 -> Snacks
```

Con `d_model = 64`, cada categoria se representa con un vector de 64 numeros:

```python
self.category_embedding = nn.Embedding(num_categories, d_model)
```

Esos valores arrancan aleatorios y se ajustan con backpropagation durante el entrenamiento.

### Tipos de embeddings considerados

Se consideraron tres tipos de embeddings:

```text
static embedding
learned embedding
dynamic embedding
```

#### Static embedding

Un static embedding es un embedding preentrenado en otro corpus y normalmente fijo.

Ejemplos:

```text
Word2Vec
GloVe
FastText
```

Caracteristica principal:

```text
mismo token -> mismo vector
```

No se recomienda como opcion principal para este TP, porque el dataset tiene un dominio muy especifico de supermercado y e-commerce. Palabras o frases como `Customer Favorite`, `Low Feedback`, `Recently Added`, marcas y categorias del dataset pueden no estar bien representadas en embeddings generales.

#### Learned embedding

Un learned embedding arranca aleatorio y se aprende durante el entrenamiento del modelo.

Ejemplo:

```python
self.category_embedding = nn.Embedding(num_categories, d_model)
self.text_embedding = nn.Embedding(vocab_size, d_model)
```

Caracteristica principal:

```text
mismo ID -> mismo vector base aprendido
```

Se recomienda usar learned embeddings para este proyecto porque:

- aprenden directamente del dataset del TP;
- son simples de implementar;
- encajan bien con una arquitectura Transformer implementada desde cero;
- permiten representar categorias, marcas, paises, comentarios y tokens de texto en el mismo espacio `d_model`;
- se entrenan end-to-end con el objetivo final `bought`.

#### Dynamic embedding

Un dynamic embedding es una representacion que cambia segun el contexto.

Ejemplo conceptual:

```text
Potato Chips + Customer Favorite + price bajo
Potato Chips + Low Feedback + price alto
```

El token `Potato Chips` puede tener el mismo embedding inicial, pero luego del Transformer su representacion final puede ser distinta porque se combina con otras features de la fila.

En esta arquitectura:

```text
learned embeddings = entrada inicial del modelo
Transformer = genera representaciones contextuales/dinamicas
```

Por eso, la decision recomendada es:

```text
usar learned embeddings como entrada
y dejar que el Transformer produzca representaciones dinamicas segun la fila
```

## 4. Embedding por columna

La arquitectura principal propuesta usa embeddings separados por columna o tipo de feature.

Ejemplo:

```python
self.category_embedding = nn.Embedding(num_categories, d_model)
self.brand_embedding = nn.Embedding(num_brands, d_model)
self.storage_embedding = nn.Embedding(num_storage_types, d_model)
self.country_embedding = nn.Embedding(num_countries, d_model)
self.text_embedding = nn.Embedding(vocab_size, d_model)
```

Las columnas numericas no usan `nn.Embedding`, porque no son categorias. Se normalizan y se proyectan al mismo espacio:

```python
self.numeric_projection = nn.Linear(num_numeric_features, d_model)
```

La idea es que cada columna produzca uno o varios vectores de dimension `d_model`.

### Por que no usar one-hot encoding como representacion principal?

El one-hot encoding es una tecnica valida para variables categoricas y puede usarse como baseline. Por ejemplo:

```text
category = Dairy
    -> [1, 0, 0, 0, ...]
```

Esto funciona bien en modelos clasicos como regresion logistica, arboles o MLPs simples, donde todas las features se concatenan en un unico vector plano.

Sin embargo, para la arquitectura Transformer propuesta no es la representacion principal mas conveniente.

Motivos:

- El Transformer espera una secuencia de vectores con la misma dimension `d_model`.
- Cada one-hot tiene una dimension distinta segun la columna:

```text
category one-hot -> num_categories
brand one-hot -> num_brands
country one-hot -> num_countries
```

- Para pasar un one-hot al Transformer igual habria que proyectarlo a `d_model`.

Ejemplo:

```python
self.category_projection = nn.Linear(num_categories, d_model)
category_vector = self.category_projection(category_one_hot)
```

Pero esta operacion es casi equivalente a usar directamente un embedding:

```python
self.category_embedding = nn.Embedding(num_categories, d_model)
category_vector = self.category_embedding(category_id)
```

Intuicion:

```text
one-hot + Linear ~= seleccionar una fila de una matriz
Embedding ~= seleccionar una fila de una matriz
```

Por eso, para variables categoricas en el Transformer, `nn.Embedding` es mas natural y eficiente que `one-hot + Linear`.

Decision propuesta:

```text
Baseline:
one-hot encoding + MLP

Modelo principal:
learned embeddings + Transformer
```

De esta forma, el one-hot encoding no se descarta completamente: se puede usar para comparar contra una arquitectura mas simple. Pero para el Transformer principal, se prefieren learned embeddings porque ya producen vectores densos de dimension `d_model`, compatibles con la entrada esperada por el encoder.

## 5. Alternativa: embedding por topico

Tambien se podria agrupar informacion en topicos antes del Transformer.

Ejemplo:

```text
producto = product_name + brand + category
popularidad = comments
precio = price + filter_price_min + filter_price_max
nutricion = ingredients + allergens + nutrition_score
logistica = storage_type + country_of_origin + net_weight_oz
```

Luego cada topico se convierte en un vector:

```text
[
  producto_vector,
  popularidad_vector,
  precio_vector,
  nutricion_vector,
  logistica_vector
]
```

Esta opcion es mas compacta, pero introduce decisiones manuales sobre como agrupar las columnas. Para una primera version, conviene usar embeddings por columna y dejar la version por topicos como posible experimento de ablacion.

## 6. Entrada al Transformer

No conviene concatenar todos los vectores en un unico vector largo. Para un Transformer, lo mas natural es apilarlos como una secuencia.

Ejemplo conceptual:

```text
[
  CLS,
  product_name_token_1,
  product_name_token_2,
  comments_token_1,
  ingredients_token_1,
  category_vector,
  brand_vector,
  storage_type_vector,
  numeric_vector
]
```

Forma esperada:

```text
[batch_size, seq_len, d_model]
```

En codigo:

```python
sequence = torch.cat([
    cls_token,
    product_name_embeddings,
    comment_embeddings,
    ingredient_embeddings,
    categorical_tokens,
    numeric_token,
], dim=1)
```

## 7. Hace falta positional encoding?

Si, pero no exactamente de la misma forma para todos los datos.

Para texto, el orden importa:

```text
"low sodium pasta sauce"
```

No es lo mismo mirar palabras sin posicion. Por eso, a los tokens de texto conviene agregar positional encoding o positional embeddings.

Para features tabulares como `price`, `brand` o `category`, el orden en la secuencia es artificial. Ahi lo mas importante no es la posicion, sino saber que columna representa cada token. Por eso conviene agregar tambien un embedding de columna o de tipo de feature.

Una forma clara de pensarlo:

```text
vector_final = value_embedding + position_embedding + feature_type_embedding
```

Donde:

- `value_embedding`: representa el valor, por ejemplo `Dairy` o `Customer Favorite`
- `position_embedding`: representa la posicion dentro de una secuencia, especialmente util para texto
- `feature_type_embedding`: representa de que columna viene el dato, por ejemplo `category`, `brand` o `price`

En la arquitectura quedaria despues de los embeddings/proyecciones y antes del Transformer:

```text
tokens ya embebidos
    |
    v
+ positional encoding / feature type embedding
    |
    v
Transformer Encoder
```

Para una primera version simple:

- usar positional encoding para los tokens de texto;
- usar un orden fijo para las features tabulares;
- si hay tiempo, agregar embeddings de columna para distinguir mejor cada feature.

## 8. Que devuelve el Transformer?

El Transformer no devuelve directamente la prediccion final.

Dentro del Transformer se calculan attention scores, que indican cuanto mira un token o feature a otro. Pero la salida del Transformer son vectores contextualizados.

Ejemplo:

```text
entrada:
[product_vector, comments_vector, price_vector]

salida:
[product_contextual, comments_contextual, price_contextual]
```

Cada vector de salida ya contiene informacion de los otros vectores de la misma fila.

Luego se toma una representacion final, por ejemplo el token `CLS`, y se pasa por una cabeza clasificadora:

```python
logit = classifier(encoded[:, 0])
probability = torch.sigmoid(logit)
```

## 9. Como elegir d_model?

`d_model` es el tamano de cada vector que entra y sale del Transformer.

Ejemplo:

```text
d_model = 64
```

significa que cada token o feature queda representado por 64 numeros.

Un `d_model` mas chico:

- entrena mas rapido;
- usa menos memoria;
- tiene menos riesgo de overfitting;
- puede quedarse corto si el problema necesita representar muchas relaciones.

Un `d_model` mas grande:

- tiene mas capacidad;
- puede aprender relaciones mas complejas;
- entrena mas lento;
- tiene mas riesgo de overfitting, especialmente con datasets chicos;
- aumenta el costo del Transformer.

El enunciado sugiere empezar con una arquitectura chica, por ejemplo `d_model < 100`. Para este proyecto conviene arrancar con:

```text
d_model = 64
n_heads = 4
num_layers = 2
dim_feedforward = 128 o 256
dropout = 0.1
```

Importante: `d_model` debe ser divisible por `n_heads`.

Ejemplo:

```text
d_model = 64
n_heads = 4
dimension por head = 64 / 4 = 16
```

Si luego el modelo queda con underfitting, se puede probar:

```text
d_model = 96
n_heads = 4
num_layers = 3
```

## 10. Como aprende el modelo?

Si, el aprendizaje es por backpropagation.

Flujo:

```text
1. El modelo recibe una fila del dataset.
2. Produce un logit.
3. El logit se compara contra el label real `bought`.
4. Se calcula una loss.
5. La loss vuelve hacia atras con backpropagation.
6. Se actualizan embeddings, proyecciones, Transformer y clasificador.
```

Para clasificacion binaria conviene que el modelo devuelva un logit, sin `Sigmoid` dentro del modelo, y usar:

```python
loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(logits, labels)
```

Para obtener probabilidades en evaluacion:

```python
probabilities = torch.sigmoid(logits)
```

Esas probabilidades se usan para PR-AUC y ROC-AUC.

## 11. Diferencia con un MLP

Un MLP normalmente recibe todas las features concatenadas en un vector plano:

```text
[price, category_one_hot, brand_one_hot, title_features, ...]
```

Luego aplica capas lineales con pesos fijos.

Un Transformer recibe una secuencia de vectores:

```text
[
  product_vector,
  comments_vector,
  category_vector,
  brand_vector,
  price_vector
]
```

Mediante self-attention, aprende que features deben mirar a otras features dentro de cada fila.

La diferencia conceptual:

```text
MLP:
mezcla todo con pesos globales.

Transformer:
calcula relaciones entre tokens/features para cada fila.
```

Esto permite que el modelo aprenda interacciones como:

```text
Customer Favorite + precio bajo -> aumenta probabilidad de compra
Low Feedback + precio alto -> baja probabilidad de compra
marca fuerte + categoria especifica -> puede influir
storage_type + category -> consistencia del producto
```

## 12. Incorporar Masked Language Modeling

Si se pide usar una estrategia tipo BERT, se puede incorporar **Masked Language Modeling** como una etapa previa de entrenamiento.

MLM no predice `bought`. MLM entrena al Transformer a reconstruir tokens de texto ocultos.

Ejemplo:

```text
Texto original:
"low sodium pasta sauce"

Texto enmascarado:
"low [MASK] pasta sauce"

Objetivo:
predecir "sodium"
```

### Donde entra MLM en la arquitectura

Se propone entrenar en dos etapas:

```text
Etapa 1: pretraining con MLM
    texto + features opcionales
        |
        v
    Transformer Encoder
        |
        v
    MLM head
        |
        v
    predecir tokens enmascarados

Etapa 2: fine-tuning para BTR
    texto + categoricas + numericas
        |
        v
    Transformer Encoder preentrenado
        |
        v
    classification head
        |
        v
    predecir bought / BTR
```

La idea es reutilizar el mismo Transformer Encoder. En la primera etapa aprende representaciones del texto; en la segunda se ajusta para predecir compra.

### Como se arma el input para MLM

Para las columnas textuales:

- `product_name`
- `comments`
- `ingredients`

se tokeniza con el tokenizer custom acotado y se agrega un token especial:

```text
MASK
```

Luego se selecciona un porcentaje de tokens, por ejemplo `15%`, y se los oculta.

Regla habitual de BERT:

```text
80% -> reemplazar por [MASK]
10% -> reemplazar por un token random
10% -> dejar igual
```

La loss se calcula solo sobre los tokens seleccionados para predecir. Los tokens no seleccionados y los tokens `PAD` se ignoran.

### MLM head

Durante pretraining, en lugar de usar la cabeza clasificadora de `bought`, se usa una cabeza que predice vocabulario:

```python
self.mlm_head = nn.Linear(d_model, vocab_size)
```

La salida tiene forma:

```text
[batch_size, seq_len, vocab_size]
```

La loss puede ser:

```python
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
```

Donde `-100` marca posiciones que no deben participar de la loss.

### Que pasa con las columnas no textuales?

Hay dos opciones razonables:

#### Opcion simple

Hacer MLM solo con texto:

```text
product_name/comments/ingredients -> MLM
```

Luego, en fine-tuning, agregar categoricas y numericas para predecir `bought`.

Esta opcion es mas simple y mas cercana a BERT clasico.

#### Opcion multimodal

Durante MLM, pasar tambien las features categoricas y numericas como contexto:

```text
[texto enmascarado] + [category] + [brand] + [price] + ...
```

La loss sigue calculandose solo sobre tokens de texto enmascarados, pero el Transformer puede usar columnas como marca, categoria o precio para reconstruir mejor el texto.

Esta opcion esta mas alineada con la arquitectura final, pero es un poco mas compleja.

### Diferencia entre MLM y BTR

```text
MLM:
objetivo auto-supervisado
predice tokens ocultos
usa CrossEntropy sobre vocabulario

BTR:
objetivo supervisado
predice bought
usa BCEWithLogitsLoss
```

MLM sirve como pretraining. BTR es la tarea final.

### Recomendacion para el TP

Si se pide incorporar MLM, la version mas defendible seria:

```text
1. Tokenizer custom acotado.
2. Learned embeddings para tokens.
3. Transformer Encoder entrenado con MLM sobre columnas textuales.
4. Reutilizar ese encoder en el modelo completo.
5. Agregar categoricas y numericas.
6. Fine-tuning con BCEWithLogitsLoss para predecir bought.
```

Esto permite decir que la arquitectura usa una etapa tipo BERT:

> Primero se preentrena el encoder con Masked Language Modeling sobre el texto del producto. Luego se reutilizan sus pesos y se ajusta el modelo completo para la tarea supervisada de prediccion de BTR.

## 13. Arquitectura recomendada para la primera version

```text
Tokenizer acotado para texto
    |
    v
Learned embeddings para texto y categoricas
    |
    v
Linear projection para numericas
    |
    v
Agregar positional/feature embeddings
    |
    v
Transformer Encoder chico
    |
    v
CLS token o pooling
    |
    v
Classification head
    |
    v
logit de compra
```

Configuracion inicial:

```text
vocab_size = 2000 a 5000
d_model = 64
n_heads = 4
num_layers = 2
dim_feedforward = 128 o 256
dropout = 0.1
loss = BCEWithLogitsLoss
metricas = PR-AUC, ROC-AUC
```

Esta arquitectura es defendible para el TP porque usa un Transformer de forma pertinente: combina informacion heterogenea de una misma fila y aprende interacciones entre texto, categorias y numeros para predecir la probabilidad de compra.
