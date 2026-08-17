# Flujo del modelo principal: Transformer multimodal

El modelo principal usa un Transformer para combinar texto, variables numericas y variables categoricas de una misma fila del dataset. El objetivo es predecir la probabilidad de compra (`bought`).

## 1. Flujo general

```text
Fila del CSV
   |
   |- numericas: price, min/max filtro, peso, nutrition_score
   |- categoricas: category, brand, storage_type, allergens, pais
   |- texto: title
   `- target: bought
   |
   v
Preprocesamiento
   |
   |- numericas -> imputar y normalizar
   |- categoricas -> mapear a IDs enteros
   `- title -> tokenizar a IDs de palabras/subwords
   |
   v
Construir tokens de dimension d_model
   |
   |- cada numero -> Linear(1, d_model)
   |- cada categoria -> Embedding(n_categorias, d_model)
   `- cada token de texto -> Embedding(vocab_size, d_model)
   |
   v
[CLS] + tokens numericos + tokens categoricos + [SEP] + tokens del titulo
   |
   v
Transformer Encoder
   |
   v
Vector final de [CLS]
   |
   v
MLP / capa lineal
   |
   v
logit -> sigmoid -> P(bought = true)
```

## 2. Ejemplo de una fila

Dados estos valores:

```text
price=4.29
filter_price_min=1.50
filter_price_max=8.93
nutrition_score=44
category=Bakery
brand=Harvest Lane
title="Family Pack Blueberry Muffins Customer Favorite"
```

el modelo construye una secuencia conceptual como la siguiente:

```text
[CLS]
[precio]
[filtro_minimo]
[filtro_maximo]
[peso]
[nutricion]
[categoria=Bakery]
[marca=Harvest Lane]
[storage=Ambient]
[alergeno=Wheat]
[pais=Vietnam]
[SEP]
[Family] [Pack] [Blueberry] [Muffins] [Customer] [Favorite]
```

Cada elemento de la secuencia termina representado por un vector de la misma dimension, por ejemplo `d_model = 64`. Asi, el Transformer puede aprender relaciones entre el precio, el contexto de busqueda, la marca, la categoria y las palabras del titulo.

## 3. Preprocesamiento

### Features numericas

Se imputan si hubiera valores faltantes y se normalizan usando estadisticas calculadas exclusivamente sobre el conjunto de entrenamiento.

```text
price = 4.29
    |
    v
StandardScaler
    |
    v
price_normalized = -0.73
    |
    v
Linear(1, d_model)
    |
    v
token de precio: [d_model]
```

Cada feature numerica tiene su propia proyeccion lineal, para que el modelo pueda distinguir el precio del puntaje nutricional, aunque ambos sean numeros.

### Features categoricas

En este modelo las categorias no se representan con one-hot encoding. Cada valor se mapea a un ID y se usa una tabla de embeddings por feature.

```text
brand = "Harvest Lane"
    |
    v
brand_id = 4
    |
    v
nn.Embedding(num_brands, d_model)
    |
    v
token de marca: [d_model]
```

Para valores no vistos en validacion, test o produccion se reserva un ID `UNK`.

One-hot encoding sigue siendo una buena opcion para el modelo baseline; los embeddings son mas naturales en este Transformer porque producen directamente un token de feature.

### Texto

El titulo se tokeniza. Para evitar que el texto ocupe una parte excesiva de la secuencia, se fija una longitud maxima, por ejemplo `max_text_length = 32`.

```text
"Family Pack Blueberry Muffins" -> [token_1, token_2, token_3, token_4]
```

Las secuencias mas cortas se completan con padding al armar el batch. La mascara de atencion indica que posiciones corresponden a padding y deben ignorarse.

## 4. Salida del Dataset

Cada item de `ProductDataset` puede devolver:

```python
{
    "numeric": tensor([
        price_normalized,
        filter_price_min_normalized,
        filter_price_max_normalized,
        net_weight_oz_normalized,
        nutrition_score_normalized,
    ]),
    "categorical": tensor([
        category_id,
        brand_id,
        storage_type_id,
        allergen_id,
        country_id,
    ]),
    "input_ids": tensor([text_token_id_1, text_token_id_2, ...]),
    "text_mask": tensor([1, 1, ...]),
    "label": tensor(0.0),
}
```

La `collate_fn` del `DataLoader` hace padding del texto y genera batches con las siguientes formas:

```text
numeric:    [B, n_numeric_features]
categorical:[B, n_categorical_features]
input_ids:  [B, max_text_length]
text_mask: [B, max_text_length]
label:      [B]
```

`B` es el tamano del batch.

## 5. Recorrido dentro del modelo

La interfaz del modelo puede ser:

```python
logits = model(
    numeric=batch["numeric"],
    categorical=batch["categorical"],
    input_ids=batch["input_ids"],
    text_mask=batch["text_mask"],
)
```

Dentro de `forward`, el recorrido conceptual es:

```python
num_tokens = numeric_to_tokens(numeric)           # [B, n_num, d_model]
cat_tokens = categorical_to_tokens(categorical)  # [B, n_cat, d_model]
text_tokens = text_embedding(input_ids)          # [B, text_len, d_model]

sequence = torch.cat([
    cls_token,
    num_tokens,
    cat_tokens,
    sep_token,
    text_tokens,
], dim=1)                                         # [B, total_len, d_model]

encoded = transformer(sequence, mask)
logits = classifier(encoded[:, 0]).squeeze(-1)    # [B]
```

El primer token es `[CLS]`. Despues de pasar por el Transformer, `encoded[:, 0]` contiene una representacion contextual de todos los features de esa fila y se usa para clasificar.

Ademas de los embeddings, debe agregarse codificacion posicional a la secuencia. Resulta conveniente agregar tambien embeddings de tipo de token para distinguir texto, numeros y categorias.

## 6. Entrenamiento

La salida del modelo debe ser un logit, sin `Sigmoid` dentro de la cabeza clasificadora. Se entrena con:

```python
loss_fn = torch.nn.BCEWithLogitsLoss()

logits = model(numeric, categorical, input_ids, text_mask)
loss = loss_fn(logits, labels)
```

En evaluacion, la probabilidad de compra se obtiene con:

```python
probabilities = torch.sigmoid(logits)
```

Estas probabilidades se usan para calcular PR-AUC y ROC-AUC. El BTR de un conjunto de productos puede estimarse como el promedio de sus probabilidades predichas.

## 7. Orden sugerido de implementacion

1. Definir el split train/validation/test, preferentemente agrupando por `query_id` si se considera que filas de una misma busqueda no deben mezclarse entre splits.
2. Ajustar el escalador numerico y los mapeos categoricos usando solo train.
3. Implementar `ProductDataset` y la funcion de padding del `DataLoader`.
4. Implementar los modulos `numeric_to_tokens` y `categorical_to_tokens`.
5. Adaptar el Transformer para usar `batch_first=True`, mascaras de padding y un token `[CLS]`.
6. Implementar la cabeza clasificadora y el loop de entrenamiento en `main.py`.
7. Comparar contra un baseline de one-hot encoding con regresion logistica o MLP y realizar ablaciones.

Para la primera ejecucion se recomienda un modelo pequeno: `d_model=64`, `n_heads=4`, `num_layers=2`, `max_text_length=32` y dropout de `0.1`.
