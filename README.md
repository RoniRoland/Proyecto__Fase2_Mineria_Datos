# Análisis de Minería de Datos - Estadísticas Hospitalarias mediante Árboles, Bosques Aleatorios y Redes Neuronales
**Parte 2 – Modelos Predictivos**
---

## 1. Objetivo del proyecto

El objetivo de esta documentación es describir de manera técnica la implementación de los modelos predictivos utilizados para analizar datos hospitalarios internos de Guatemala.  
Se abordan tres enfoques de aprendizaje supervisado:

- Árboles de decisión (R)
- Bosques aleatorios (R)
- Redes neuronales (Python – Google Colab)

Cada modelo utiliza variables clínicas y demográficas del dataset unificado para predecir categorías relevantes del comportamiento hospitalario.

---
## 2. Estructura general del proyecto

El flujo general del proyecto es:

1. **Unificación de bases de datos anuales (2018–2024)** en R.  
2. **Modelos en R**:
   - 4 Árboles de decisión:
     - Árbol 1: Tratamiento simple vs complejo (`TRATA_COMPLEJO`).
     - Árbol 2: Mes de atención (`MES`).
     - Árbol 3: Rango de edad (`EDAD_RANGO`).
     - Árbol 4: Tipo de tratamiento detallado (`TRATARECIB` = 1, 2, 3).
   - 3 Bosques aleatorios:
     - RF1: Tratamiento simple vs complejo.
     - RF2: Mes de atención.
     - RF3: Rango de edad.
3. **Modelos en Python**:
   - 3 Redes neuronales que replican/“versionan” tres de los modelos de árboles:
     - RN1: Tratamiento simple vs complejo.
     - RN2: Rango de edad.
     - RN3: Tipo de tratamiento (TRATARECIB 1/2/3).




---
## 3. Requisitos previos

### Software
- **R** versión 4.3.3 o superior
- (Opcional) **RStudio**
- **Python 3.x** (Google Colab)

### Paquetes necesarios en R
Instala los siguientes paquetes:

```r
install.packages("readxl")
install.packages("dplyr")
install.packages("purrr")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("randomForest")
```
Librerías utilizadas:
```r
library(readxl)
library(dplyr)
library(purrr)
library(rpart)
library(rpart.plot)
library(randomForest)
```


### Paquetes necesarios en Python (Google Colab)

Para las redes neuronales se utiliza TensorFlow y Scikit-Learn. Instalar (si no está preinstalado en Colab):

```python
!pip install tensorflow
!pip install scikit-learn
!pip install pandas
!pip install numpy
```

Luego se importan:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

```

> En caso de error durante la instalación, asegúrate de tener configurado Rtools (Windows) o las herramientas de compilación en Linux/macOS.



### Set de datos
- El conjunto de datos de los servicios internos hospitalarios se pueden conseguir directamente en la pagina del INE (Instituto Nacional de Estadistica): [interna-servicios-hospitalarios](https://datos.ine.gob.gt/dataset/estadisticas-hospitalarias-servicios-internos).
> En la misma pagina del INE se encuentra el diccionario de definciones para entender el significado de cada columna del set de datos.

---

## 4. Estructura esperada del dataset

Se utilizaron los datasets del año 2018 al 2024 la cual cada una debe contener, al menos, las siguientes columnas (según el diccionario provisto):

| Variable        | Descripción (según diccionario) | Observaciones/Valores típicos |
|-----------------|----------------------------------|-------------------------------|
| **AÑO**         | Año del registro                 | Constante (ej. 2024); se elimina del análisis de reglas si no aporta variación. |
| **MES**         | Mes del registro                 | 1–12 (Enero–Diciembre). |
| **DIASESTAN**   | Días de estancia                 | Válidos: 1–98; **9999 = ignorado** (filtrar antes de graficar/clusterizar). |
| **SEXO**        | Sexo del paciente                | 1 = Hombre; 2 = Mujer. |
| **PPERTENENCIA**| Pueblo de pertenencia            | 1 = Maya; (otros códigos según diccionario: Garífuna, Xinka, Ladino/Mestizo, etc.). |
| **EDAD**        | Edad numérica                    | En **unidades indicadas por `PERIODOEDA`**. Para análisis se usa en años con rango 0–99. |
| **PERIODOEDA**  | Período de Edad (unidad)         | 1 = Días; 2 = Meses; 3 = Años; 9 = Ignorado. |
| **DEPTORESIDEN**| Departamento de residencia       | Códigos departamentales (p. ej. 1 = Guatemala). |
| **MUNIRESIDEN** | Municipio de residencia          | Códigos municipales (p. ej. 0101 = Guatemala). |
| **CAUFIN / Causa de atención** | Causa (diagnóstico) | Codificado en **CIE-10** (ver hoja *CIE-10* del diccionario). |
| **CONDIEGRES**  | Condición de egreso              | 1 = Vivo; (otros códigos según diccionario, p. ej. 2 = Fallecido). |
| **TRATARECIB**  | Tratamiento recibido             | 1 = Médico; (otros códigos según diccionario, p. ej. quirúrgico/obstétrico). |

> Nota: Los códigos exactos y etiquetas completas están en el archivo **diccionario-variables-interna.xlsx** (hoja *Interna* y *CIE-10*). Ajusta los mapeos de etiquetas en tus reportes si requieres nombres legibles.

-----------|--------------|
| **AÑO** | Año del registro (constante en 2024, se elimina del análisis) |
| **EDAD** | Edad del paciente (0–99 años) |
| **DIASESTAN** | Días de estancia hospitalaria (1–98 válidos, 9999 = ignorado) |
| **TRATARECIB** | Tipo de tratamiento recibido (1 a 3) |
| **PERIODOEDA** | Unidad de edad (1: Días, 2: Meses, 3: Años, 9: Ignorado) |

> Los valores fuera de rango (p. ej. 999 o 9999) representan datos **ignorados** y se filtran antes del clustering.

---

## 5. Carga y unificación del dataset (2018–2024)

Ajusta la ruta del archivo Excel a tu entorno (usa `C:/` o `C\\` en Windows), los archivos anuales fueron almacenados en una carpeta y unidos mediante `map_dfr()`.

```r
ruta <- "/home/data_hospitalaria/"

archivos <- list.files(ruta, pattern = "\\.xlsx$", full.names = TRUE)

leer_base <- function(path) {
  read_excel(path)
}

datos <- map_dfr(archivos, leer_base)

```

Esto genera un único dataframe consolidado, lo cual permite entrenar modelos con mayor robustez estadística.

- list.files(...) obtiene todos los archivos .xlsx.
- map_dfr lee cada archivo y los une “uno debajo del otro” en un solo data.frame llamado datos.

---

## 6. Árboles de decisión en R

### Árbol 1 – Predicción del Tipo de Tratamiento (Simple vs Complejo)

#### Preparación del dataset

```r
datos_modelo <- datos %>%
  filter(
    DIASESTAN >= 1, DIASESTAN <= 98,
    EDAD >= 0, EDAD <= 99
  ) %>%
  mutate(
    TRATA_COMPLEJO = case_when(
      TRATARECIB == 1 ~ "SIMPLE",
      TRATARECIB %in% c(2, 3) ~ "COMPLEJO",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(TRATA_COMPLEJO))
```

- Se filtran valores atípicos (EDAD fuera de rango, DIASESTAN inválidos).
- Se crea la nueva variable objetivo TRATA_COMPLEJO a partir de TRATARECIB.

#### Definición del dataset del árbol

```r
data_arbol1 <- datos_modelo %>%
  select(
    TRATA_COMPLEJO,
    EDAD,
    SEXO,
    PPERTENENCIA,
    DIASESTAN
  ) %>%
  mutate(
    TRATA_COMPLEJO = factor(TRATA_COMPLEJO),
    SEXO           = factor(SEXO),
    PPERTENENCIA   = factor(PPERTENENCIA)
  ) %>%
  na.omit()

```
- Se seleccionan solo las variables necesarias para el modelo.
- Se convierten a factor las variables categóricas.
- na.omit() elimina filas con valores faltantes.

#### Entrenamiento del árbol

```r
library(rpart)
library(rpart.plot)

arbol_tratamiento <- rpart(
  TRATA_COMPLEJO ~ EDAD + SEXO + PPERTENENCIA + DIASESTAN,
  data   = data_arbol1,
  method = "class",
  control = rpart.control(
    maxdepth = 4,      # profundidad máxima
    minbucket = 3000,  # mínimo de casos en nodos terminales
    cp = 0.01          # parámetro de complejidad (poda)
  )
)

```
- method = "class" → clasificación.
- Hiperparámetros se configuraron para evitar árboles gigantes y sobreajuste.

#### Gráfica

```r
rpart.plot(
  arbol_tratamiento,
  type          = 2,
  extra         = 104,      # muestra clase + porcentaje + conteo
  under         = TRUE,
  fallen.leaves = TRUE,
  box.palette   = "Blues",
  varlen        = 0,
  faclen        = 0,
  cex           = 0.8,
  main          = "Árbol 1 – Predicción de Tratamiento (Simple vs Complejo)"
)

```

La gráfica muestra en cada nodo:

- La clase predicha (SIMPLE o COMPLEJO).
- El porcentaje de casos por clase.
- El número de observaciones que caen en ese nodo.


### Árbol 2 – Predicción del mes de atención
Predecir el mes (MES) en el que ocurre la atención hospitalaria a partir de variables como edad, pertenencia étnica, dias de estancia y tipo de tratamiento.
#### Preparación 

```r
datos_arbol2 <- datos %>%
  filter(EDAD >= 0, EDAD <= 99) %>%
  mutate(
    MES  = factor(MES),
    SEXO = factor(SEXO),
    PPERTENENCIA = factor(PPERTENENCIA)
  ) %>%
  select(MES, EDAD, SEXO, PPERTENENCIA, DIASESTAN, TRATARECIB) %>%
  na.omit()

```
#### Entrenamiento del árbol

```r
arbol_mes_simple <- rpart(
  MES ~ EDAD + SEXO + PPERTENENCIA + DIASESTAN + TRATARECIB,
  data = datos_arbol2,
  method = "class",
  control = rpart.control(
    maxdepth = 6,
    cp = 0.0002
  )
)
```
#### Gráfica

```r
rpart.plot(
  arbol_mes_simple,
  extra = 2,
  cex   = 0.6,
  main  = "Árbol 2 – Mes de Atención"
)
```

### Árbol 3 – Predicción del rango de edad
Clasificar a cada paciente en un rango etario: 

- NIÑO
- ADULTO_JOVEN
- ADULTO
- ADULTO_MAYOR

#### Preparación 

```r
datos_edad <- datos %>%
  filter(EDAD >= 0, EDAD <= 99) %>%
  mutate(
    EDAD_RANGO = case_when(
      EDAD < 15 ~ "NIÑO",
      EDAD < 40 ~ "ADULTO_JOVEN",
      EDAD < 65 ~ "ADULTO",
      TRUE ~ "ADULTO_MAYOR"
    ),
    EDAD_RANGO  = factor(EDAD_RANGO),
    SEXO        = factor(SEXO),
    PPERTENENCIA = factor(PPERTENENCIA),
    TRATARECIB  = factor(TRATARECIB)
  ) %>%
  select(EDAD_RANGO, DIASESTAN, SEXO, PPERTENENCIA, TRATARECIB) %>%
  na.omit()
```
#### Entrenamiento del árbol

```r
arbol_edad <- rpart(
  EDAD_RANGO ~ DIASESTAN + SEXO + PPERTENENCIA + TRATARECIB,
  data = datos_edad,
  method = "class"
)
```
#### Gráfica

```r
rpart.plot(
  arbol_edad,
  type = 2,
  extra = 2,
  cex = 0.7,
  main = "Árbol 3 – Predicción del Rango de Edad"
)
```

### Árbol 4 – Predicción del tipo de procedimiento (TRATARECIB 1, 2, 3)
Predecir el tipo de tratamiento recibido (código directo en TRATARECIB).

#### Preparación 

```r
datos_tratatipo <- datos %>%
  filter(
    DIASESTAN >= 1, DIASESTAN <= 98,
    EDAD >= 0, EDAD <= 99
  ) %>%
  mutate(
    TRATARECIB   = factor(TRATARECIB),
    SEXO         = factor(SEXO),
    PPERTENENCIA = factor(PPERTENENCIA)
  ) %>%
  select(
    TRATARECIB,
    EDAD,
    SEXO,
    PPERTENENCIA,
    DIASESTAN
  ) %>%
  na.omit()

```
#### Entrenamiento del árbol

```r
arbol_tratatipo <- rpart(
  TRATARECIB ~ EDAD + SEXO + PPERTENENCIA + DIASESTAN,
  data = datos_tratatipo,
  method = "class"
)

```
#### Gráfica

```r
rpart.plot(
  arbol_tratatipo,
  type = 2,
  extra = 2,
  cex   = 0.7,
  main  = "Árbol 4 – Predicción de TRATARECIB (1, 2, 3)"
)

```

## 7. Bosques Aleatorios (Random Forest) en R

### RF1 – Predicción de tipo de tratamiento (SIMPLE vs COMPLEJO)

#### Preparación de datos
```r
datos_rf1 <- datos %>%
  filter(
    DIASESTAN >= 1, DIASESTAN <= 98,
    EDAD >= 0, EDAD <= 99
  ) %>%
  mutate(
    TRATA_TIPO = case_when(
      TRATARECIB == 1 ~ "SIMPLE",
      TRATARECIB %in% c(2,3) ~ "COMPLEJO",
      TRUE ~ NA_character_
    ),
    TRATA_TIPO = factor(TRATA_TIPO),
    SEXO = factor(SEXO),
    PPERTENENCIA = factor(PPERTENENCIA)
  ) %>%
  select(TRATA_TIPO, EDAD, SEXO, DIASESTAN, PPERTENENCIA) %>%
  na.omit()

```
#### Entrenamiento del árbol

```r
bosque1 <- randomForest(
  TRATA_TIPO ~ .,
  data = train1,
  ntree = 200,
  mtry = 3
)
```
La curva muestra el error OOB disminuyendo conforme aumenta el número de árboles hasta estabilizarse.

#### Escenarios

```r
predict(bosque1, esc_rf1_A, type="prob")
```


### RF2 – Predicción del mes de atención

#### Preparación de datos
```r
datos_rf2 <- datos %>%
  filter(EDAD >= 0, EDAD <= 99) %>%
  mutate(
    MES = factor(MES),
    SEXO = factor(SEXO),
    PPERTENENCIA = factor(PPERTENENCIA),
    TRATARECIB = factor(TRATARECIB)
  ) %>%
  select(MES, EDAD, SEXO, DIASESTAN, PPERTENENCIA, TRATARECIB) %>%
  na.omit()

```
#### Entrenamiento del árbol

```r
bosque2 <- randomForest(
  MES ~ .,
  data = train2,
  ntree = 250,
  mtry = 3
)

```
La gráfica muestra el error para una clasificación de 12 categorías, típicamente más alta debido a la complejidad del problema.

#### Escenarios

```r
predict(bosque2, esc_rf2_A, type="prob")
```

### RF3 – Predicción del rango de edad

#### Preparación de datos
```r
datos_rf3 <- datos %>%
  filter(EDAD >= 0, EDAD <= 99) %>%
  mutate(
    EDAD_RANGO = case_when(
      EDAD < 15 ~ "NIÑO",
      EDAD < 40 ~ "ADULTO_JOVEN",
      EDAD < 65 ~ "ADULTO",
      TRUE ~ "ADULTO_MAYOR"
    ),
    EDAD_RANGO = factor(EDAD_RANGO),
    SEXO = factor(SEXO),
    PPERTENENCIA = factor(PPERTENENCIA),
    TRATARECIB = factor(TRATARECIB)
  ) %>%
  select(EDAD_RANGO, DIASESTAN, SEXO, PPERTENENCIA, TRATARECIB) %>%
  na.omit()

```
#### Entrenamiento del árbol

```r
bosque3 <- randomForest(
  EDAD_RANGO ~ .,
  data = train3,
  ntree = 250,
  mtry = 2
)


```
La gráfica muestra la evolución del error OOB para las cuatro clases (NIÑO, ADULTO_JOVEN, ADULTO, ADULTO_MAYOR).

#### Escenarios

```r
predict(bosque3, esc_rf3_A, type="prob")

```

## 8. Redes Neuronales (Python – Google Colab)
Las redes neuronales replican tres modelos de los árboles de decisión:

- RN1 → equivalente al Árbol 1
- RN2 → equivalente al Árbol 3
- RN3 → equivalente al Árbol 4

### Carga de datos en Python


Primero se montó Google Drive en Colab y luego se leyó cada archivo `.xlsx` desde una carpeta específica:

```python
import os
import glob
import pandas as pd

# Ruta en Google Drive donde están los archivos 2018–2024
ruta = "/content/drive/My Drive/Maestria/Mineria de datos/Dataset proyecto/"

# Buscar todos los .xlsx en la carpeta
archivos = glob.glob(os.path.join(ruta, "*.xlsx"))

# Leer cada archivo Excel y concatenar en un solo DataFrame
lista_df = [pd.read_excel(f) for f in archivos]
datos = pd.concat(lista_df, ignore_index=True)
```

Antes de esto, en Colab se debe ejecutar:

```python
from google.colab import drive
drive.mount('/content/drive')

```

para poder acceder a "/content/drive/My Drive/...".

### RN1 – Clasificación del tratamiento SIMPLE vs COMPLEJO

Primero se crea un subconjunto `datos_nn1` a partir del DataFrame unificado `datos`, filtrando rangos válidos de edad y días de estancia:

```python
datos_nn1 = datos.copy()
datos_nn1 = datos_nn1[
    (datos_nn1["EDAD"].between(0, 99)) &
    (datos_nn1["DIASESTAN"].between(1, 98))
]
```

Luego se construye la variable objetivo TRATA_TIPO, derivada de TRATARECIB:

- TRATARECIB == 1 → "SIMPLE"

- TRATARECIB ∈ {2, 3} → "COMPLEJO"
```python
def clasificar_trata_tipo(x):
    if x == 1:
        return "SIMPLE"
    elif x in [2, 3]:
        return "COMPLEJO"
    else:
        return np.nan

datos_nn1["TRATA_TIPO"] = datos_nn1["TRATARECIB"].apply(clasificar_trata_tipo)

# Se eliminan registros sin clase
datos_nn1 = datos_nn1.dropna(subset=["TRATA_TIPO"])

```

Las variables de entrada y salida se definen así:

```python
X1 = datos_nn1[["EDAD", "SEXO", "PPERTENENCIA", "DIASESTAN"]].copy()
y1 = datos_nn1["TRATA_TIPO"].copy()
```

La clase de salida se codifica a binaria (0/1) mediante un diccionario:

```python
mapa_trata1 = {"SIMPLE": 0, "COMPLEJO": 1}
y1_bin = y1.map(mapa_trata1).values

```

Luego se normalizan las variables de entrada con StandardScaler:

```python
scaler1 = StandardScaler()
X1_scaled = scaler1.fit_transform(X1)

```

Y se realiza la división entrenamiento/prueba:

```python
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1_scaled, y1_bin, test_size=0.2, random_state=42
)

```

El modelo RN1 es una red neuronal binaria con dos capas ocultas y una capa de salida con activación sigmoide:


```python
model1 = Sequential()
model1.add(Dense(32, activation='relu', input_dim=X1_train.shape[1]))
model1.add(Dense(16, activation='relu'))
model1.add(Dense(1, activation='sigmoid'))  # salida binaria

model1.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history1 = model1.fit(
    X1_train, y1_train,
    epochs=20,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)
```

La evaluación del modelo se realiza sobre el conjunto de prueba:

```python
loss1, acc1 = model1.evaluate(X1_test, y1_test, verbose=0)
print(f"Precisión RN1 en test: {acc1:.4f}")

```

#### Escenarios de prueba para RN1

Para probar el modelo se definen cuatro escenarios (esc1_A a esc1_D) en forma de DataFrame, usando las mismas variables de entrada que el modelo (EDAD, SEXO, PPERTENENCIA, DIASESTAN):

```python
esc1_A = pd.DataFrame({
    "EDAD": [22],
    "SEXO": [2],
    "PPERTENENCIA": [1],
    "DIASESTAN": [2]
})

esc1_B = pd.DataFrame({
    "EDAD": [22],
    "SEXO": [2],
    "PPERTENENCIA": [2],
    "DIASESTAN": [1]
})

esc1_C = pd.DataFrame({
    "EDAD": [10],
    "SEXO": [2],
    "PPERTENENCIA": [3],
    "DIASESTAN": [1]
})

esc1_D = pd.DataFrame({
    "EDAD": [65],
    "SEXO": [1],
    "PPERTENENCIA": [1],
    "DIASESTAN": [4]
})
```

La función predecir_rn1 aplica el mismo scaler1 utilizado en el entrenamiento y obtiene la probabilidad de COMPLEJO:

```python
def predecir_rn1(escenario_df, nombre):
    esc_scaled = scaler1.transform(escenario_df)
    prob = model1.predict(esc_scaled)[0][0]
    clase = "COMPLEJO" if prob >= 0.5 else "SIMPLE"
    print(f"\nEscenario {nombre} – RN1")
    print(f"Probabilidad de COMPLEJO: {prob:.3f}")
    print(f"Clasificación final: {clase}")

predecir_rn1(esc1_A, "A")
predecir_rn1(esc1_B, "B")
predecir_rn1(esc1_C, "C")
predecir_rn1(esc1_D, "D")

```

### RN2 – Clasificación del rango de edad (EDAD_RANGO)

El segundo modelo replica la idea del Árbol 3 / RF3, donde la edad se agrupa en rangos:

- NIÑO: EDAD < 15
- ADULTO_JOVEN: 15 ≤ EDAD < 40
- ADULTO: 40 ≤ EDAD < 65
- ADULTO_MAYOR: EDAD ≥ 65

Primero se filtra el subconjunto datos_nn2:

```python
datos_nn2 = datos.copy()
datos_nn2 = datos_nn2[
    (datos_nn2["EDAD"].between(0, 99)) &
    (datos_nn2["DIASESTAN"].between(1, 98))
]

```

Luego se crea la columna EDAD_RANGO:

```python
def rango_edad(e):
    if e < 15:
        return "NIÑO"
    elif e < 40:
        return "ADULTO_JOVEN"
    elif e < 65:
        return "ADULTO"
    else:
        return "ADULTO_MAYOR"

datos_nn2["EDAD_RANGO"] = datos_nn2["EDAD"].apply(rango_edad)
```

Se eliminan filas con valores faltantes en las variables clave usadas como entradas:
```python
datos_nn2 = datos_nn2.dropna(subset=["EDAD_RANGO", "SEXO", "PPERTENENCIA", "TRATARECIB", "DIASESTAN"])

```

Las entradas y la salida se definen como:
```python
X2 = datos_nn2[["DIASESTAN", "SEXO", "PPERTENENCIA", "TRATARECIB"]].copy()
y2 = datos_nn2["EDAD_RANGO"].copy()


```

La salida se codifica a enteros usando un diccionario generado a partir de las clases presentes:
```python
clases_rango = sorted(y2.unique())  # e.g. ['ADULTO', 'ADULTO_JOVEN', 'ADULTO_MAYOR', 'NIÑO']
mapa_rango = {clase: idx for idx, clase in enumerate(clases_rango)}
y2_int = y2.map(mapa_rango).values

y2_cat = to_categorical(y2_int)  # one-hot


```

Las entradas se normalizan con StandardScaler:

```python
scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(X2)


```

Y se separan en entrenamiento y prueba:

```python
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2_scaled, y2_cat, test_size=0.2, random_state=42
)
```

RN2 es un modelo multiclase con softmax:

```python
model2 = Sequential()
model2.add(Dense(32, activation='relu', input_dim=X2_train.shape[1]))
model2.add(Dense(16, activation='relu'))
model2.add(Dense(y2_cat.shape[1], activation='softmax'))

model2.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model2.fit(
    X2_train, y2_train,
    epochs=25,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

```
#### Evaluacion 
```python
loss2, acc2 = model2.evaluate(X2_test, y2_test, verbose=0)
print(f"Precisión RN2 en test: {acc2:.4f}")
print("Clases (índice → etiqueta):", mapa_rango)


```

#### Escenarios de prueba para RN2

En este modelo la entrada son solo variables de estancia, sexo, pertenencia étnica y tipo de tratamiento:

```python
esc2_A = pd.DataFrame({
    "DIASESTAN": [1],
    "SEXO": [2],
    "PPERTENENCIA": [1],
    "TRATARECIB": [1]
})

esc2_B = pd.DataFrame({
    "DIASESTAN": [3],
    "SEXO": [2],
    "PPERTENENCIA": [2],
    "TRATARECIB": [2]
})

esc2_C = pd.DataFrame({
    "DIASESTAN": [5],
    "SEXO": [1],
    "PPERTENENCIA": [1],
    "TRATARECIB": [3]
})

```

La función de predicción muestra las probabilidades por clase y la etiqueta final:

```python
def predecir_rn2(escenario_df, nombre):
    esc_scaled = scaler2.transform(escenario_df)
    prob = model2.predict(esc_scaled)[0]
    idx = np.argmax(prob)
    inv_mapa = {v: k for k, v in mapa_rango.items()}
    clase = inv_mapa[idx]
    print(f"\nEscenario {nombre} – RN2")
    print("Probabilidades por clase:")
    for i, p in enumerate(prob):
        print(f"  {inv_mapa[i]}: {p:.3f}")
    print(f"Clasificación final: {clase}")

predecir_rn2(esc2_A, "A")
predecir_rn2(esc2_B, "B")
predecir_rn2(esc2_C, "C")

```


### RN3 – Clasificación de TRATARECIB (1, 2, 3)

Para el tercer modelo se toma nuevamente datos y se filtra el rango válido de edad y días de estancia, además de quedarse solo con tratamientos 1, 2 y 3:

```python
datos_nn3 = datos.copy()
datos_nn3 = datos_nn3[
    (datos_nn3["EDAD"].between(0, 99)) &
    (datos_nn3["DIASESTAN"].between(1, 98))
]

datos_nn3 = datos_nn3[datos_nn3["TRATARECIB"].isin([1, 2, 3])]
datos_nn3 = datos_nn3.dropna(subset=["EDAD", "SEXO", "PPERTENENCIA", "DIASESTAN", "TRATARECIB"])

```

Las variables de entrada y la salida:
```python
X3 = datos_nn3[["EDAD", "SEXO", "PPERTENENCIA", "DIASESTAN"]].copy()
y3 = datos_nn3["TRATARECIB"].copy()

```

Se mapearon las clases de TRATARECIB a índices 0, 1, 2:

```python
clases_trata = sorted(y3.unique())  # [1, 2, 3]
mapa_trata3 = {clase: idx for idx, clase in enumerate(clases_trata)}
y3_int = y3.map(mapa_trata3).values
y3_cat = to_categorical(y3_int)
```

### Normalización:

```python
scaler3 = StandardScaler()
X3_scaled = scaler3.fit_transform(X3)
```

### División train/test:


```python
X3_train, X3_test, y3_train, y3_test = train_test_split(
    X3_scaled, y3_cat, test_size=0.2, random_state=42
)
```
### Definición del modelo model3

```python
model3 = Sequential()
model3.add(Dense(32, activation='relu', input_dim=X3_train.shape[1]))
model3.add(Dense(16, activation='relu'))
model3.add(Dense(y3_cat.shape[1], activation='softmax'))

model3.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history3 = model3.fit(
    X3_train, y3_train,
    epochs=25,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)
```

#### Evaluacion 
```python
loss3, acc3 = model3.evaluate(X3_test, y3_test, verbose=0)
print(f"Precisión RN3 en test: {acc3:.4f}")
print("Clases (índice → TRATARECIB):", mapa_trata3)
```

#### Escenarios de prueba para RN3

Se definieron cuatro escenarios con distintas combinaciones de edad, sexo, pertenencia y días de estancia:

```python
esc3_A = pd.DataFrame({
    "EDAD": [22],
    "SEXO": [2],
    "PPERTENENCIA": [1],
    "DIASESTAN": [1]
})

esc3_B = pd.DataFrame({
    "EDAD": [50],
    "SEXO": [1],
    "PPERTENENCIA": [2],
    "DIASESTAN": [4]
})

esc3_C = pd.DataFrame({
    "EDAD": [10],
    "SEXO": [2],
    "PPERTENENCIA": [3],
    "DIASESTAN": [2]
})

esc3_D = pd.DataFrame({
    "EDAD": [70],
    "SEXO": [1],
    "PPERTENENCIA": [1],
    "DIASESTAN": [7]
})

```

Función de predicción:
```python
def predecir_rn3(escenario_df, nombre):
    esc_scaled = scaler3.transform(escenario_df)
    prob = model3.predict(esc_scaled)[0]
    idx = np.argmax(prob)
    inv_mapa = {v: k for k, v in mapa_trata3.items()}
    clase_trata = inv_mapa[idx]
    print(f"\nEscenario {nombre} – RN3")
    print("Probabilidades por tipo de tratamiento (TRATARECIB):")
    for i, p in enumerate(prob):
        print(f"  {inv_mapa[i]}: {p:.3f}")
    print(f"Clasificación final TRATARECIB: {clase_trata}")

predecir_rn3(esc3_A, "A")
predecir_rn3(esc3_B, "B")
predecir_rn3(esc3_C, "C")
predecir_rn3(esc3_D, "D")

```














## 9. Implementación en otros ambientes 

### R
### Windows

Cambiar la ruta:
```r
ruta <- "C:/Users/TuUsuario/data_hospitalaria/"
```

### Linux / macOS

Cambiar la ruta:
```r
ruta <- "/home/data_hospitalaria/"
```
### Google Colab

Cambiar la ruta:
```python
/content/data_hospitalaria/
```

### Redes neuronales

Colocar los archivos .xlsx (2018–2024) en una carpeta local, por ejemplo:


```text
./Dataset_proyecto/
```

Sustituir en el código la ruta de Colab:

```text
ruta = "/content/drive/My Drive/Maestria/Mineria de datos/Dataset proyecto/"
```

por algo como:

```text
ruta = "./Dataset_proyecto/"

```

#### Ejecución local o en Colab

- En Google Colab, mantener el montaje de Drive y la ruta original.

- En ambiente local (VS Code, PyCharm, etc.):

  - Eliminar el bloque drive.mount(...).

  - Asegurarse de que ruta apunte a la carpeta donde están los .xlsx.


#### Orden recomendado de ejecución

- Cargar y unificar los datos en el DataFrame datos.
- Ejecutar los bloques de RN1, RN2 y RN3 en ese orden.
- Revisar:
  - Las métricas de precisión en prueba (acc1, acc2, acc3).
  - Las probabilidades y clases resultantes en los escenarios de prueba.

---

## 10. Licencia y créditos

- **Autor:** Edgar Rolando Ramirez Lopez
- **Universidad:** Universidad San Carlos de Guatemala
- **Curso:** Minería de Datos
- **Licencia:** Uso académico libre.

---
