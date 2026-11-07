# Fase 2 – Scripts y Docker: Modelo de Abandono de Clientes (Churn)

**Asignatura:** Modelos y Simulación de Sistemas I – Proyecto Integrador

**Integrantes:**
- Shara Olaya Araque
- Liceth Alexandra Sarmiento

**Versión entregada:** etiqueta de Git `entrega-2` (rama `main`).

## 1. Problema y relación con la Fase 1

Se predice si un cliente de una empresa de telecomunicaciones abandonará el
servicio (`Churn`: `Yes`/`No`), a partir de sus datos demográficos, servicios
contratados e información de facturación. Es un problema de
**clasificación binaria supervisada**.

En la Fase 1 ([`../fase-1`](../fase-1)) el modelo se desarrolló en un notebook.
En esta fase el mismo proceso se convirtió en una aplicación reutilizable:

- `train.py` entrena el modelo y guarda una versión identificable.
- `predict.py` carga una versión guardada y genera predicciones sobre un CSV
  nuevo, sin reentrenar.
- Ambos scripts se pueden ejecutar localmente o dentro de un contenedor Docker.

Se conservan el problema, los datos, el preprocesamiento, el algoritmo
(regresión logística balanceada) y la métrica principal (F1 de la clase
"abandona"). Con los mismos datos, partición y semilla, la versión `v1`
reproduce exactamente los resultados de la Fase 1.

## 2. Cambios y correcciones respecto a la Fase 1

Al momento de esta entrega no se había recibido retroalimentación de la
Fase 1. Se aplicaron las siguientes correcciones técnicas identificadas por
el equipo:

| Cambio | Motivo |
|---|---|
| La conversión de `TotalCharges` de texto a número pasó a un módulo compartido (`src/datos.py`), usado por el entrenamiento y la predicción | En la Fase 1 se hacía fuera del Pipeline, por lo que el modelo guardado no podía recibir los datos en su formato original |
| Validación de tipos: un texto no numérico en una columna numérica produce un error claro | Antes se convertía en valor faltante sin avisar |
| Imputación por la categoría más frecuente en variables categóricas | Definir una estrategia para celdas categóricas vacías en datos nuevos (no altera los resultados con los datos actuales) |
| El código del notebook se reorganizó en módulos (`src/`) | Reutilizar la lógica en entrenamiento, predicción, pruebas y la futura API |
| Se fijó `scipy==1.14.1` en `requirements.txt` | La versión instalada por defecto era incompatible con scikit-learn 1.6.1 y generaba advertencias |
| En el `Dockerfile`, pip usa mayor tiempo de espera y reintentos | Evitar fallos de la construcción por conexiones lentas |

Ninguno de estos cambios modifica el preprocesamiento aprendido, la
validación ni el modelo: las métricas de `v1` son iguales a las de la Fase 1.

## 3. Estructura

```
fase-2/
├── train.py                 # Punto de entrada: entrenamiento
├── predict.py               # Punto de entrada: predicción
├── src/                     # Lógica compartida
│   ├── config.py            # Columnas y valores por defecto
│   ├── errores.py           # Errores propios con mensajes claros
│   ├── datos.py             # Lectura, validación y limpieza de datos
│   ├── modelo.py            # Preprocesamiento, modelos, validación cruzada y evaluación
│   ├── almacenamiento.py    # Guardado y carga de versiones
│   └── prediccion.py        # Generación y escritura de predicciones
├── tests/                   # Pruebas con pytest
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset completo
│   └── ejemplo_prediccion.csv                  # Ejemplo para predecir (sin Churn)
├── models/                  # Versiones del modelo
│   ├── v1/
│   ├── v2/
│   └── v-docker/
├── outputs/                 # Predicciones generadas (se crea al ejecutar; no se sube a Git)
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── pytest.ini
```

## 4. Entorno

- **Python:** 3.13 (probado localmente con 3.13.15 y 3.13.5) · imagen Docker
  `python:3.13-slim`.
- **Sistema y terminal usados:** Windows con **Git Bash**.

Desde la carpeta `fase-2`:

```bash
python -m venv .venv
source .venv/Scripts/activate      # En Linux o macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Dependencias (`requirements.txt`): numpy 2.1.3, pandas 2.2.3, scipy 1.14.1,
scikit-learn 1.6.1, joblib 1.6.0 y pytest 8.3.5. Los modelos guardados deben
cargarse con **scikit-learn 1.6.1**.

## 5. Datos

- **Fuente:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
- **Ubicación:** `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`. El archivo ya
  está incluido en el repositorio (menos de 1 MB), por lo que no se requiere
  cuenta de Kaggle ni descarga adicional.
- **Contenido:** 7043 clientes y 21 columnas; variable objetivo `Churn`.

### Esquema del CSV de predicción

El CSV de entrada debe tener estas 19 columnas (en cualquier orden).
`Churn` **no es necesaria**. Ejemplo: `data/ejemplo_prediccion.csv`.

| Columna | Tipo esperado | Valores |
|---|---|---|
| `tenure` | numérico | meses como cliente |
| `MonthlyCharges` | numérico | cargo mensual |
| `TotalCharges` | numérico | cargo total; puede estar vacío |
| `SeniorCitizen` | numérico 0/1 | 1 = adulto mayor |
| `gender` | texto | `Female`, `Male` |
| `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` | texto | `Yes`, `No` |
| `MultipleLines` | texto | `Yes`, `No`, `No phone service` |
| `InternetService` | texto | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | texto | `Yes`, `No`, `No internet service` |
| `Contract` | texto | `Month-to-month`, `One year`, `Two year` |
| `PaymentMethod` | texto | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` |

La columna `customerID` es opcional: no se usa para predecir, pero si está
presente se copia a la salida para identificar cada fila.

## 6. Entrenamiento

Consultar las opciones:

```bash
python train.py --help
```

Entrenar una versión (en una sola línea):

```bash
python train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --target Churn --output-dir models --version v1 --seed 42
```

| Argumento | Descripción | Por defecto |
|---|---|---|
| `--data` | CSV con datos etiquetados | obligatorio |
| `--target` | Columna objetivo (`Yes`/`No`) | `Churn` |
| `--output-dir` | Carpeta de las versiones | `models` |
| `--version` | Identificador único de la versión | obligatorio |
| `--seed` | Semilla aleatoria | `42` |
| `--test-size` | Proporción de prueba | `0.2` |
| `--cv-folds` | Particiones de la validación cruzada | `5` |

**Proceso:** validación de datos → división estratificada en entrenamiento y
prueba → selección de `class_weight` con validación cruzada **solo sobre
entrenamiento** (el preprocesamiento se ajusta dentro de cada partición
gracias al Pipeline) → entrenamiento → evaluación del modelo y del modelo
base con la misma partición y métrica → guardado de la versión.

**Salidas por versión** (`models/<version>/`):

- `modelo.joblib`: Pipeline completo (preprocesamiento ajustado + modelo).
- `metrics.json`: métricas en prueba del modelo y del modelo base, matrices
  de confusión y resultados de la validación cruzada.
- `metadata.json`: versión, fecha, tipo de problema, variable objetivo,
  algoritmo, variables y tipos, parámetros, semilla, datos (fuente, archivo,
  huella SHA-256 y cantidad), separación, métrica, versión de Python y hash
  del commit.

**Versionamiento:** si el identificador ya existe, el entrenamiento se
rechaza con un error y la versión existente no se modifica.

## 7. Predicción

```bash
python predict.py --help
python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
```

`--model` acepta el archivo `modelo.joblib` o la carpeta de la versión
(`models/v1`, `models/v2`, `models/v-docker`).

**Salida (CSV):**

| Columna | Significado |
|---|---|
| `customerID` | Identificador del cliente (si existe en la entrada) |
| `fila_entrada` | Número de fila de datos en el CSV de entrada (1 = primera fila) |
| `prediccion` | **1 = el cliente abandona; 0 = el cliente permanece** |
| `etiqueta` | `abandona` o `permanece` |
| `probabilidad_abandono` | Probabilidad estimada de abandono |
| `version_modelo` | Versión del modelo utilizada |

**Reglas de validación:**

| Situación | Comportamiento |
|---|---|
| Falta una columna requerida | Error claro, código de salida 1 y no se genera archivo |
| Texto no numérico en una columna numérica | Error que indica la columna y la fila del CSV, código de salida 1 y no se genera archivo |
| Celdas vacías | Se aplica la imputación de la versión: mediana (`tenure`, `MonthlyCharges`), 0 (`TotalCharges`) o categoría más frecuente (categóricas) |
| Columnas adicionales | Se ignoran y se informan |
| Categorías no vistas en el entrenamiento | Se predice (sus columnas codificadas quedan en 0) y se informa |
| Modelo inexistente | Error claro y código de salida 1 |

Se genera una predicción por cada fila, en el mismo orden de la entrada.
El archivo de salida se escribe solo al final y de forma segura, por lo que
un proceso fallido no deja un archivo que aparente contener resultados.

## 8. Docker

Comandos ejecutados desde `fase-2` en **Git Bash (Windows)**.
`MSYS_NO_PATHCONV=1` evita que Git Bash modifique las rutas del contenedor,
y `$(pwd -W)` entrega la ruta de la carpeta actual en formato Windows.
En Linux o macOS se omite `MSYS_NO_PATHCONV=1` y se usa `$PWD` en lugar de
`$(pwd -W)`.

**Construir la imagen** (no entrena el modelo):

```bash
docker build -t modelos-fase2:1.0 .
```

**Entrenar dentro del contenedor:**

```bash
mkdir -p models outputs

MSYS_NO_PATHCONV=1 docker run --rm \
  -v "$(pwd -W)/data:/app/data:ro" \
  -v "$(pwd -W)/models:/app/models" \
  -e GIT_COMMIT="$(git rev-parse HEAD)" \
  modelos-fase2:1.0 python train.py \
  --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --target Churn \
  --output-dir models --version v-docker --seed 42
```

**Predecir dentro del contenedor:**

```bash
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "$(pwd -W)/data:/app/data:ro" \
  -v "$(pwd -W)/models:/app/models:ro" \
  -v "$(pwd -W)/outputs:/app/outputs" \
  modelos-fase2:1.0 python predict.py \
  --model models/v-docker/modelo.joblib \
  --input data/ejemplo_prediccion.csv \
  --output outputs/predicciones_docker.csv
```

**Montajes:**

- `data` (solo lectura): datos de entrada.
- `models`: las versiones entrenadas se guardan en el equipo anfitrión
  (solo lectura al predecir).
- `outputs`: las predicciones se guardan en el equipo anfitrión.

Con `--rm` el contenedor se elimina al terminar, y los modelos y
predicciones **persisten** en el equipo. Como la carpeta `.git` no se copia
a la imagen, el hash del commit se pasa con `-e GIT_COMMIT=...` para
registrarlo en `metadata.json`.

## 9. Pruebas

Desde `fase-2`:

```bash
python -m pytest -v
```

| Prueba | Comportamiento verificado |
|---|---|
| `test_entrenamiento_genera_artefacto_metricas_y_metadatos` | Se genera un modelo cargable, métricas (modelo y baseline) y metadatos |
| `test_modelo_cargado_conserva_las_predicciones` | El modelo cargado conserva las predicciones; una por fila, en orden |
| `test_valores_faltantes_se_procesan_segun_la_estrategia` | `TotalCharges` vacío se imputa con 0; una categórica vacía se procesa |
| `test_columna_faltante_produce_error_claro` | Una columna ausente produce un error con su nombre |
| `test_predict_cli_con_columna_faltante_termina_con_error` | `predict.py` termina con código distinto de cero y sin archivo de salida |
| `test_nueva_version_conserva_la_anterior_y_no_sobrescribe` | Dos versiones coexisten y se cargan; un identificador repetido no modifica la existente |
| `test_tipo_incompatible_produce_error_claro` | Un texto en una columna numérica produce un error claro |
| `test_orden_de_columnas_no_afecta_la_prediccion` | El orden de las columnas del CSV no cambia las predicciones |
| `test_categoria_no_vista_se_informa_y_se_predice` | Una categoría nueva se predice y se informa |

Las pruebas usan una muestra de 300 filas del dataset del repositorio (sin
descargas) y guardan todo en carpetas temporales, sin modificar `models/`.

## 10. Resultados

Métrica principal: **F1 de la clase "abandona"**. Resultados en el conjunto
de prueba:

| Versión | Semilla | Configuración | F1 | Recall | Precision | ROC-AUC | F1 modelo base |
|---|---|---|---|---|---|---|---|
| `v1` | 42 | balanceado | 0,6136 | 0,7834 | 0,5043 | 0,8415 | 0,0000 |
| `v2` | 2026 | balanceado | 0,6246 | 0,8075 | 0,5093 | 0,8399 | 0,0000 |
| `v-docker` | 42 | balanceado | 0,6136 | 0,7834 | 0,5043 | 0,8415 | 0,0000 |

- El modelo supera ampliamente al modelo base (`DummyClassifier`), que no
  detecta ningún abandono (F1 = 0) en ninguna de las versiones.
- `v1` reproduce exactamente los resultados de la Fase 1, y `v-docker`
  (misma semilla, entrenada en Docker) obtiene los mismos valores.
- `v2` usa otra semilla y, por lo tanto, otra partición de los datos. Sus
  métricas son similares a las de `v1`; la pequeña diferencia refleja la
  variabilidad según la partición y no indica que un modelo sea mejor que el
  otro, ya que fueron evaluados con conjuntos de prueba distintos.

### Limitaciones conocidas

- El dataset corresponde a un único momento en el tiempo y no incluye
  fechas.
- Con el balanceo de clases, aproximadamente la mitad de las alertas de
  abandono son falsas alarmas (precision ≈ 0,50).
- El umbral de decisión es 0,5 y no se ajustó a costos reales de negocio.
- El valor exacto de las métricas puede variar levemente entre sistemas por
  diferencias numéricas; con el mismo entorno, datos y semilla los
  resultados son consistentes.

## 11. Evidencia de ejecución

**Entrenamiento de `v1` (local):**

```
Versión 'v1' entrenada y guardada correctamente.
Configuración elegida: balanceado

Métrica        Modelo base        Modelo
accuracy            0.7346        0.7381
precision           0.0000        0.5043
recall              0.0000        0.7834
f1                  0.0000        0.6136
roc_auc             0.5000        0.8415
```

**Entrenamiento de `v2` (local):**

```
Versión 'v2' entrenada y guardada correctamente.
Configuración elegida: balanceado

Métrica        Modelo base        Modelo
accuracy            0.7346        0.7424
precision           0.0000        0.5093
recall              0.0000        0.8075
f1                  0.0000        0.6246
roc_auc             0.5000        0.8399
```

**Predicción con `v1` y `v2`** (ambas versiones se cargan y predicen):

```
customerID,fila_entrada,prediccion,etiqueta,probabilidad_abandono,version_modelo
EJEMPLO-001,1,1,abandona,0.8951,v1
EJEMPLO-002,2,0,permanece,0.02,v1
EJEMPLO-003,3,0,permanece,0.2496,v1
EJEMPLO-004,4,1,abandona,0.831,v1
EJEMPLO-005,5,0,permanece,0.0875,v1

customerID,fila_entrada,prediccion,etiqueta,probabilidad_abandono,version_modelo
EJEMPLO-001,1,1,abandona,0.9036,v2
EJEMPLO-002,2,0,permanece,0.017,v2
EJEMPLO-003,3,0,permanece,0.256,v2
EJEMPLO-004,4,1,abandona,0.8345,v2
EJEMPLO-005,5,0,permanece,0.0746,v2
```

El cliente `EJEMPLO-003` tiene `TotalCharges` vacío y recibe predicción
gracias a la imputación incluida en el modelo.

**Errores previsibles:**

```
$ python train.py ... --version v1 ...
ERROR: La versión 'v1' ya existe en models\v1. No se sobrescribe una versión existente: use otro identificador con --version.
Código de salida: 1

$ python predict.py --model models/v1 --input outputs/sin_tenure.csv ...
ERROR: Faltan columnas requeridas en el archivo: tenure
Código de salida: 1

$ python predict.py --model models/v1 --input outputs/tipo_invalido.csv ...
ERROR: La columna 'MonthlyCharges' debe ser numérica, pero contiene valores no numéricos en las filas del CSV: 3.
Código de salida: 1
```

**Docker:** construcción de la imagen sin entrenamiento; entrenamiento de
`v-docker` con las mismas métricas de `v1`; predicción con `v-docker` y con
`v1` con las mismas probabilidades; tras eliminar el contenedor
(`docker ps -a` vacío), `models/v-docker` y `outputs/predicciones_docker.csv`
permanecen en el equipo. Una entrada sin la columna `tenure` produce
`ERROR: Faltan columnas requeridas en el archivo: tenure` con código de
salida 1 dentro del contenedor.

**Pruebas:**

```
platform win32 -- Python 3.13.5, pytest-8.3.5, pluggy-1.7.0
rootdir: C:\proyectos\Proyecto_Modelos1\fase-2
configfile: pytest.ini
testpaths: tests
collected 9 items

tests/test_aplicacion.py::test_entrenamiento_genera_artefacto_metricas_y_metadatos PASSED [ 11%]
tests/test_aplicacion.py::test_modelo_cargado_conserva_las_predicciones PASSED            [ 22%]
tests/test_aplicacion.py::test_valores_faltantes_se_procesan_segun_la_estrategia PASSED   [ 33%]
tests/test_aplicacion.py::test_columna_faltante_produce_error_claro PASSED                [ 44%]
tests/test_aplicacion.py::test_predict_cli_con_columna_faltante_termina_con_error PASSED  [ 55%]
tests/test_aplicacion.py::test_tipo_incompatible_produce_error_claro PASSED               [ 66%]
tests/test_aplicacion.py::test_nueva_version_conserva_la_anterior_y_no_sobrescribe PASSED [ 77%]
tests/test_aplicacion.py::test_orden_de_columnas_no_afecta_la_prediccion PASSED           [ 88%]
tests/test_aplicacion.py::test_categoria_no_vista_se_informa_y_se_predice PASSED          [100%]

9 passed in 7.74s
```

## 12. Fuentes consultadas

- Dataset: [Telco Customer Churn – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
- Documentación oficial de scikit-learn (`Pipeline`, `ColumnTransformer`,
  `LogisticRegression`, `cross_validate`), pandas, joblib, `argparse`,
  pytest y Docker.
- Se utilizó Claude (asistente de inteligencia artificial) como
  apoyo para la explicación de conceptos, la organización y verificación del código.
  
No se incorporó código externo adicional más allá de los ejemplos de uso de
la documentación oficial.