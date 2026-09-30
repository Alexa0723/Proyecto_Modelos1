# Fase 1 – Modelo Predictivo de Abandono de Clientes (Churn)

Modelos y Simulación de Sistemas I – Proyecto Integrador

**Integrantes:**
- Shara Olaya Araque
- Liceth Alexandra Sarmiento Campos

## Descripción del problema

Las empresas de telecomunicaciones pierden ingresos cuando sus clientes
cancelan el servicio. Identificar con anticipación a los clientes con mayor
riesgo de abandono permite a la empresa aplicar acciones de retención antes
de que el abandono ocurra.

Se trata de un problema de **clasificación binaria supervisada**, cuya
variable objetivo es `Churn`: indica si el cliente abandonó el servicio
(`Yes`) o permaneció (`No`).

## Fuente de datos

Dataset **Telco Customer Churn**, disponible en Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Contiene 7043 clientes y 21 variables (datos demográficos, servicios
contratados, información de la cuenta y facturación). El 26,54 % de los
clientes abandonó el servicio. Una copia del archivo se incluye en
`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Objetivo

Construir un modelo de Machine Learning que, a partir de las
características de un cliente, prediga si abandonará el servicio, y que
supere a un modelo base que no aprende ningún patrón.

## Metodología

1. Análisis exploratorio de datos (EDA).
2. Tratamiento de valores faltantes: 11 valores vacíos en `TotalCharges`,
   correspondientes a clientes nuevos, imputados con 0.
3. División estratificada en entrenamiento (80 %) y prueba (20 %).
4. Preprocesamiento con `Pipeline` y `ColumnTransformer` (imputación,
   `StandardScaler` y `OneHotEncoder`), ajustado únicamente con los datos de
   entrenamiento para evitar la fuga de información.
5. Modelo base: `DummyClassifier` (clase más frecuente).
6. Modelo predictivo seleccionado mediante validación cruzada sobre el
   conjunto de entrenamiento.
7. Evaluación final en el conjunto de prueba.

## Algoritmo utilizado

**Regresión logística** con balanceo de clases (`class_weight="balanced"`).
Se eligió por ser un modelo estándar para clasificación binaria,
interpretable mediante sus coeficientes y con regularización incorporada.
El balanceo de clases se seleccionó mediante validación cruzada de 5
particiones, por obtener un mayor F1 promedio.

## Métrica utilizada

- **Métrica principal:** F1-score de la clase "abandona", porque equilibra
  la detección de clientes que abandonan (recall) con la confiabilidad de
  las alertas (precision).
- **Métricas complementarias:** recall, precision y ROC-AUC.
- **Accuracy:** solo como referencia, ya que es engañosa con clases
  desbalanceadas.

## Principales resultados

Resultados en el conjunto de prueba (1409 clientes):

| Métrica   | Modelo base | Modelo predictivo |
|-----------|-------------|-------------------|
| F1-score  | 0,0000      | **0,6136**        |
| Recall    | 0,0000      | **0,7834**        |
| Precision | 0,0000      | **0,5043**        |
| ROC-AUC   | 0,5000      | **0,8415**        |
| Accuracy  | 0,7346      | 0,7381            |

- El modelo detecta aproximadamente **8 de cada 10 clientes que abandonan**.
- No presenta sobreajuste: F1 de 0,6337 en entrenamiento, 0,6293 en
  validación cruzada y 0,6136 en prueba.
- Los factores más asociados al abandono son la baja antigüedad, el contrato
  mensual, el servicio de fibra óptica y el pago con cheque electrónico.

## Estructura de la carpeta

```
fase-1/
├── notebook.ipynb       # Desarrollo completo del proyecto
├── modelo.joblib        # Modelo entrenado (Pipeline completo)
├── requirements.txt     # Librerías y versiones utilizadas
├── README.md            # Este archivo
└── data/
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Instrucciones para ejecutar el notebook

### Opción 1: Google Colab (recomendada)

1. Abrir https://colab.research.google.com
2. **Archivo → Abrir cuaderno → GitHub**, buscar el repositorio y abrir
   `fase-1/notebook.ipynb` en la rama `main`.
3. **Entorno de ejecución → Ejecutar todas.**

El dataset se carga automáticamente desde el repositorio, por lo que no es
necesario subir archivos.

### Opción 2: Ejecución local

```bash
git clone <https://github.com/Alexa0723/Proyecto_Modelos1.git>
cd <nombre-del-repositorio>/fase-1
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

Luego, ejecutar todas las celdas en orden. Se requiere conexión a internet
para cargar el dataset.

### Uso del modelo guardado

```python
import joblib

modelo = joblib.load("modelo.joblib")
probabilidades = modelo.predict_proba(datos_clientes)[:, 1]
predicciones = modelo.predict(datos_clientes)
```

`datos_clientes` debe ser un DataFrame con las 19 variables predictoras
(todas las columnas del dataset excepto `customerID` y `Churn`), con
`TotalCharges` en formato numérico; los valores vacíos deben representarse
como `NaN`. El modelo debe cargarse con **scikit-learn 1.6.1** para
garantizar la compatibilidad.

## Reproducibilidad

Todos los procesos aleatorios utilizan `random_state = 42`, por lo que el
notebook produce los mismos resultados en cada ejecución.
