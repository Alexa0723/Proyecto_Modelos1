"""
Construcción, selección y evaluación de los modelos.

Contiene el preprocesador (el mismo de la Fase 1), el modelo base, el modelo
predictivo, la selección de class_weight con validación cruzada y la
función de evaluación.
"""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import COLUMNA_CARGOS_TOTALES, COLUMNAS_CATEGORICAS, COLUMNAS_NUMERICAS

MAXIMO_ITERACIONES = 1000

CONFIGURACIONES_CLASS_WEIGHT = {
    "sin_balanceo": None,
    "balanceado": "balanced",
}

METRICAS_VALIDACION = ["f1", "recall", "precision", "roc_auc"]


def construir_preprocesador():
    """Construye el preprocesador (sin ajustar). Solo aprende cuando se llama a fit."""
    transformador_numerico = Pipeline(steps=[
        ("imputacion", SimpleImputer(strategy="median")),
        ("escalamiento", StandardScaler()),
    ])

    transformador_cargos_totales = Pipeline(steps=[
        ("imputacion", SimpleImputer(strategy="constant", fill_value=0)),
        ("escalamiento", StandardScaler()),
    ])

    transformador_categorico = Pipeline(steps=[
        ("imputacion", SimpleImputer(strategy="most_frequent")),
        ("codificacion", OneHotEncoder(handle_unknown="ignore",
                                       drop="if_binary",
                                       sparse_output=False)),
    ])

    preprocesador = ColumnTransformer(transformers=[
        ("numericas", transformador_numerico, COLUMNAS_NUMERICAS),
        ("cargos_totales", transformador_cargos_totales, COLUMNA_CARGOS_TOTALES),
        ("categoricas", transformador_categorico, COLUMNAS_CATEGORICAS),
    ])

    return preprocesador


def construir_modelo_predictivo(class_weight, semilla):
    """Construye el Pipeline completo: preprocesador + regresión logística."""
    modelo = Pipeline(steps=[
        ("preprocesamiento", construir_preprocesador()),
        ("clasificador", LogisticRegression(max_iter=MAXIMO_ITERACIONES,
                                            class_weight=class_weight,
                                            random_state=semilla)),
    ])
    return modelo


def construir_modelo_base():
    """Construye el modelo base: siempre predice la clase más frecuente."""
    modelo_base = Pipeline(steps=[
        ("preprocesamiento", construir_preprocesador()),
        ("clasificador", DummyClassifier(strategy="most_frequent")),
    ])
    return modelo_base


def dividir_datos(predictoras, objetivo, proporcion_prueba, semilla):
    """Divide los datos en entrenamiento y prueba, de forma estratificada."""
    (predictoras_entrenamiento,
     predictoras_prueba,
     objetivo_entrenamiento,
     objetivo_prueba) = train_test_split(
        predictoras,
        objetivo,
        test_size=proporcion_prueba,
        stratify=objetivo,
        random_state=semilla,
    )
    return predictoras_entrenamiento, predictoras_prueba, objetivo_entrenamiento, objetivo_prueba


def seleccionar_class_weight(predictoras_entrenamiento, objetivo_entrenamiento,
                             semilla, numero_particiones):
    """
    Elige class_weight con validación cruzada, usando SOLO el entrenamiento.

    Devuelve:
    - el valor elegido para LogisticRegression (None o "balanced"),
    - el nombre de la configuración elegida,
    - los resultados de la validación cruzada de cada configuración.
    """
    validacion_cruzada = StratifiedKFold(n_splits=numero_particiones,
                                         shuffle=True,
                                         random_state=semilla)

    resultados_por_configuracion = {}

    for nombre_configuracion, valor_class_weight in CONFIGURACIONES_CLASS_WEIGHT.items():
        modelo_candidato = construir_modelo_predictivo(valor_class_weight, semilla)

        resultados_validacion = cross_validate(
            modelo_candidato,
            predictoras_entrenamiento,
            objetivo_entrenamiento,
            cv=validacion_cruzada,
            scoring=METRICAS_VALIDACION,
        )

        resumen_metricas = {}
        for nombre_metrica in METRICAS_VALIDACION:
            valores_particiones = resultados_validacion["test_" + nombre_metrica]
            resumen_metricas[nombre_metrica] = {
                "promedio": float(valores_particiones.mean()),
                "desviacion_estandar": float(valores_particiones.std()),
            }

        resultados_por_configuracion[nombre_configuracion] = resumen_metricas

    f1_sin_balanceo = resultados_por_configuracion["sin_balanceo"]["f1"]["promedio"]
    f1_balanceado = resultados_por_configuracion["balanceado"]["f1"]["promedio"]

    if f1_balanceado > f1_sin_balanceo:
        nombre_elegido = "balanceado"
    else:
        nombre_elegido = "sin_balanceo"

    valor_elegido = CONFIGURACIONES_CLASS_WEIGHT[nombre_elegido]

    return valor_elegido, nombre_elegido, resultados_por_configuracion


def evaluar_modelo(modelo_entrenado, predictoras, objetivo_real):
    """Calcula las métricas de clasificación y la matriz de confusión."""
    predicciones = modelo_entrenado.predict(predictoras)
    probabilidades_abandono = modelo_entrenado.predict_proba(predictoras)[:, 1]

    matriz = confusion_matrix(objetivo_real, predicciones, labels=[0, 1])

    resultados = {
        "accuracy": float(accuracy_score(objetivo_real, predicciones)),
        "precision": float(precision_score(objetivo_real, predicciones, zero_division=0)),
        "recall": float(recall_score(objetivo_real, predicciones, zero_division=0)),
        "f1": float(f1_score(objetivo_real, predicciones, zero_division=0)),
        "roc_auc": float(roc_auc_score(objetivo_real, probabilidades_abandono)),
        "matriz_confusion": {
            "verdaderos_negativos": int(matriz[0, 0]),
            "falsos_positivos": int(matriz[0, 1]),
            "falsos_negativos": int(matriz[1, 0]),
            "verdaderos_positivos": int(matriz[1, 1]),
        },
    }

    return resultados