"""
Generación de predicciones con un modelo ya entrenado.

Lo usa predict.py y podrá reutilizarse en la API REST de la siguiente fase.
Este módulo nunca entrena ni reajusta el modelo: solo aplica lo aprendido.
"""

import os
import warnings
from pathlib import Path

import pandas as pd

from src.config import COLUMNA_IDENTIFICADOR
from src.datos import preparar_datos_prediccion
from src.errores import ErrorDatos

ETIQUETAS_PREDICCION = {
    0: "permanece",
    1: "abandona",
}


def obtener_columnas_esperadas(modelo):
    """Devuelve las columnas, en orden, con las que se entrenó el modelo."""
    return list(modelo.feature_names_in_)


def detectar_categorias_no_vistas(modelo, predictoras):
    """
    Identifica valores de variables categóricas que no existían en el entrenamiento.

    Devuelve un diccionario {columna: [valores nuevos]}. Si la estructura del
    modelo no permite verificarlo, devuelve None.
    """
    try:
        preprocesador = modelo.named_steps["preprocesamiento"]
        transformador_categorico = preprocesador.named_transformers_["categoricas"]
        codificador = transformador_categorico.named_steps["codificacion"]
    except (AttributeError, KeyError):
        return None

    columnas_categoricas = []
    for nombre_transformador, transformador, columnas in preprocesador.transformers_:
        if nombre_transformador == "categoricas":
            columnas_categoricas = list(columnas)

    categorias_no_vistas = {}

    for posicion, nombre_columna in enumerate(columnas_categoricas):
        categorias_conocidas = set(codificador.categories_[posicion])
        valores_nuevos = []

        for valor in predictoras[nombre_columna].dropna().unique():
            if valor not in categorias_conocidas:
                valores_nuevos.append(str(valor))

        if len(valores_nuevos) > 0:
            categorias_no_vistas[nombre_columna] = valores_nuevos

    return categorias_no_vistas


def generar_predicciones(modelo, datos_entrada):
    """
    Genera una predicción por cada fila de datos_entrada, conservando el orden.

    Devuelve:
    - un DataFrame con los resultados,
    - un informe con las columnas adicionales ignoradas y las categorías no vistas.
    """
    columnas_esperadas = obtener_columnas_esperadas(modelo)
    predictoras, columnas_adicionales = preparar_datos_prediccion(datos_entrada, columnas_esperadas)
    categorias_no_vistas = detectar_categorias_no_vistas(modelo, predictoras)

    # Las categorías no vistas ya se informan en el resumen; se evita repetir
    # la advertencia técnica de scikit-learn
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories", category=UserWarning)
        predicciones = modelo.predict(predictoras)
        probabilidades_abandono = modelo.predict_proba(predictoras)[:, 1]

    resultados = pd.DataFrame()

    if COLUMNA_IDENTIFICADOR in datos_entrada.columns:
        resultados[COLUMNA_IDENTIFICADOR] = datos_entrada[COLUMNA_IDENTIFICADOR].values

    resultados["fila_entrada"] = range(1, len(datos_entrada) + 1)
    resultados["prediccion"] = predicciones

    etiquetas = []
    for valor_prediccion in predicciones:
        etiquetas.append(ETIQUETAS_PREDICCION[int(valor_prediccion)])
    resultados["etiqueta"] = etiquetas

    resultados["probabilidad_abandono"] = probabilidades_abandono.round(4)

    # Verificación de seguridad: debe haber exactamente una predicción por fila
    if len(resultados) != len(datos_entrada):
        raise ErrorDatos("La cantidad de predicciones no coincide con la cantidad de filas de entrada.")

    informe = {
        "columnas_adicionales": columnas_adicionales,
        "categorias_no_vistas": categorias_no_vistas,
    }

    return resultados, informe


def validar_ruta_salida(ruta_salida):
    """Verifica que la ruta de salida sea un archivo .csv."""
    ruta = Path(ruta_salida)

    if ruta.suffix.lower() != ".csv":
        raise ErrorDatos(f"El archivo de salida debe tener extensión .csv: {ruta}")

    if ruta.exists() and ruta.is_dir():
        raise ErrorDatos(f"La ruta de salida es una carpeta, no un archivo: {ruta}")

    return ruta


def guardar_predicciones(resultados, ruta_salida):
    """
    Guarda las predicciones en un CSV de forma segura.

    Primero escribe un archivo temporal y, solo si se completó, lo renombra al
    nombre final. Así nunca queda un archivo incompleto que parezca válido.
    """
    ruta = validar_ruta_salida(ruta_salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta_temporal = ruta.with_name(ruta.name + ".tmp")

    try:
        resultados.to_csv(ruta_temporal, index=False)
        os.replace(ruta_temporal, ruta)
    except OSError as error:
        if ruta_temporal.exists():
            ruta_temporal.unlink()
        raise ErrorDatos(f"No se pudo escribir el archivo de salida {ruta}: {error}")

    return ruta