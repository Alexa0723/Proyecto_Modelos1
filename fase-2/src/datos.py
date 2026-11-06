"""
Lectura, validación y limpieza de los datos.

Este módulo lo usan tanto el entrenamiento como la predicción, para que ambos
procesos apliquen exactamente las mismas reglas a los datos.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    COLUMNAS_PREDICTORAS,
    COLUMNAS_QUE_DEBEN_SER_NUMERICAS,
    VALOR_CLASE_NEGATIVA,
    VALOR_CLASE_POSITIVA,
)
from src.errores import ErrorDatos


def leer_csv(ruta_archivo):
    """Lee un archivo CSV y verifica que exista y que tenga filas."""
    ruta = Path(ruta_archivo)

    if not ruta.exists():
        raise ErrorDatos(f"No se encontró el archivo de datos: {ruta}")

    if not ruta.is_file():
        raise ErrorDatos(f"La ruta indicada no corresponde a un archivo: {ruta}")

    try:
        datos = pd.read_csv(ruta)
    except pd.errors.EmptyDataError:
        raise ErrorDatos(f"El archivo está vacío: {ruta}")
    except Exception as error:
        raise ErrorDatos(f"No se pudo leer el archivo CSV {ruta}: {error}")

    if len(datos) == 0:
        raise ErrorDatos(f"El archivo no contiene filas de datos: {ruta}")

    return datos


def validar_columnas_requeridas(datos, columnas_requeridas):
    """Lanza un error si falta alguna de las columnas requeridas."""
    columnas_faltantes = []

    for nombre_columna in columnas_requeridas:
        if nombre_columna not in datos.columns:
            columnas_faltantes.append(nombre_columna)

    if len(columnas_faltantes) > 0:
        raise ErrorDatos(
            "Faltan columnas requeridas en el archivo: " + ", ".join(columnas_faltantes)
        )


def convertir_texto_vacio_en_nulo(valor):
    """Convierte un texto vacío o compuesto solo por espacios en un valor nulo (NaN)."""
    if isinstance(valor, str) and valor.strip() == "":
        return np.nan
    else:
        return valor


def convertir_columnas_numericas(datos, columnas_numericas):
    """
    Convierte a número las columnas indicadas.

    - Los textos vacíos (como los 11 de TotalCharges) se convierten en NaN,
      que luego trata el imputador del Pipeline.
    - Si una celda contiene un texto que no es un número (por ejemplo, "abc"),
      se lanza un error en lugar de convertirlo en NaN sin avisar.
    """
    datos_convertidos = datos.copy()

    for nombre_columna in columnas_numericas:
        valores_limpios = datos_convertidos[nombre_columna].map(convertir_texto_vacio_en_nulo)
        valores_numericos = pd.to_numeric(valores_limpios, errors="coerce")

        # Un valor es inválido si NO era nulo, pero no se pudo convertir a número
        es_valor_invalido = valores_limpios.notna() & valores_numericos.isna()

        if es_valor_invalido.any():
            numeros_fila_csv = []
            for indice in datos_convertidos.index[es_valor_invalido][:5]:
                # +2: una fila por el encabezado y otra porque pandas cuenta desde 0
                numeros_fila_csv.append(str(indice + 2))

            raise ErrorDatos(
                f"La columna '{nombre_columna}' debe ser numérica, pero contiene valores "
                f"no numéricos en las filas del CSV: {', '.join(numeros_fila_csv)}."
            )

        datos_convertidos[nombre_columna] = valores_numericos

    return datos_convertidos


def codificar_objetivo(valores_objetivo, nombre_columna):
    """Convierte la variable objetivo a números: 'Yes' -> 1 (abandona), 'No' -> 0 (permanece)."""
    if valores_objetivo.isna().any():
        cantidad_nulos = int(valores_objetivo.isna().sum())
        raise ErrorDatos(
            f"La variable objetivo '{nombre_columna}' tiene {cantidad_nulos} valores vacíos. "
            "Todas las filas de entrenamiento deben tener la variable objetivo."
        )

    valores_permitidos = {VALOR_CLASE_NEGATIVA, VALOR_CLASE_POSITIVA}
    valores_encontrados = set(valores_objetivo.unique())
    valores_inesperados = valores_encontrados - valores_permitidos

    if len(valores_inesperados) > 0:
        textos_inesperados = []
        for valor in valores_inesperados:
            textos_inesperados.append(str(valor))
        raise ErrorDatos(
            f"La variable objetivo '{nombre_columna}' solo puede contener "
            f"'{VALOR_CLASE_NEGATIVA}' o '{VALOR_CLASE_POSITIVA}'. "
            f"Se encontraron valores inesperados: {', '.join(textos_inesperados)}."
        )

    if len(valores_encontrados) < 2:
        raise ErrorDatos(
            f"La variable objetivo '{nombre_columna}' debe contener ambas clases "
            f"('{VALOR_CLASE_NEGATIVA}' y '{VALOR_CLASE_POSITIVA}') para entrenar."
        )

    codificacion = {VALOR_CLASE_NEGATIVA: 0, VALOR_CLASE_POSITIVA: 1}
    return valores_objetivo.map(codificacion).astype(int)


def preparar_datos_entrenamiento(datos, columna_objetivo):
    """
    Valida y prepara los datos etiquetados para entrenar.

    Devuelve las variables predictoras (en el orden definido en config.py)
    y la variable objetivo codificada como 0/1.
    """
    if columna_objetivo in COLUMNAS_PREDICTORAS:
        raise ErrorDatos(
            f"'{columna_objetivo}' es una variable predictora y no puede usarse como objetivo."
        )

    validar_columnas_requeridas(datos, COLUMNAS_PREDICTORAS + [columna_objetivo])

    datos_limpios = convertir_columnas_numericas(datos, COLUMNAS_QUE_DEBEN_SER_NUMERICAS)

    objetivo = codificar_objetivo(datos_limpios[columna_objetivo], columna_objetivo)
    predictoras = datos_limpios[COLUMNAS_PREDICTORAS].copy()

    return predictoras, objetivo


def preparar_datos_prediccion(datos, columnas_esperadas):
    """
    Valida y prepara los datos nuevos para predecir.

    - Verifica que estén todas las columnas que usó el modelo al entrenar.
    - Ordena las columnas según el esquema del entrenamiento, sin depender
      del orden del CSV.
    - Las columnas adicionales (por ejemplo, customerID) se ignoran y se
      informan.

    Devuelve las predictoras listas para el modelo y la lista de columnas
    adicionales ignoradas.
    """
    validar_columnas_requeridas(datos, columnas_esperadas)

    columnas_adicionales = []
    for nombre_columna in datos.columns:
        if nombre_columna not in columnas_esperadas:
            columnas_adicionales.append(nombre_columna)

    columnas_numericas_a_convertir = []
    for nombre_columna in COLUMNAS_QUE_DEBEN_SER_NUMERICAS:
        if nombre_columna in columnas_esperadas:
            columnas_numericas_a_convertir.append(nombre_columna)

    datos_limpios = convertir_columnas_numericas(datos, columnas_numericas_a_convertir)
    predictoras = datos_limpios[columnas_esperadas].copy()

    return predictoras, columnas_adicionales