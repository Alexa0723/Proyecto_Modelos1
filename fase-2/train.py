"""
Script de entrenamiento del modelo de abandono de clientes (Churn).

Lee un CSV etiquetado, entrena el modelo con el mismo procedimiento de la
Fase 1 y guarda una nueva versión con el modelo, sus métricas y sus metadatos.

Ejemplo de uso (desde la carpeta fase-2, en una sola línea):
    python train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --target Churn --output-dir models --version v1 --seed 42
"""

import argparse
import hashlib
import platform
import sys
from pathlib import Path

import sklearn

from src.almacenamiento import (
    guardar_version,
    obtener_fecha_actual,
    obtener_referencia_codigo,
    verificar_version_disponible,
)
from src.config import (
    COLUMNA_CARGOS_TOTALES,
    COLUMNA_OBJETIVO_POR_DEFECTO,
    COLUMNAS_NUMERICAS,
    NUMERO_PARTICIONES_VALIDACION_POR_DEFECTO,
    PROPORCION_PRUEBA_POR_DEFECTO,
    SEMILLA_POR_DEFECTO,
    VALOR_CLASE_NEGATIVA,
    VALOR_CLASE_POSITIVA,
)
from src.datos import leer_csv, preparar_datos_entrenamiento
from src.errores import ErrorAplicacion
from src.modelo import (
    construir_modelo_base,
    construir_modelo_predictivo,
    dividir_datos,
    evaluar_modelo,
    seleccionar_class_weight,
)

FUENTE_DATOS = "Telco Customer Churn (Kaggle): https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
METRICAS_A_MOSTRAR = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def crear_parser_argumentos():
    """Define los argumentos que acepta el script."""
    parser = argparse.ArgumentParser(
        description="Entrena el modelo de abandono de clientes y guarda una nueva versión.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True,
                        help="Ruta del archivo CSV con los datos etiquetados.")
    parser.add_argument("--target", default=COLUMNA_OBJETIVO_POR_DEFECTO,
                        help="Nombre de la columna objetivo (con valores 'Yes' y 'No').")
    parser.add_argument("--output-dir", default="models",
                        help="Directorio donde se crean las carpetas de las versiones.")
    parser.add_argument("--version", required=True,
                        help="Identificador único de la versión (por ejemplo: v1). "
                             "Una versión existente no se sobrescribe.")
    parser.add_argument("--seed", type=int, default=SEMILLA_POR_DEFECTO,
                        help="Semilla aleatoria para la división de datos, "
                             "la validación cruzada y el modelo.")
    parser.add_argument("--test-size", type=float, default=PROPORCION_PRUEBA_POR_DEFECTO,
                        help="Proporción de los datos reservada para prueba (mayor que 0 y menor que 1).")
    parser.add_argument("--cv-folds", type=int, default=NUMERO_PARTICIONES_VALIDACION_POR_DEFECTO,
                        help="Número de particiones de la validación cruzada (mínimo 2).")
    return parser


def validar_parametros(proporcion_prueba, numero_particiones):
    """Verifica que los parámetros numéricos tengan valores válidos."""
    if proporcion_prueba <= 0 or proporcion_prueba >= 1:
        raise ErrorAplicacion(
            "--test-size debe ser un número mayor que 0 y menor que 1 (por ejemplo, 0.2)."
        )

    if numero_particiones < 2:
        raise ErrorAplicacion("--cv-folds debe ser al menos 2.")


def calcular_huella_archivo(ruta_archivo):
    """Calcula la huella SHA-256 del archivo, para identificar exactamente los datos usados."""
    calculadora = hashlib.sha256()

    with open(ruta_archivo, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(65536), b""):
            calculadora.update(bloque)

    return calculadora.hexdigest()


def describir_variables_entrada(columnas):
    """Devuelve el nombre y el tipo esperado de cada variable de entrada."""
    descripcion = []

    for nombre_columna in columnas:
        if nombre_columna in COLUMNAS_NUMERICAS or nombre_columna in COLUMNA_CARGOS_TOTALES:
            tipo_esperado = "numérico"
        elif nombre_columna == "SeniorCitizen":
            tipo_esperado = "numérico 0/1 (tratado como categórico)"
        else:
            tipo_esperado = "texto (categórico)"

        descripcion.append({"nombre": nombre_columna, "tipo_esperado": tipo_esperado})

    return descripcion


def entrenar(ruta_datos, columna_objetivo, directorio_salida, version,
             semilla=SEMILLA_POR_DEFECTO,
             proporcion_prueba=PROPORCION_PRUEBA_POR_DEFECTO,
             numero_particiones=NUMERO_PARTICIONES_VALIDACION_POR_DEFECTO):
    """
    Ejecuta el proceso completo de entrenamiento y guarda una nueva versión.

    Devuelve un diccionario con la versión, la configuración elegida,
    las métricas y las rutas de los archivos generados.
    """
    validar_parametros(proporcion_prueba, numero_particiones)

    # Se verifica antes de entrenar, para no perder tiempo si la versión ya existe
    verificar_version_disponible(directorio_salida, version)
    referencia_codigo = obtener_referencia_codigo()

    print("1/5 Leyendo y validando los datos...")
    datos = leer_csv(ruta_datos)
    predictoras, objetivo = preparar_datos_entrenamiento(datos, columna_objetivo)

    print("2/5 Dividiendo en entrenamiento y prueba...")
    (predictoras_entrenamiento,
     predictoras_prueba,
     objetivo_entrenamiento,
     objetivo_prueba) = dividir_datos(predictoras, objetivo, proporcion_prueba, semilla)

    print("3/5 Seleccionando class_weight con validación cruzada (solo entrenamiento)...")
    valor_class_weight, nombre_configuracion, resultados_validacion = seleccionar_class_weight(
        predictoras_entrenamiento, objetivo_entrenamiento, semilla, numero_particiones
    )

    print("4/5 Entrenando y evaluando el modelo predictivo y el modelo base...")
    modelo = construir_modelo_predictivo(valor_class_weight, semilla)
    modelo.fit(predictoras_entrenamiento, objetivo_entrenamiento)

    modelo_base = construir_modelo_base()
    modelo_base.fit(predictoras_entrenamiento, objetivo_entrenamiento)

    metricas_modelo = evaluar_modelo(modelo, predictoras_prueba, objetivo_prueba)
    metricas_modelo_base = evaluar_modelo(modelo_base, predictoras_prueba, objetivo_prueba)

    metricas = {
        "version": version,
        "conjunto_evaluado": "prueba (datos no usados para entrenar)",
        "metrica_principal": "f1 (clase 'abandona')",
        "configuracion_elegida": nombre_configuracion,
        "modelo": metricas_modelo,
        "modelo_base": metricas_modelo_base,
        "validacion_cruzada_entrenamiento": resultados_validacion,
    }

    parametros_clasificador = modelo.named_steps["clasificador"].get_params()

    metadatos = {
        "version": version,
        "fecha_creacion": obtener_fecha_actual(),
        "tipo_problema": "clasificación binaria",
        "variable_objetivo": columna_objetivo,
        "significado_etiquetas": {
            "0": f"{VALOR_CLASE_NEGATIVA}: el cliente permanece",
            "1": f"{VALOR_CLASE_POSITIVA}: el cliente abandona",
        },
        "algoritmo": "LogisticRegression (scikit-learn), dentro de un Pipeline con el preprocesamiento",
        "modelo_base": "DummyClassifier (estrategia most_frequent)",
        "parametros_modelo": {
            "class_weight": parametros_clasificador["class_weight"],
            "C": parametros_clasificador["C"],
            "penalty": parametros_clasificador["penalty"],
            "solver": parametros_clasificador["solver"],
            "max_iter": parametros_clasificador["max_iter"],
        },
        "semilla": semilla,
        "variables_entrada": describir_variables_entrada(list(predictoras.columns)),
        "tratamiento_valores_faltantes": {
            "tenure y MonthlyCharges": "mediana aprendida en entrenamiento",
            "TotalCharges": "constante 0 (clientes nuevos sin cobros)",
            "variables categóricas": "categoría más frecuente aprendida en entrenamiento",
        },
        "reglas_prediccion": {
            "columnas_adicionales": "se ignoran (por ejemplo, customerID)",
            "categorias_no_vistas": "se codifican con todas sus columnas en 0 (handle_unknown='ignore')",
        },
        "datos": {
            "fuente": FUENTE_DATOS,
            "archivo": Path(ruta_datos).name,
            "huella_sha256": calcular_huella_archivo(ruta_datos),
            "cantidad_observaciones": len(datos),
        },
        "separacion": {
            "estrategia": "train_test_split estratificado según la variable objetivo",
            "proporcion_prueba": proporcion_prueba,
            "tamano_entrenamiento": len(predictoras_entrenamiento),
            "tamano_prueba": len(predictoras_prueba),
            "validacion_cruzada": f"StratifiedKFold de {numero_particiones} particiones, solo con entrenamiento",
        },
        "metrica_evaluacion": "f1 de la clase 'abandona' (principal); recall, precision, roc_auc y accuracy (complementarias)",
        "entorno": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "archivo_dependencias": "requirements.txt",
        },
        "codigo": referencia_codigo,
    }

    print("5/5 Guardando la versión...")
    rutas_archivos = guardar_version(directorio_salida, version, modelo, metricas, metadatos)

    return {
        "version": version,
        "configuracion_elegida": nombre_configuracion,
        "metricas": metricas,
        "rutas_archivos": rutas_archivos,
    }


def mostrar_resumen(resultado):
    """Muestra en pantalla un resumen del entrenamiento y las rutas de los archivos."""
    metricas_modelo = resultado["metricas"]["modelo"]
    metricas_modelo_base = resultado["metricas"]["modelo_base"]

    print()
    print("=" * 50)
    print(f"Versión '{resultado['version']}' entrenada y guardada correctamente.")
    print(f"Configuración elegida: {resultado['configuracion_elegida']}")
    print()
    print("Métricas en el conjunto de prueba:")
    print(f"{'Métrica':<12}{'Modelo base':>14}{'Modelo':>14}")

    for nombre_metrica in METRICAS_A_MOSTRAR:
        valor_base = metricas_modelo_base[nombre_metrica]
        valor_modelo = metricas_modelo[nombre_metrica]
        print(f"{nombre_metrica:<12}{valor_base:>14.4f}{valor_modelo:>14.4f}")

    print()
    print("Archivos generados:")
    for tipo_archivo, ruta_archivo in resultado["rutas_archivos"].items():
        print(f"  - {tipo_archivo}: {ruta_archivo}")
    print("=" * 50)


def main():
    """Punto de entrada: lee los argumentos, entrena y maneja los errores previsibles."""
    parser = crear_parser_argumentos()
    argumentos = parser.parse_args()

    try:
        resultado = entrenar(
            ruta_datos=argumentos.data,
            columna_objetivo=argumentos.target,
            directorio_salida=argumentos.output_dir,
            version=argumentos.version,
            semilla=argumentos.seed,
            proporcion_prueba=argumentos.test_size,
            numero_particiones=argumentos.cv_folds,
        )
    except ErrorAplicacion as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)

    mostrar_resumen(resultado)


if __name__ == "__main__":
    main()