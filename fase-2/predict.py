"""
Script de predicción del modelo de abandono de clientes (Churn).

Carga una versión guardada del modelo y genera una predicción por cada fila
de un CSV de entrada. No vuelve a entrenar el modelo ni a ajustar las
transformaciones.

Ejemplo de uso (desde la carpeta fase-2, en una sola línea):
    python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
"""

import argparse
import sys

from src.almacenamiento import cargar_metadatos, cargar_modelo
from src.datos import leer_csv
from src.errores import ErrorAplicacion
from src.prediccion import (
    ETIQUETAS_PREDICCION,
    generar_predicciones,
    guardar_predicciones,
    validar_ruta_salida,
)


def crear_parser_argumentos():
    """Define los argumentos que acepta el script."""
    parser = argparse.ArgumentParser(
        description="Genera predicciones de abandono de clientes con una versión guardada del modelo.",
    )
    parser.add_argument("--model", required=True,
                        help="Ruta del archivo modelo.joblib o de la carpeta de la versión "
                             "(por ejemplo: models/v1/modelo.joblib o models/v1).")
    parser.add_argument("--input", required=True,
                        help="Ruta del CSV con las variables de entrada. "
                             "La variable objetivo (Churn) no es necesaria.")
    parser.add_argument("--output", required=True,
                        help="Ruta del CSV de salida con las predicciones (por ejemplo: outputs/predicciones.csv).")
    return parser


def predecir(ruta_modelo, ruta_entrada, ruta_salida):
    """
    Carga el modelo, predice sobre el CSV de entrada y guarda los resultados.

    Devuelve un diccionario con la versión del modelo, los resultados,
    el informe de validación y la ruta del archivo generado.
    """
    # Se valida la salida al inicio, para no hacer todo el proceso y fallar al final
    validar_ruta_salida(ruta_salida)

    modelo = cargar_modelo(ruta_modelo)
    metadatos = cargar_metadatos(ruta_modelo)

    if metadatos is None:
        version_modelo = "desconocida (sin metadata.json)"
    else:
        version_modelo = metadatos.get("version", "desconocida")

    datos_entrada = leer_csv(ruta_entrada)
    resultados, informe = generar_predicciones(modelo, datos_entrada)
    resultados["version_modelo"] = version_modelo

    ruta_generada = guardar_predicciones(resultados, ruta_salida)

    return {
        "version_modelo": version_modelo,
        "resultados": resultados,
        "informe": informe,
        "ruta_salida": ruta_generada,
    }


def mostrar_resumen(resultado):
    """Muestra un resumen de las predicciones generadas."""
    resultados = resultado["resultados"]
    informe = resultado["informe"]

    cantidad_abandona = int((resultados["prediccion"] == 1).sum())
    cantidad_permanece = int((resultados["prediccion"] == 0).sum())

    print("=" * 50)
    print(f"Predicciones generadas con la versión: {resultado['version_modelo']}")
    print(f"Filas procesadas: {len(resultados)}")
    print(f"  - Predicción 1 ({ETIQUETAS_PREDICCION[1]}): {cantidad_abandona}")
    print(f"  - Predicción 0 ({ETIQUETAS_PREDICCION[0]}): {cantidad_permanece}")

    if len(informe["columnas_adicionales"]) > 0:
        print(f"Columnas adicionales ignoradas: {', '.join(informe['columnas_adicionales'])}")

    if informe["categorias_no_vistas"] is None:
        print("Aviso: no fue posible verificar las categorías no vistas para este modelo.")
    elif len(informe["categorias_no_vistas"]) > 0:
        print("Aviso: categorías no vistas en el entrenamiento "
              "(se codificaron con todas sus columnas en 0):")
        for nombre_columna, valores in informe["categorias_no_vistas"].items():
            print(f"  - {nombre_columna}: {', '.join(valores)}")

    print(f"Archivo generado: {resultado['ruta_salida']}")
    print("=" * 50)


def main():
    """Punto de entrada: lee los argumentos, predice y maneja los errores previsibles."""
    parser = crear_parser_argumentos()
    argumentos = parser.parse_args()

    try:
        resultado = predecir(
            ruta_modelo=argumentos.model,
            ruta_entrada=argumentos.input,
            ruta_salida=argumentos.output,
        )
    except ErrorAplicacion as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)

    mostrar_resumen(resultado)


if __name__ == "__main__":
    main()