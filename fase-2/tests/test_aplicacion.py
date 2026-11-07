"""
Pruebas automatizadas de la aplicación (Fase 2).

Cubren los cinco comportamientos mínimos de la especificación:
1. Entrenamiento y almacenamiento.
2. Carga y predicción.
3. Valores faltantes.
4. Entrada inválida (función y línea de comandos).
5. Versionamiento.

Además, incluyen pruebas adicionales de validación de tipos, orden de
columnas y categorías no vistas.

Ejecutar desde la carpeta fase-2:
    python -m pytest
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.almacenamiento import cargar_modelo, guardar_version
from src.datos import leer_csv, preparar_datos_entrenamiento, preparar_datos_prediccion
from src.errores import ErrorDatos, ErrorVersion
from src.modelo import construir_modelo_predictivo
from src.prediccion import generar_predicciones
from train import entrenar

RUTA_FASE_2 = Path(__file__).resolve().parent.parent
SEMILLA_PRUEBAS = 42
PARTICIONES_PRUEBAS = 3
PROPORCION_PRUEBA_PRUEBAS = 0.25
NOMBRES_METRICAS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


# ---------------------------------------------------------------
# Funciones y fixtures auxiliares
# ---------------------------------------------------------------

def entrenar_version_pequena(ruta_datos, directorio_modelos, version):
    """Entrena una versión con el proceso completo de train.py, usando datos pequeños."""
    return entrenar(
        ruta_datos=ruta_datos,
        columna_objetivo="Churn",
        directorio_salida=directorio_modelos,
        version=version,
        semilla=SEMILLA_PRUEBAS,
        proporcion_prueba=PROPORCION_PRUEBA_PRUEBAS,
        numero_particiones=PARTICIONES_PRUEBAS,
    )


@pytest.fixture
def modelo_pequeno(ruta_datos_pequenos):
    """Entrena en memoria un modelo con los datos pequeños (sin guardarlo)."""
    datos = leer_csv(ruta_datos_pequenos)
    predictoras, objetivo = preparar_datos_entrenamiento(datos, "Churn")
    modelo = construir_modelo_predictivo("balanced", SEMILLA_PRUEBAS)
    modelo.fit(predictoras, objetivo)
    return modelo


def leer_json(ruta_archivo):
    """Lee un archivo JSON y devuelve su contenido como diccionario."""
    with open(ruta_archivo, "r", encoding="utf-8") as archivo:
        return json.load(archivo)


def leer_contenido_version(carpeta_version):
    """Devuelve el contenido exacto (bytes) de los archivos de una versión."""
    contenido = {}
    for nombre_archivo in ["modelo.joblib", "metrics.json", "metadata.json"]:
        contenido[nombre_archivo] = (carpeta_version / nombre_archivo).read_bytes()
    return contenido


# ---------------------------------------------------------------
# 1. Entrenamiento y almacenamiento
# ---------------------------------------------------------------

def test_entrenamiento_genera_artefacto_metricas_y_metadatos(ruta_datos_pequenos, tmp_path):
    directorio_modelos = tmp_path / "models"

    entrenar_version_pequena(ruta_datos_pequenos, directorio_modelos, "v-prueba")

    carpeta_version = directorio_modelos / "v-prueba"
    assert (carpeta_version / "modelo.joblib").exists()
    assert (carpeta_version / "metrics.json").exists()
    assert (carpeta_version / "metadata.json").exists()

    # El artefacto se puede cargar y es un modelo
    modelo = cargar_modelo(carpeta_version / "modelo.joblib")
    assert hasattr(modelo, "predict")
    assert len(modelo.feature_names_in_) == 19

    # Las métricas incluyen el modelo y el modelo base, con valores válidos
    metricas = leer_json(carpeta_version / "metrics.json")
    assert "modelo" in metricas
    assert "modelo_base" in metricas
    for nombre_metrica in NOMBRES_METRICAS:
        assert 0.0 <= metricas["modelo"][nombre_metrica] <= 1.0
        assert 0.0 <= metricas["modelo_base"][nombre_metrica] <= 1.0

    # Los metadatos identifican la versión y su configuración
    metadatos = leer_json(carpeta_version / "metadata.json")
    assert metadatos["version"] == "v-prueba"
    assert metadatos["variable_objetivo"] == "Churn"
    assert metadatos["semilla"] == SEMILLA_PRUEBAS
    assert metadatos["datos"]["cantidad_observaciones"] == 300
    assert len(metadatos["variables_entrada"]) == 19
    total_separacion = (metadatos["separacion"]["tamano_entrenamiento"]
                        + metadatos["separacion"]["tamano_prueba"])
    assert total_separacion == 300
    assert "commit" in metadatos["codigo"]


# ---------------------------------------------------------------
# 2. Carga y predicción
# ---------------------------------------------------------------

def test_modelo_cargado_conserva_las_predicciones(modelo_pequeno, datos_ejemplo, tmp_path):
    columnas_esperadas = list(modelo_pequeno.feature_names_in_)
    predictoras_ejemplo, _ = preparar_datos_prediccion(datos_ejemplo, columnas_esperadas)

    probabilidades_antes = modelo_pequeno.predict_proba(predictoras_ejemplo)[:, 1]

    guardar_version(tmp_path / "models", "v-carga", modelo_pequeno,
                    {"prueba": True}, {"version": "v-carga"})
    modelo_cargado = cargar_modelo(tmp_path / "models" / "v-carga")

    probabilidades_despues = modelo_cargado.predict_proba(predictoras_ejemplo)[:, 1]
    assert np.allclose(probabilidades_antes, probabilidades_despues)

    resultados, informe = generar_predicciones(modelo_cargado, datos_ejemplo)

    # Una predicción por fila, en el mismo orden de la entrada
    assert len(resultados) == len(datos_ejemplo)
    assert list(resultados["customerID"]) == list(datos_ejemplo["customerID"])
    assert list(resultados["fila_entrada"]) == [1, 2, 3, 4, 5]
    assert set(resultados["prediccion"].unique()).issubset({0, 1})
    assert np.allclose(resultados["probabilidad_abandono"], probabilidades_antes, atol=1e-4)


# ---------------------------------------------------------------
# 3. Valores faltantes
# ---------------------------------------------------------------

def test_valores_faltantes_se_procesan_segun_la_estrategia(modelo_pequeno, datos_ejemplo):
    # TotalCharges vacío de tres formas: texto con espacio, celda nula y la fila
    # EJEMPLO-003, que ya viene vacía en el archivo de ejemplo
    datos_con_faltantes = datos_ejemplo.copy()
    datos_con_faltantes["TotalCharges"] = datos_con_faltantes["TotalCharges"].astype(object)
    datos_con_faltantes.loc[0, "TotalCharges"] = " "
    datos_con_faltantes.loc[1, "TotalCharges"] = np.nan

    # Los mismos clientes, pero con TotalCharges = 0 escrito explícitamente
    datos_con_cero = datos_ejemplo.copy()
    datos_con_cero.loc[[0, 1, 2], "TotalCharges"] = 0.0

    resultados_faltantes, _ = generar_predicciones(modelo_pequeno, datos_con_faltantes)
    resultados_cero, _ = generar_predicciones(modelo_pequeno, datos_con_cero)

    # No se elimina ninguna fila y todas tienen predicción
    assert len(resultados_faltantes) == len(datos_ejemplo)
    assert resultados_faltantes["probabilidad_abandono"].notna().all()

    # Estrategia documentada: TotalCharges vacío se imputa con 0
    assert np.allclose(resultados_faltantes["probabilidad_abandono"],
                       resultados_cero["probabilidad_abandono"])

    # Una variable categórica vacía también se procesa (categoría más frecuente)
    datos_categoria_vacia = datos_ejemplo.copy()
    datos_categoria_vacia.loc[3, "Contract"] = np.nan
    resultados_categoria, _ = generar_predicciones(modelo_pequeno, datos_categoria_vacia)
    assert len(resultados_categoria) == len(datos_ejemplo)
    assert resultados_categoria["probabilidad_abandono"].notna().all()


# ---------------------------------------------------------------
# 4. Entrada inválida
# ---------------------------------------------------------------

def test_columna_faltante_produce_error_claro(modelo_pequeno, datos_ejemplo):
    datos_sin_columna = datos_ejemplo.drop(columns=["tenure"])

    with pytest.raises(ErrorDatos, match="tenure"):
        generar_predicciones(modelo_pequeno, datos_sin_columna)


def test_predict_cli_con_columna_faltante_termina_con_error(modelo_pequeno, datos_ejemplo, tmp_path):
    guardar_version(tmp_path / "models", "v-cli", modelo_pequeno,
                    {"prueba": True}, {"version": "v-cli"})

    ruta_entrada = tmp_path / "sin_tenure.csv"
    datos_ejemplo.drop(columns=["tenure"]).to_csv(ruta_entrada, index=False)
    ruta_salida = tmp_path / "salida" / "predicciones.csv"

    entorno = dict(os.environ, PYTHONIOENCODING="utf-8")
    proceso = subprocess.run(
        [sys.executable, "predict.py",
         "--model", str(tmp_path / "models" / "v-cli"),
         "--input", str(ruta_entrada),
         "--output", str(ruta_salida)],
        cwd=RUTA_FASE_2,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=entorno,
    )

    assert proceso.returncode != 0
    assert "tenure" in proceso.stderr
    assert not ruta_salida.exists()


def test_tipo_incompatible_produce_error_claro(modelo_pequeno, datos_ejemplo):
    datos_tipo_invalido = datos_ejemplo.copy()
    datos_tipo_invalido["MonthlyCharges"] = datos_tipo_invalido["MonthlyCharges"].astype(object)
    datos_tipo_invalido.loc[1, "MonthlyCharges"] = "abc"

    with pytest.raises(ErrorDatos, match="MonthlyCharges"):
        generar_predicciones(modelo_pequeno, datos_tipo_invalido)


# ---------------------------------------------------------------
# 5. Versionamiento
# ---------------------------------------------------------------

def test_nueva_version_conserva_la_anterior_y_no_sobrescribe(ruta_datos_pequenos, tmp_path):
    directorio_modelos = tmp_path / "models"

    entrenar_version_pequena(ruta_datos_pequenos, directorio_modelos, "v1")
    contenido_v1_antes = leer_contenido_version(directorio_modelos / "v1")

    entrenar_version_pequena(ruta_datos_pequenos, directorio_modelos, "v2")

    # Ambas versiones existen y se pueden cargar
    assert (directorio_modelos / "v1" / "modelo.joblib").exists()
    assert (directorio_modelos / "v2" / "modelo.joblib").exists()
    modelo_v1 = cargar_modelo(directorio_modelos / "v1")
    modelo_v2 = cargar_modelo(directorio_modelos / "v2")
    assert hasattr(modelo_v1, "predict")
    assert hasattr(modelo_v2, "predict")

    # Repetir un identificador produce un error y no sobrescribe
    with pytest.raises(ErrorVersion, match="v1"):
        entrenar_version_pequena(ruta_datos_pequenos, directorio_modelos, "v1")

    contenido_v1_despues = leer_contenido_version(directorio_modelos / "v1")
    assert contenido_v1_antes == contenido_v1_despues


# ---------------------------------------------------------------
# Pruebas adicionales
# ---------------------------------------------------------------

def test_orden_de_columnas_no_afecta_la_prediccion(modelo_pequeno, datos_ejemplo):
    columnas_invertidas = list(reversed(datos_ejemplo.columns))
    datos_reordenados = datos_ejemplo[columnas_invertidas]

    resultados_original, _ = generar_predicciones(modelo_pequeno, datos_ejemplo)
    resultados_reordenados, _ = generar_predicciones(modelo_pequeno, datos_reordenados)

    assert np.allclose(resultados_original["probabilidad_abandono"],
                       resultados_reordenados["probabilidad_abandono"])


def test_categoria_no_vista_se_informa_y_se_predice(modelo_pequeno, datos_ejemplo):
    datos_categoria_nueva = datos_ejemplo.copy()
    datos_categoria_nueva.loc[0, "Contract"] = "Three year"

    resultados, informe = generar_predicciones(modelo_pequeno, datos_categoria_nueva)

    assert len(resultados) == len(datos_ejemplo)
    assert "Contract" in informe["categorias_no_vistas"]
    assert "Three year" in informe["categorias_no_vistas"]["Contract"]