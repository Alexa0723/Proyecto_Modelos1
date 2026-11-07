"""
Almacenamiento y carga de versiones del modelo.

"""

import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import joblib

from src.errores import ErrorModelo, ErrorVersion

NOMBRE_ARCHIVO_MODELO = "modelo.joblib"
NOMBRE_ARCHIVO_METRICAS = "metrics.json"
NOMBRE_ARCHIVO_METADATOS = "metadata.json"

# Solo letras, números, punto, guion y guion bajo (por ejemplo: v1, v2, v-docker)
PATRON_IDENTIFICADOR_VERSION = r"[A-Za-z0-9._-]+"


def obtener_ruta_version(directorio_salida, version):
    """Valida el identificador de versión y devuelve la ruta de su carpeta."""
    if version is None or version.strip() == "":
        raise ErrorVersion("El identificador de versión no puede estar vacío.")

    if re.fullmatch(PATRON_IDENTIFICADOR_VERSION, version) is None or version in (".", ".."):
        raise ErrorVersion(
            f"Identificador de versión inválido: '{version}'. "
            "Use solo letras, números, puntos, guiones o guiones bajos (por ejemplo: v1)."
        )

    return Path(directorio_salida) / version


def verificar_version_disponible(directorio_salida, version):
    """Lanza un error si la versión ya existe, para no sobrescribirla."""
    ruta_version = obtener_ruta_version(directorio_salida, version)

    if ruta_version.exists():
        raise ErrorVersion(
            f"La versión '{version}' ya existe en {ruta_version}. "
            "No se sobrescribe una versión existente: use otro identificador con --version."
        )

    return ruta_version


def guardar_json(contenido, ruta_archivo):
    """Guarda un diccionario como archivo JSON legible."""
    with open(ruta_archivo, "w", encoding="utf-8") as archivo:
        json.dump(contenido, archivo, indent=2, ensure_ascii=False)


def guardar_version(directorio_salida, version, modelo, metricas, metadatos):
    """
    Crea la carpeta de la versión y guarda el modelo, las métricas y los metadatos.

    Si algo falla mientras se guarda, se elimina la carpeta incompleta para no
    dejar una versión que aparente ser válida.
    """
    ruta_version = obtener_ruta_version(directorio_salida, version)

    try:
        ruta_version.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise ErrorVersion(
            f"La versión '{version}' ya existe en {ruta_version}. "
            "No se sobrescribe una versión existente: use otro identificador con --version."
        )

    rutas_archivos = {
        "modelo": ruta_version / NOMBRE_ARCHIVO_MODELO,
        "metricas": ruta_version / NOMBRE_ARCHIVO_METRICAS,
        "metadatos": ruta_version / NOMBRE_ARCHIVO_METADATOS,
    }

    try:
        joblib.dump(modelo, rutas_archivos["modelo"])
        guardar_json(metricas, rutas_archivos["metricas"])
        guardar_json(metadatos, rutas_archivos["metadatos"])
    except Exception as error:
        shutil.rmtree(ruta_version, ignore_errors=True)
        raise ErrorVersion(f"No se pudieron guardar los archivos de la versión '{version}': {error}")

    return rutas_archivos


def cargar_modelo(ruta_modelo):
    """
    Carga un modelo guardado.

    Acepta la ruta del archivo modelo.joblib o la carpeta de la versión
    (por ejemplo, models/v1).
    """
    ruta = Path(ruta_modelo)

    if ruta.is_dir():
        ruta = ruta / NOMBRE_ARCHIVO_MODELO

    if not ruta.exists():
        raise ErrorModelo(f"No se encontró el modelo: {ruta}")

    try:
        modelo = joblib.load(ruta)
    except Exception as error:
        raise ErrorModelo(f"No se pudo cargar el modelo {ruta}: {error}")

    if not hasattr(modelo, "predict") or not hasattr(modelo, "feature_names_in_"):
        raise ErrorModelo(
            f"El archivo {ruta} no contiene un modelo entrenado válido para esta aplicación."
        )

    return modelo


def cargar_metadatos(ruta_modelo):
    """Carga metadata.json de la versión, si existe. Si no existe, devuelve None."""
    ruta = Path(ruta_modelo)

    if ruta.is_dir():
        carpeta_version = ruta
    else:
        carpeta_version = ruta.parent

    ruta_metadatos = carpeta_version / NOMBRE_ARCHIVO_METADATOS

    if not ruta_metadatos.exists():
        return None

    with open(ruta_metadatos, "r", encoding="utf-8") as archivo:
        return json.load(archivo)


def ejecutar_comando_git(argumentos):
    """Ejecuta un comando de git y devuelve su salida, o None si no está disponible."""
    try:
        resultado = subprocess.run(["git"] + argumentos,
                                   capture_output=True,
                                   text=True,
                                   check=True)
        return resultado.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def obtener_referencia_codigo():
    """
    Identifica la versión del código usada para entrenar.

    1. Si existe la variable de entorno GIT_COMMIT (útil en Docker, donde no
       se copia la carpeta .git), se usa su valor.
    2. Si no, se consulta el hash del commit actual con git.
    3. Si git no está disponible, se registra como "no disponible".
    """
    commit_desde_entorno = os.environ.get("GIT_COMMIT")

    if commit_desde_entorno:
        return {
            "commit": commit_desde_entorno,
            "origen": "variable de entorno GIT_COMMIT",
            "cambios_sin_commit": None,
        }

    commit = ejecutar_comando_git(["rev-parse", "HEAD"])

    if commit is None:
        return {
            "commit": "no disponible",
            "origen": "git no disponible en este entorno",
            "cambios_sin_commit": None,
        }

    estado = ejecutar_comando_git(["status", "--porcelain"])

    if estado is None:
        hay_cambios_sin_commit = None
    elif estado == "":
        hay_cambios_sin_commit = False
    else:
        hay_cambios_sin_commit = True

    return {
        "commit": commit,
        "origen": "git",
        "cambios_sin_commit": hay_cambios_sin_commit,
    }


def obtener_fecha_actual():
    """Devuelve la fecha y hora actual en formato ISO (por ejemplo: 2026-11-05T14:30:00)."""
    return datetime.now().isoformat(timespec="seconds")