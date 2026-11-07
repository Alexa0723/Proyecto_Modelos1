"""
Configuración compartida de las pruebas.

Define rutas del proyecto y fixtures con datos pequeños, para que las pruebas
sean rápidas, no descarguen nada de internet y no modifiquen los modelos de
la entrega (todo lo que generan se guarda en carpetas temporales).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Carpeta fase-2 (la carpeta de arriba de tests/)
RUTA_FASE_2 = Path(__file__).resolve().parent.parent

# Permite importar src/ y train.py desde las pruebas
if str(RUTA_FASE_2) not in sys.path:
    sys.path.insert(0, str(RUTA_FASE_2))

RUTA_DATASET = RUTA_FASE_2 / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RUTA_EJEMPLO_PREDICCION = RUTA_FASE_2 / "data" / "ejemplo_prediccion.csv"
CANTIDAD_FILAS_MUESTRA = 300


@pytest.fixture
def ruta_datos_pequenos(tmp_path):
    """Crea un CSV temporal con una muestra pequeña del dataset y devuelve su ruta."""
    datos_completos = pd.read_csv(RUTA_DATASET)
    muestra = datos_completos.sample(n=CANTIDAD_FILAS_MUESTRA, random_state=0)

    ruta_muestra = tmp_path / "datos_pequenos.csv"
    muestra.to_csv(ruta_muestra, index=False)

    return ruta_muestra


@pytest.fixture
def datos_ejemplo():
    """Devuelve el CSV de ejemplo para predicción (5 clientes, sin la variable objetivo)."""
    return pd.read_csv(RUTA_EJEMPLO_PREDICCION)