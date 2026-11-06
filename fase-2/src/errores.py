"""
Errores propios de la aplicación.

Representan problemas previsibles (datos inválidos, modelo inexistente,
versión repetida). Los scripts los capturan para mostrar un mensaje
comprensible y terminar con un código de salida distinto de cero.
"""


class ErrorAplicacion(Exception):
    """Error previsible de la aplicación, con un mensaje claro para el usuario."""


class ErrorDatos(ErrorAplicacion):
    """Problema con el archivo de datos: no existe, faltan columnas o tipos inválidos."""


class ErrorModelo(ErrorAplicacion):
    """Problema con el modelo: no existe o no se puede cargar."""


class ErrorVersion(ErrorAplicacion):
    """Problema con la versión del modelo: identificador inválido o repetido."""