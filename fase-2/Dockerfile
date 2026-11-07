# Imagen base de Python con versión explícita (la misma versión menor del entorno local)
FROM python:3.13-slim

# Configuración de Python dentro del contenedor
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# 1. Dependencias primero (se reutilizan en caché si requirements.txt no cambia)
COPY requirements.txt .
RUN pip install --default-timeout=120 --retries 10 -r requirements.txt

# 2. Código de la aplicación
COPY src/ ./src/
COPY train.py predict.py ./

# Los datos, los modelos y los resultados NO se copian a la imagen:
# se montan desde el equipo anfitrión al ejecutar el contenedor.
# El modelo NO se entrena durante la construcción.

# Comando por defecto: mostrar la ayuda del entrenamiento
CMD ["python", "train.py", "--help"]