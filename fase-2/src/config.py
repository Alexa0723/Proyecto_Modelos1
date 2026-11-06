"""
Configuración central de la aplicación.

Define los nombres de las columnas del dataset Telco Customer Churn y los
valores por defecto que comparten train.py, predict.py y las pruebas.
"""

# Variable objetivo y codificación de sus clases
COLUMNA_OBJETIVO_POR_DEFECTO = "Churn"
VALOR_CLASE_POSITIVA = "Yes"   # El cliente abandonó  -> se codifica como 1
VALOR_CLASE_NEGATIVA = "No"    # El cliente permaneció -> se codifica como 0

# Columna que identifica a cada cliente (no es una variable predictora)
COLUMNA_IDENTIFICADOR = "customerID"

# Variables numéricas: imputación con la mediana y escalamiento
COLUMNAS_NUMERICAS = ["tenure", "MonthlyCharges"]

# TotalCharges: imputación con 0 (clientes nuevos sin cobros) y escalamiento
COLUMNA_CARGOS_TOTALES = ["TotalCharges"]

# Variables categóricas (incluye SeniorCitizen, que representa sí/no)
COLUMNAS_CATEGORICAS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

# Todas las variables predictoras, en el orden en que las recibe el modelo
COLUMNAS_PREDICTORAS = COLUMNAS_NUMERICAS + COLUMNA_CARGOS_TOTALES + COLUMNAS_CATEGORICAS

# Columnas que deben contener valores numéricos en el CSV
COLUMNAS_QUE_DEBEN_SER_NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

# Valores por defecto del entrenamiento
SEMILLA_POR_DEFECTO = 42
PROPORCION_PRUEBA_POR_DEFECTO = 0.2
NUMERO_PARTICIONES_VALIDACION_POR_DEFECTO = 5