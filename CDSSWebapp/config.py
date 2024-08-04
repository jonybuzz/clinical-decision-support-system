# General Configurations
UPLOAD_FOLDER = 'uploads'
USERS = {'admin': '123'}

# Flask Configurations
DEBUG = True
TESTING = False
SESSION_COOKIE_NAME = 'cdss_session'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB upload limit
PORT = 5000

# Processing Configurations
DATASET_TARGET = "depression_diagnosis"  # columna destino Boolean con resultado esDepresivo
SESSION_ID = str(uuid.uuid7())  # numero random
TMP_DATA_FOLDER = "tmp_data"
VALIDATION_RATIO = 0.2

# Database Configurations
