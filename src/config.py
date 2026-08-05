from pathlib import Path


# Pasta principal do projeto
BASE_DIR = Path(__file__).resolve().parent.parent

# Pastas de dados
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Pastas de resultados
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
MODELS_DIR = BASE_DIR / "models"
# Dataset utilizado no projeto
DATA_URL = (
    "https://storage.googleapis.com/"
    "download.tensorflow.org/data/creditcard.csv"
)

# Caminhos dos arquivos
RAW_DATA_PATH = RAW_DATA_DIR / "creditcard.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "creditcard_processed.csv"

# Configurações gerais
RANDOM_STATE = 42
TEST_SIZE = 0.30
TARGET_COLUMN = "Class"