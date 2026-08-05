from pathlib import Path

import pandas as pd

from src.config import DATA_URL, RAW_DATA_PATH


def download_dataset(
    url: str = DATA_URL,
    destination: Path = RAW_DATA_PATH,
) -> Path:
    """
    Baixa o dataset apenas quando ele ainda não existe localmente.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        print(f"Dataset já encontrado em: {destination}")
        return destination

    print("Baixando o dataset...")

    dataframe = pd.read_csv(url)
    dataframe.to_csv(destination, index=False)

    print(f"Dataset salvo em: {destination}")

    return destination


def load_dataset(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Carrega o arquivo CSV e retorna um DataFrame.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            "Dataset não encontrado. Execute download_dataset() primeiro."
        )

    dataframe = pd.read_csv(file_path)

    if dataframe.empty:
        raise ValueError("O dataset carregado está vazio.")

    return dataframe


def get_dataset() -> pd.DataFrame:
    """
    Garante o download do dataset e retorna os dados carregados.
    """
    file_path = download_dataset()
    return load_dataset(file_path)