import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def prepare_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Cria as variáveis utilizadas pelos modelos.

    A função:
    - cria a hora da transação;
    - aplica transformação logarítmica ao valor;
    - remove as colunas originais Time e Amount.
    """
    required_columns = {"Time", "Amount", TARGET_COLUMN}

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Colunas obrigatórias ausentes: {sorted(missing_columns)}"
        )

    prepared_dataframe = dataframe.copy()

    prepared_dataframe["Hour"] = (
        prepared_dataframe["Time"] / 3600
    ) % 24

    prepared_dataframe["Amount_log"] = np.log1p(
        prepared_dataframe["Amount"]
    )

    prepared_dataframe = prepared_dataframe.drop(
        columns=["Time", "Amount"]
    )

    return prepared_dataframe


def split_features_target(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa as variáveis explicativas da variável alvo.
    """
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"A coluna alvo '{TARGET_COLUMN}' não foi encontrada."
        )

    X = dataframe.drop(columns=[TARGET_COLUMN])
    y = dataframe[TARGET_COLUMN]

    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
):
    """
    Divide os dados entre treino e teste preservando a proporção das classes.
    """
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )