import pandas as pd

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score
)


def evaluate_thresholds(
    y_true,
    probabilities,
    thresholds=None
) -> pd.DataFrame:
    """
    Avalia diferentes thresholds de classificação.

    Parameters
    ----------
    y_true:
        Valores reais.
    probabilities:
        Probabilidades previstas para a classe positiva.
    thresholds:
        Limiares que serão testados.

    Returns
    -------
    pd.DataFrame
        Tabela com Precision, Recall e F1-score.
    """
    if thresholds is None:
        thresholds = [
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90
        ]

    results = []

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        results.append({
            "Threshold": threshold,
            "Precision": precision_score(
                y_true,
                predictions,
                zero_division=0
            ),
            "Recall": recall_score(
                y_true,
                predictions,
                zero_division=0
            ),
            "F1-score": f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        })

    return pd.DataFrame(results)