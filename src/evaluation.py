import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str,
) -> tuple[dict, object, object, object]:
    """
    Avalia um modelo de classificação.

    Retorna:
    - dicionário com métricas;
    - previsões;
    - probabilidades;
    - matriz de confusão.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "F1-score": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "PR-AUC": average_precision_score(y_test, y_prob)
    }

    matrix = confusion_matrix(y_test, y_pred)

    return metrics, y_pred, y_prob, matrix


def create_metrics_table(
    results: list[dict],
) -> pd.DataFrame:
    """
    Converte as métricas dos modelos em uma tabela comparativa.
    """
    return (
        pd.DataFrame(results)
        .sort_values(
            by="Recall",
            ascending=False
        )
        .reset_index(drop=True)
    )