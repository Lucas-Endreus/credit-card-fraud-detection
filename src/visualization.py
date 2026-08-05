from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve
)

def save_confusion_matrix(
    matrix,
    model_name: str,
    destination: Path,
) -> None:
    """
    Salva a matriz de confusão como imagem.
    """
    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False
    )

    plt.title(f"Matriz de Confusão — {model_name}")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(destination, dpi=300)
    plt.close()


def save_roc_curve(
    y_test,
    model_probabilities: dict[str, object],
    destination: Path,
) -> None:
    """
    Salva as curvas ROC dos modelos.
    """
    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(8, 6))

    for model_name, probabilities in model_probabilities.items():
        fpr, tpr, _ = roc_curve(
            y_test,
            probabilities
        )

        plt.plot(
            fpr,
            tpr,
            label=model_name
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Classificador aleatório"
    )

    plt.title("Comparação das Curvas ROC")
    plt.xlabel("Taxa de falsos positivos")
    plt.ylabel("Taxa de verdadeiros positivos")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=300)
    plt.close()


def save_precision_recall_curve(
    y_test,
    model_probabilities: dict[str, object],
    destination: Path,
) -> None:
    """
    Salva as curvas Precision-Recall dos modelos.
    """
    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(8, 6))

    for model_name, probabilities in model_probabilities.items():
        precision, recall, _ = precision_recall_curve(
            y_test,
            probabilities
        )

        plt.plot(
            recall,
            precision,
            label=model_name
        )

    plt.title("Comparação das Curvas Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=300)
    plt.close()
def save_threshold_comparison(
    threshold_table: pd.DataFrame,
    destination: Path,
    model_name: str
) -> None:
    """
    Salva a comparação das métricas para diferentes thresholds.
    """
    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plot_data = threshold_table.set_index(
        "Threshold"
    )[
        [
            "Precision",
            "Recall",
            "F1-score"
        ]
    ]

    axes = plot_data.plot(
        figsize=(10, 6),
        marker="o"
    )

    axes.set_title(
        f"Ajuste de Threshold — {model_name}"
    )
    axes.set_xlabel("Threshold")
    axes.set_ylabel("Resultado")
    axes.set_ylim(0, 1.05)
    axes.grid(True)

    plt.tight_layout()
    plt.savefig(destination, dpi=300)
    plt.close()

def save_model_comparison(
    metrics_table: pd.DataFrame,
    destination: Path,
) -> None:
    """
    Salva um gráfico comparando as métricas dos modelos.
    """
    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plot_data = metrics_table.set_index("Model")[
        [
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC",
            "PR-AUC"
        ]
    ]

    axes = plot_data.plot(
        kind="bar",
        figsize=(12, 6)
    )

    axes.set_title("Comparação entre Modelos")
    axes.set_xlabel("Modelo")
    axes.set_ylabel("Resultado")
    axes.set_ylim(0, 1.05)

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(destination, dpi=300)
    plt.close()