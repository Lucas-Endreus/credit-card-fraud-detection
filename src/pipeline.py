import pandas as pd

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
)
from src.data_loader import get_dataset
from src.evaluation import (
    create_metrics_table,
    evaluate_model,
)
from src.model_io import save_model
from src.models import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)
from src.preprocessing import (
    prepare_data,
    split_features_target,
    split_train_test,
)
from src.threshold import evaluate_thresholds
from src.visualization import (
    save_confusion_matrix,
    save_model_comparison,
    save_precision_recall_curve,
    save_roc_curve,
    save_threshold_comparison,
)


def run_pipeline() -> pd.DataFrame:
    """
    Executa o pipeline completo de detecção de fraudes.

    Etapas:
    1. Carrega o dataset.
    2. Prepara os dados.
    3. Divide os dados entre treino e teste.
    4. Treina os modelos.
    5. Avalia os modelos.
    6. Salva os modelos treinados.
    7. Testa diferentes thresholds.
    8. Salva métricas e gráficos.

    Returns
    -------
    pd.DataFrame
        Tabela comparativa com as métricas dos modelos.
    """
    print("=" * 60)
    print("PIPELINE DE DETECÇÃO DE FRAUDES")
    print("=" * 60)

    # 1. Carregamento dos dados
    print("\n1. Carregando os dados...")

    dataframe = get_dataset()

    print(f"Registros carregados: {len(dataframe)}")

    # 2. Preparação dos dados
    print("\n2. Preparando os dados...")

    prepared_dataframe = prepare_data(dataframe)

    X, y = split_features_target(prepared_dataframe)

    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
    )

    print(f"Registros de treino: {len(X_train)}")
    print(f"Registros de teste: {len(X_test)}")

    # 3. Treinamento dos modelos
    print("\n3. Treinando os modelos...")

    models = {
        "Logistic Regression": train_logistic_regression(
            X_train,
            y_train,
        ),
        "Random Forest": train_random_forest(
            X_train,
            y_train,
        ),
        "XGBoost": train_xgboost(
            X_train,
            y_train,
        ),
    }

    metrics_results = []
    model_probabilities = {}

    # Garante a criação das pastas necessárias
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 4. Avaliação e salvamento dos modelos
    print("\n4. Avaliando os modelos...")

    for model_name, model in models.items():
        print(f"   Avaliando: {model_name}")

        metrics, _, probabilities, matrix = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
        )

        metrics_results.append(metrics)
        model_probabilities[model_name] = probabilities

        # Cria um nome seguro para arquivos
        safe_name = (
            model_name
            .lower()
            .replace(" ", "_")
        )

        # Salva o modelo treinado
        save_model(
            model=model,
            destination=(
                MODELS_DIR
                / f"{safe_name}.pkl"
            ),
        )

        # Salva a matriz de confusão
        save_confusion_matrix(
            matrix=matrix,
            model_name=model_name,
            destination=(
                FIGURES_DIR
                / f"confusion_matrix_{safe_name}.png"
            ),
        )

    # 5. Avaliação dos thresholds
    print("\n5. Avaliando diferentes thresholds...")

    for model_name, probabilities in model_probabilities.items():
        print(f"   Avaliando thresholds: {model_name}")

        threshold_table = evaluate_thresholds(
            y_true=y_test,
            probabilities=probabilities,
        )

        safe_name = (
            model_name
            .lower()
            .replace(" ", "_")
        )

        threshold_path = (
            METRICS_DIR
            / f"thresholds_{safe_name}.csv"
        )

        threshold_table.to_csv(
            threshold_path,
            index=False,
        )

        save_threshold_comparison(
            threshold_table=threshold_table,
            destination=(
                FIGURES_DIR
                / f"thresholds_{safe_name}.png"
            ),
            model_name=model_name,
        )

    # 6. Criação da tabela comparativa
    metrics_table = create_metrics_table(
        metrics_results,
    )

    # 7. Salvamento dos resultados
    print("\n6. Salvando resultados...")

    metrics_path = (
        METRICS_DIR
        / "model_comparison.csv"
    )

    metrics_table.to_csv(
        metrics_path,
        index=False,
    )

    save_roc_curve(
        y_test=y_test,
        model_probabilities=model_probabilities,
        destination=(
            FIGURES_DIR
            / "roc_curve.png"
        ),
    )

    save_precision_recall_curve(
        y_test=y_test,
        model_probabilities=model_probabilities,
        destination=(
            FIGURES_DIR
            / "precision_recall_curve.png"
        ),
    )

    save_model_comparison(
        metrics_table=metrics_table,
        destination=(
            FIGURES_DIR
            / "model_comparison.png"
        ),
    )

    print("\nPipeline concluído com sucesso!")

    print(f"\nMétricas salvas em:\n{metrics_path}")
    print(f"\nGráficos salvos em:\n{FIGURES_DIR}")
    print(f"\nModelos salvos em:\n{MODELS_DIR}")

    print("\nComparação dos modelos:")

    print(
        metrics_table.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    return metrics_table