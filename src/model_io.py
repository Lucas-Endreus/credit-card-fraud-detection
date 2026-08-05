from pathlib import Path

import joblib


def save_model(
    model,
    destination: Path
):
    """
    Salva um modelo treinado.
    """
    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        destination
    )


def load_model(
    path: Path
):
    """
    Carrega um modelo salvo.
    """
    return joblib.load(path)