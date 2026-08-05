from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def train_logistic_regression(X_train, y_train):
    """
    Treina uma Regressão Logística com padronização e balanceamento.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ])

    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train):
    """
    Treina um modelo Random Forest com balanceamento das classes.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def train_xgboost(X_train, y_train):
    """
    Treina um XGBoost considerando o desbalanceamento das classes.
    """
    if XGBClassifier is None:
        raise ImportError(
            "XGBoost não está instalado. "
            "Execute: pip install xgboost"
        )

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    if positive_count == 0:
        raise ValueError(
            "O conjunto de treino não possui exemplos de fraude."
        )

    scale_pos_weight = negative_count / positive_count

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model