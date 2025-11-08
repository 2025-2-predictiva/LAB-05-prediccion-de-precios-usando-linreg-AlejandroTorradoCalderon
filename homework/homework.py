import os
import gzip
import json
import pickle
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

class Settings:
    """Configuración centralizada de rutas, columnas y parámetros del modelo."""

    BASE_PATH = Path(__file__).resolve().parent.parent
    INPUT_PATH = BASE_PATH / "files" / "input"
    MODEL_PATH = BASE_PATH / "files" / "models"
    OUTPUT_PATH = BASE_PATH / "files" / "output"

    TRAIN_FILE_ZIP = INPUT_PATH / "train_data.csv.zip"
    TEST_FILE_ZIP = INPUT_PATH / "test_data.csv.zip"
    TRAIN_FILE = "train_data.csv"
    TEST_FILE = "test_data.csv"

    CURRENT_YEAR = 2021
    TARGET = "Present_Price"
    CATEGORICAL = ["Fuel_Type", "Selling_type", "Transmission"]
    NUMERIC = ["Driven_kms", "Owner", "Age"]
    PRICE_FEATURE = ["Selling_Price"]

    GRID_PARAMETERS = {
        "kbest__k": list(range(4, 12))  # desde 4 hasta 11
    }

def load_csv_from_zip(zip_path: Path, csv_filename: str) -> pd.DataFrame:

    with zipfile.ZipFile(zip_path, "r") as archive:
        with archive.open(csv_filename) as file:
            df = pd.read_csv(file)
            if df.columns[0].startswith("Unnamed"):
                df = df.drop(columns=df.columns[0])
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    data = df.copy()
    data["Age"] = Settings.CURRENT_YEAR - data["Year"]
    data.drop(columns=["Year", "Car_Name"], inplace=True)
    data.dropna(inplace=True)
    data["Selling_Price"] = np.log1p(data["Selling_Price"])
    return data


def build_pipeline() -> GridSearchCV:

    transformer = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), Settings.CATEGORICAL),
            ("scale", MinMaxScaler(), Settings.NUMERIC),
            ("price_passthrough", "passthrough", Settings.PRICE_FEATURE),
        ],
        remainder="drop",
    )

    model_pipeline = Pipeline(
        steps=[
            ("prep", transformer),
            ("kbest", SelectKBest(score_func=f_regression)),
            ("linreg", LinearRegression()),
        ]
    )

    search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=Settings.GRID_PARAMETERS,
        cv=10,
        scoring="neg_mean_absolute_error",
        refit=True,
        verbose=1,
    )

    return search


def compute_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:

    return {
        "type": "metrics",
        "dataset": name,
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mad": float(mean_absolute_error(y_true, y_pred)),
    }


def save_model(model: Any, path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as file:
        pickle.dump(model, file)


def save_metrics(metrics: List[Dict[str, Any]], path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in metrics:
            f.write(json.dumps(entry) + "\n")

def main():


    df_train = preprocess_data(load_csv_from_zip(Settings.TRAIN_FILE_ZIP, Settings.TRAIN_FILE))
    df_test = preprocess_data(load_csv_from_zip(Settings.TEST_FILE_ZIP, Settings.TEST_FILE))


    X_train = df_train.drop(Settings.TARGET, axis=1)
    y_train = np.log1p(df_train[Settings.TARGET])
    X_test = df_test.drop(Settings.TARGET, axis=1)
    y_test = np.log1p(df_test[Settings.TARGET])

    model_search = build_pipeline()
    model_search.fit(X_train, y_train)

    model_path = Settings.MODEL_PATH / "model.pkl.gz"
    save_model(model_search, model_path)

    y_train_pred = np.expm1(model_search.predict(X_train))
    y_test_pred = np.expm1(model_search.predict(X_test))
    y_train_real = np.expm1(y_train)
    y_test_real = np.expm1(y_test)

    metrics_train = compute_metrics("train", y_train_real, y_train_pred)
    metrics_test = compute_metrics("test", y_test_real, y_test_pred)

    metrics_path = Settings.OUTPUT_PATH / "metrics.json"
    save_metrics([metrics_train, metrics_test], metrics_path)


if __name__ == "__main__":
    main()

#############################################################################