
import csv
import math
import os
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = Path(
    os.getenv("DATASET_PATH", str(BASE_DIR / "fertility.csv"))
)

MODEL_PATH = BASE_DIR / "model.joblib"

TARGET_COLUMN = os.getenv("TARGET_COLUMN", "Fertility Rates")

# The dataset's index is an identifier, not a useful predictor.
EXCLUDED_COLUMNS = {"index"}

app = FastAPI(
    title="Fertility Rate MLOps API",
    description=(
        "Predict fertility rates using a machine-learning model. "
        "Includes data preprocessing, model training, evaluation, "
        "health checks, and prediction endpoints."
    ),
    version="1.0.0",
)

model = None
feature_columns: list[str] = []
numeric_features: list[str] = []
categorical_features: list[str] = []
training_metrics: dict[str, Any] = {}
dataset_rows = 0
startup_error = None


# ============================================================
# CSV LOADING — NO PANDAS
# ============================================================

def read_csv_file(file_path: Path) -> list[dict[str, str]]:
    """Load CSV data using Python's built-in csv module."""

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    with file_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("CSV file has no header row.")

        reader.fieldnames = [
            name.strip() for name in reader.fieldnames
        ]

        rows = []

        for row in reader:
            cleaned = {
                str(key).strip(): (
                    value.strip() if value is not None else ""
                )
                for key, value in row.items()
                if key is not None
            }

            if any(value != "" for value in cleaned.values()):
                rows.append(cleaned)

    if not rows:
        raise ValueError("CSV file contains no data rows.")

    return rows


def to_number(value: Any) -> float:
    """Convert a value to a finite float or return NaN."""

    if value is None:
        return math.nan

    text = str(value).strip()

    if text.lower() in {"", "na", "n/a", "nan", "null", "none"}:
        return math.nan

    try:
        number = float(text)
    except (TypeError, ValueError):
        return math.nan

    return number if math.isfinite(number) else math.nan


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_dataset(rows):
    """
    Prepare the dataset without Pandas.

    Features:
        Year, Hispanic Origin, Live Births, Birth Rates

    Target:
        Fertility Rates
    """

    global feature_columns
    global numeric_features
    global categorical_features

    headers = list(rows[0].keys())

    target = next(
        (
            name for name in headers
            if name.lower() == TARGET_COLUMN.lower()
        ),
        None,
    )

    if target is None:
        raise ValueError(
            f"Target '{TARGET_COLUMN}' not found. "
            f"Available columns: {headers}"
        )

    # Do not use the index column as a predictor.
    candidates = [
        name for name in headers
        if name != target and name.lower() not in EXCLUDED_COLUMNS
    ]

    if not candidates:
        raise ValueError("No feature columns were found.")

    valid_rows = []
    targets = []

    for row in rows:
        y_value = to_number(row.get(target))

        # The target must be numeric.
        if math.isnan(y_value):
            continue

        valid_rows.append(row)
        targets.append(y_value)

    if len(valid_rows) < 2:
        raise ValueError(
            "At least two rows with valid target values are required."
        )

    # Identify columns by their actual values.
    numeric_features = []
    categorical_features = []

    for column in candidates:
        nonempty_values = [
            row.get(column, "").strip()
            for row in valid_rows
            if row.get(column, "").strip() != ""
        ]

        if not nonempty_values:
            continue

        if all(not math.isnan(to_number(value)) for value in nonempty_values):
            numeric_features.append(column)
        else:
            categorical_features.append(column)

    feature_columns = numeric_features + categorical_features

    if not feature_columns:
        raise ValueError("No usable feature columns were found.")

    # Numeric values become floats; categorical values remain strings.
    X = []

    for row in valid_rows:
        feature_row = []

        for column in numeric_features:
            feature_row.append(to_number(row.get(column)))

        for column in categorical_features:
            value = row.get(column, "").strip()
            feature_row.append(value if value else None)

        X.append(feature_row)

    return X, targets


# ============================================================
# MODEL PIPELINE
# ============================================================

def build_pipeline():
    """Create preprocessing and regression steps."""

    transformers = []

    numeric_count = len(numeric_features)
    categorical_count = len(categorical_features)

    # Numeric columns are placed first in each feature row.
    if numeric_count:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                )
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                list(range(numeric_count)),
            )
        )

    if categorical_count:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="Unknown",
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                list(
                    range(
                        numeric_count,
                        numeric_count + categorical_count,
                    )
                ),
            )
        )

    preprocessing = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                ),
            ),
        ]
    )


# ============================================================
# MODEL TRAINING
# ============================================================

def train_model():
    """Train the model and save its metadata."""

    global model
    global training_metrics
    global dataset_rows

    rows = read_csv_file(DATASET_PATH)
    X, y = prepare_dataset(rows)

    dataset_rows = len(y)

    pipeline = build_pipeline()

    if len(X) >= 5:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        evaluation_model = build_pipeline()
        evaluation_model.fit(X_train, y_train)

        predictions = evaluation_model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)

        training_metrics = {
            "evaluation_rows": len(y_test),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(math.sqrt(mse)),
            "r2": (
                float(r2_score(y_test, predictions))
                if len(y_test) >= 2
                else None
            ),
        }
    else:
        training_metrics = {
            "evaluation_rows": 0,
            "mae": None,
            "rmse": None,
            "r2": None,
            "note": "Dataset too small for holdout evaluation.",
        }

    # Train the final model using all valid records.
    pipeline.fit(X, y)
    model = pipeline

    # Save model and metadata for future use.
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "target_column": TARGET_COLUMN,
        },
        MODEL_PATH,
    )

    print("Model trained successfully.")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Valid rows: {dataset_rows}")
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    print(f"Target: {TARGET_COLUMN}")
    print(f"Evaluation metrics: {training_metrics}")
    print(f"Saved model: {MODEL_PATH}")


# ============================================================
# API REQUEST SCHEMA
# ============================================================

class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description=(
            "Provide the feature names and values returned by /features. "
            "Categorical features should be strings."
        ),
        examples=[
            {
                "Year": 2020,
                "Hispanic Origin": "All origins",
                "Live Births": 3600000,
                "Birth Rates": 10.9,
            }
        ],
    )


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():
    """Train a fresh model from the CSV on startup."""

    global startup_error

    try:
        # Deliberately retrain instead of loading the old model,
        # which may be incompatible with the installed sklearn version.
        train_model()
        startup_error = None

    except Exception as exc:
        startup_error = str(exc)
        print(f"Model startup failed: {startup_error}")


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Fertility Rate MLOps API is running.",
        "model_loaded": model is not None,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "model_unavailable",
        "model_loaded": model is not None,
        "dataset_path": str(DATASET_PATH),
        "dataset_rows": dataset_rows,
        "startup_error": startup_error,
    }


@app.get("/features")
def get_features():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable: {startup_error}",
        )

    return {
        "features": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": TARGET_COLUMN,
    }


@app.get("/metrics")
def get_metrics():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable: {startup_error}",
        )

    return {
        "model": "RandomForestRegressor",
        "dataset_rows": dataset_rows,
        "metrics": training_metrics,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable: {startup_error}",
        )

    missing = [
        column
        for column in feature_columns
        if column not in request.features
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Required features are missing.",
                "missing_features": missing,
                "expected_features": feature_columns,
            },
        )

    try:
        input_row = []

        # Keep the same column order used during training.
        for column in numeric_features:
            value = to_number(request.features[column])

            if math.isnan(value):
                raise ValueError(
                    f"Feature '{column}' must contain a valid number."
                )

            input_row.append(value)

        for column in categorical_features:
            value = request.features[column]

            if value is None:
                input_row.append(None)
            else:
                input_row.append(str(value))

        prediction = float(model.predict([input_row])[0])

        return {
            "prediction": prediction,
            "target": TARGET_COLUMN,
            "model": "RandomForestRegressor",
        }

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc


@app.post("/reload")
def reload_model():
    """Retrain the model from the current dataset."""

    global startup_error

    try:
        train_model()
        startup_error = None

        return {
            "message": "Model retrained successfully.",
            "dataset_rows": dataset_rows,
            "features": feature_columns,
            "metrics": training_metrics,
        }

    except Exception as exc:
        startup_error = str(exc)

        raise HTTPException(
            status_code=500,
            detail=f"Model retraining failed: {startup_error}",
        ) from exc
