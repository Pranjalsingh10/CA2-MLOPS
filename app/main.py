from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"
model_bundle: Any = joblib.load(MODEL_PATH)
if not isinstance(model_bundle, dict) or not {"model", "scaler"} <= model_bundle.keys():
    raise RuntimeError("models/model.joblib must contain 'model' and 'scaler' entries")

model = model_bundle["model"]
scaler = model_bundle["scaler"]
if getattr(scaler, "n_features_in_", None) != len(FEATURE_COLUMNS):
    raise RuntimeError("The saved scaler does not match the 11 expected wine features")

app = FastAPI(
    title="Red Wine Quality API",
    description="Predict red wine quality using the model trained in the case study.",
    version="1.0.0",
)


class WineFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixed_acidity: float = Field(allow_inf_nan=False)
    volatile_acidity: float = Field(allow_inf_nan=False)
    citric_acid: float = Field(allow_inf_nan=False)
    residual_sugar: float = Field(allow_inf_nan=False)
    chlorides: float = Field(allow_inf_nan=False)
    free_sulfur_dioxide: float = Field(allow_inf_nan=False)
    total_sulfur_dioxide: float = Field(allow_inf_nan=False)
    density: float = Field(allow_inf_nan=False)
    pH: float = Field(allow_inf_nan=False)
    sulphates: float = Field(allow_inf_nan=False)
    alcohol: float = Field(allow_inf_nan=False)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Red Wine Quality API",
        "docs": "/docs",
        "health": "/health",
        "prediction": "/predict",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(wine: WineFeatures) -> dict[str, Any]:
    values = wine.model_dump()
    row = pd.DataFrame(
        [[
            values["fixed_acidity"],
            values["volatile_acidity"],
            values["citric_acid"],
            values["residual_sugar"],
            values["chlorides"],
            values["free_sulfur_dioxide"],
            values["total_sulfur_dioxide"],
            values["density"],
            values["pH"],
            values["sulphates"],
            values["alcohol"],
        ]],
        columns=FEATURE_COLUMNS,
    )

    try:
        scaled_row = scaler.transform(row)
        quality = int(model.predict(scaled_row)[0])
        probabilities = model.predict_proba(scaled_row)[0]
    except (AttributeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="The saved model could not make a prediction") from exc

    return {
        "predicted_quality": quality,
        "probabilities": {
            str(int(label)): float(probability)
            for label, probability in zip(model.classes_, probabilities)
        },
    }
