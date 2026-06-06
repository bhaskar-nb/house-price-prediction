from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


@st.cache_resource
def get_model(model_path: str):
    """Load and cache the trained model once per session."""
    return joblib.load(model_path)


@st.cache_data
def get_reference_data(data_path: str) -> pd.DataFrame:
    """Load and cache reference dataset for defaults and insights."""
    try:
        return pd.read_csv(data_path)
    except Exception:
        return pd.DataFrame()


def yes_no_to_int(value: str) -> int:
    return 1 if str(value).strip().lower() == "yes" else 0


def prepare_features(inputs: dict[str, Any]) -> np.ndarray:
    """Keep feature order compatible with the existing model."""
    furnishing = str(inputs.get("furnishingstatus", "unfurnished")).strip().lower()
    furnished = 1 if furnishing == "furnished" else 0
    semi_furnished = 1 if furnishing == "semi-furnished" else 0

    features = np.array(
        [
            [
                float(inputs.get("area", 0)),
                float(inputs.get("bedrooms", 0)),
                float(inputs.get("bathrooms", 0)),
                float(inputs.get("stories", 0)),
                yes_no_to_int(inputs.get("mainroad", "no")),
                yes_no_to_int(inputs.get("guestroom", "no")),
                yes_no_to_int(inputs.get("basement", "no")),
                yes_no_to_int(inputs.get("hotwaterheating", "no")),
                yes_no_to_int(inputs.get("airconditioning", "no")),
                float(inputs.get("parking", 0)),
                yes_no_to_int(inputs.get("prefarea", "no")),
                furnished,
                semi_furnished,
            ]
        ]
    )
    return features


@st.cache_data(show_spinner=False)
def predict_price(_model, inputs: dict[str, Any]) -> float:
    features = prepare_features(inputs)
    prediction = _model.predict(features)
    return float(prediction[0])


def validate_inputs(inputs: dict[str, Any]) -> tuple[bool, str]:
    if float(inputs.get("area", 0)) <= 0:
        return False, "Area must be greater than 0."
    if float(inputs.get("bedrooms", 0)) < 0 or float(inputs.get("bathrooms", 0)) < 0:
        return False, "Bedrooms and bathrooms cannot be negative."
    if float(inputs.get("stories", 0)) < 0 or float(inputs.get("parking", 0)) < 0:
        return False, "Stories and parking cannot be negative."
    return True, ""


def format_currency(value: float) -> str:
    return f"{value:,.2f}"


@st.cache_data
def get_feature_importance_chart() -> go.Figure:
    """Placeholder interactive chart to explain model behavior."""
    features = [
        "Area",
        "Bathrooms",
        "Air Conditioning",
        "Preferred Area",
        "Bedrooms",
        "Stories",
        "Main Road",
    ]
    weights = [0.34, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06]
    df = pd.DataFrame({"Feature": features, "Relative Impact": weights})
    fig = px.bar(
        df.sort_values("Relative Impact"),
        x="Relative Impact",
        y="Feature",
        orientation="h",
        color="Relative Impact",
        color_continuous_scale="Blues",
        title="Estimated Feature Influence",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), coloraxis_showscale=False)
    return fig
