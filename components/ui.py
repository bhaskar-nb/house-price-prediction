from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd
import streamlit as st


def _default_value(df: pd.DataFrame, col: str, fallback: int) -> int:
    if df.empty or col not in df.columns:
        return fallback
    try:
        return int(df[col].median())
    except Exception:
        return fallback


def render_sidebar_inputs(reference_df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    st.markdown("Adjust property details and run predictions.")
    with st.expander("Basic Property Inputs", expanded=True):
        area = st.slider(
            "Area (sq ft)",
            min_value=100,
            max_value=20000,
            value=_default_value(reference_df, "area", 1000),
            step=50,
            help="Use built-up area in square feet.",
        )
        bedrooms = st.selectbox("Bedrooms", options=list(range(0, 11)), index=2)
        bathrooms = st.selectbox("Bathrooms", options=list(range(0, 11)), index=1)
        stories = st.selectbox("Stories", options=list(range(0, 6)), index=1)
        parking = st.selectbox("Parking Spots", options=list(range(0, 6)), index=0)

    with st.expander("Amenities and Location", expanded=True):
        mainroad = st.selectbox("Main Road Access", ["yes", "no"], help="Road frontage can impact value.")
        guestroom = st.selectbox("Guest Room", ["yes", "no"])
        basement = st.selectbox("Basement", ["yes", "no"])
        hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
        airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
        prefarea = st.selectbox(
            "Preferred Area",
            ["yes", "no"],
            help="Marks higher-demand neighborhood zones.",
        )
        furnishingstatus = st.selectbox(
            "Furnishing Status",
            ["furnished", "semi-furnished", "unfurnished"],
        )

    with st.expander("Advanced Settings", expanded=False):
        show_details = st.toggle("Show detailed model insights", value=True)
        theme = st.radio("Visual Theme", ["Professional", "Compact"], index=0, horizontal=True)

    inputs = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus,
    }
    advanced = {"show_details": show_details, "theme": theme}
    return inputs, advanced


def render_upload_and_batch_predict() -> pd.DataFrame | None:
    st.header("Batch Scoring")
    uploaded = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        help="Upload file with required model columns for batch predictions.",
    )
    if not uploaded:
        return None

    try:
        if uploaded.name.endswith(".csv"):
            return pd.read_csv(uploaded)
        file_data = BytesIO(uploaded.read())
        return pd.read_excel(file_data)
    except Exception as exc:
        st.error(f"Could not parse uploaded file: {exc}")
        return None
