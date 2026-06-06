import streamlit as st

from components.ui import render_sidebar_inputs, render_upload_and_batch_predict
from utils import (
    format_currency,
    get_feature_importance_chart,
    get_model,
    get_reference_data,
    predict_price,
    validate_inputs,
)


st.set_page_config(
    page_title="House Price Predictor Pro",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    st.title("🏠 House Price Predictor Pro")
    st.caption("Modern, interactive valuation assistant powered by machine learning.")
    st.divider()

    model = get_model("model.pkl")
    reference_df = get_reference_data("Housing.csv")

    with st.sidebar:
        st.header("Property Inputs")
        user_inputs, advanced = render_sidebar_inputs(reference_df)
        st.divider()
        batch_df = render_upload_and_batch_predict()

    tabs = st.tabs(["Single Prediction", "Batch Predictions", "Model Insights"])

    with tabs[0]:
        st.subheader("Single Property Prediction")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "Area (sq ft)",
                f"{int(user_inputs['area']):,}",
                help="Built-up area in square feet",
            )
        with c2:
            st.metric(
                "Bedrooms",
                int(user_inputs["bedrooms"]),
                help="Total number of bedrooms",
            )
        with c3:
            st.metric(
                "Bathrooms",
                int(user_inputs["bathrooms"]),
                help="Total number of bathrooms",
            )

        col_left, col_right = st.columns([2, 1], gap="large")
        with col_left:
            with st.container(border=True):
                st.markdown("#### Predict Price")
                is_valid, error_message = validate_inputs(user_inputs)
                if error_message:
                    st.error(error_message)

                if st.button("Predict Price", type="primary", use_container_width=True):
                    if not is_valid:
                        st.warning("Please fix input issues before predicting.")
                    else:
                        with st.spinner("Running model inference..."):
                            prediction = predict_price(model, user_inputs)

                        st.success(f"Predicted Price: {format_currency(prediction)}")
                        st.balloons()
                        st.session_state["latest_prediction"] = prediction

        with col_right:
            with st.container(border=True):
                st.markdown("#### Quick Insights")
                if "latest_prediction" in st.session_state:
                    base = float(reference_df["price"].median()) if "price" in reference_df else 0
                    delta = st.session_state["latest_prediction"] - base
                    st.metric(
                        "Predicted Value",
                        format_currency(st.session_state["latest_prediction"]),
                        delta=f"{delta:,.0f}",
                        delta_color="normal",
                        help="Delta compares to median price in historical data.",
                    )
                else:
                    st.info("Run a prediction to view value insights.")

                st.caption(
                    f"Advanced mode: {'On' if advanced['show_details'] else 'Off'} | "
                    f"Theme: {advanced['theme']}"
                )

    with tabs[1]:
        st.subheader("Batch Predictions")
        st.write("Upload a CSV/Excel file from the sidebar to score multiple properties.")
        if batch_df is None:
            st.info("No file uploaded yet.")
        else:
            required_cols = [
                "area",
                "bedrooms",
                "bathrooms",
                "stories",
                "mainroad",
                "guestroom",
                "basement",
                "hotwaterheating",
                "airconditioning",
                "parking",
                "prefarea",
                "furnishingstatus",
            ]
            missing = [col for col in required_cols if col not in batch_df.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                with st.spinner("Scoring uploaded file..."):
                    scored_df = batch_df.copy()
                    scored_df["predicted_price"] = scored_df.apply(
                        lambda row: predict_price(model, row.to_dict()), axis=1
                    )
                st.success("Batch scoring complete.")
                st.dataframe(scored_df.head(50), use_container_width=True)
                st.download_button(
                    "Download Scored Results (CSV)",
                    data=scored_df.to_csv(index=False).encode("utf-8"),
                    file_name="house_price_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    with tabs[2]:
        st.subheader("Model Insights")
        left, right = st.columns([2, 1], gap="large")
        with left:
            st.plotly_chart(
                get_feature_importance_chart(),
                use_container_width=True,
                theme="streamlit",
            )
        with right:
            with st.container(border=True):
                st.markdown("#### How to Read This")
                st.write(
                    "- Longer bars indicate larger influence on prediction.\n"
                    "- Area and quality/accessibility indicators typically drive value.\n"
                    "- Use these insights to understand model behavior, not as strict rules."
                )

    st.divider()
    st.caption(
        "Built with Streamlit | Production-focused UI, caching, and robust input handling."
    )


if __name__ == "__main__":
    main()