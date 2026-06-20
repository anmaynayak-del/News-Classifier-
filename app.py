from __future__ import annotations

import streamlit as st

from classifier import CATEGORIES, predict_category


EXAMPLE_HEADLINES = {
    "Sports": "India defeats Australia in a thrilling cricket final",
    "Business": "Stock markets rally after central bank keeps rates unchanged",
    "Technology": "New AI chip promises faster cloud computing for developers",
    "Politics": "Parliament debates new election reform bill",
}


st.set_page_config(
    page_title="News Category Classifier",
    page_icon="N",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f7f8fb;
    }
    .block-container {
        max-width: 1100px;
        padding-top: 2.5rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #17202a;
        letter-spacing: 0;
    }
    .app-subtitle {
        color: #526071;
        font-size: 1.05rem;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }
    .prediction-box {
        background: #ffffff;
        border: 1px solid #dde3ea;
        border-radius: 8px;
        padding: 1.2rem 1.3rem;
        box-shadow: 0 6px 18px rgba(24, 39, 75, 0.06);
    }
    .prediction-label {
        color: #526071;
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
    }
    .prediction-value {
        color: #0f6b4f;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .metric-note {
        color: #657386;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("News Category Classifier")
st.markdown(
    """
    <p class="app-subtitle">
    Classify a news headline or short article into Sports, Business, Technology,
    or Politics using TF-IDF text features and a scikit-learn machine learning model.
    </p>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.25, 1], gap="large")

with left_col:
    selected_example = st.selectbox(
        "Try a sample headline",
        options=["Write my own"] + list(EXAMPLE_HEADLINES.keys()),
    )

    default_text = ""
    if selected_example != "Write my own":
        default_text = EXAMPLE_HEADLINES[selected_example]

    news_text = st.text_area(
        "Headline or short article",
        value=default_text,
        height=180,
        placeholder="Example: Government announces new policy for digital payments",
    )

    predict_button = st.button("Predict category", type="primary", use_container_width=True)

with right_col:
    st.subheader("Prediction")

    if predict_button:
        try:
            category, confidence, scores = predict_category(news_text)
            st.markdown(
                f"""
                <div class="prediction-box">
                    <div class="prediction-label">Predicted category</div>
                    <div class="prediction-value">{category}</div>
                    <div class="metric-note">Confidence: {confidence:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            st.caption("Category score breakdown")
            for label in CATEGORIES:
                st.progress(scores.get(label, 0.0), text=f"{label}: {scores.get(label, 0.0):.1%}")
        except ValueError as exc:
            st.warning(str(exc))
    else:
        st.info("Enter text and run a prediction to see the result.")

st.divider()

st.subheader("Model Details")
detail_cols = st.columns(3)
detail_cols[0].metric("Text features", "TF-IDF")
detail_cols[1].metric("Classifier", "Logistic Regression")
detail_cols[2].metric("Categories", str(len(CATEGORIES)))

st.caption(
    "This demo trains a lightweight scikit-learn pipeline at startup from bundled sample headlines."
)
