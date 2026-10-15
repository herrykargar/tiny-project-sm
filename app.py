import streamlit as st
import pandas as pd
import plotly.express as px
from analyzer import load_posts, analyze_posts, hashtag_frequency, sentiment_summary

st.set_page_config(
    page_title="Social Media Hashtag Trend Analyzer",
    page_icon="#",
    layout="wide"
)

st.title("Social Media Hashtag Trend Analyzer")
st.caption("Analyze hashtag popularity and the sentiment of social media posts.")

with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    st.markdown("**Required column:** `text`")
    st.markdown("Optional columns: `date`, `platform`")
    st.info("No live social media API is used. Results are based only on the dataset you provide.")

@st.cache_data
def get_sample_data():
    return pd.read_csv("sample_posts.csv")

try:
    if uploaded_file is not None:
        raw_df = load_posts(uploaded_file)
        source_name = uploaded_file.name
    else:
        raw_df = load_posts("sample_posts.csv")
        source_name = "sample_posts.csv"

    if raw_df.empty:
        st.warning("The selected CSV has no rows to analyze.")
        st.stop()

    analyzed = analyze_posts(raw_df)
    hashtags = hashtag_frequency(analyzed)
    sentiment = sentiment_summary(analyzed)

    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total posts", f"{len(analyzed):,}")
    c2.metric("Unique hashtags", f"{analyzed['hashtags'].explode().nunique():,}")
    positive_count = int((analyzed["sentiment"] == "Positive").sum())
    c3.metric("Positive posts", f"{positive_count:,}")
    c4.metric("Negative posts", f"{int((analyzed['sentiment'] == 'Negative').sum()):,}")

    st.caption(f"Dataset: {source_name}")

    left, right = st.columns(2)
    with left:
        st.subheader("Top Hashtags")
        if hashtags.empty:
            st.info("No hashtags found. Add hashtags such as #AI or #Education to the text column.")
        else:
            top_n = st.slider("Number of hashtags to show", min_value=5, max_value=20, value=10)
            top_tags = hashtags.head(top_n)
            fig = px.bar(
                top_tags.sort_values("mentions"),
                x="mentions", y="hashtag", orientation="h",
                title="Hashtag mentions",
                labels={"mentions": "Mentions", "hashtag": "Hashtag"}
            )
            fig.update_layout(yaxis_title="", xaxis_title="Number of mentions")
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Sentiment Distribution")
        fig2 = px.pie(
            sentiment, names="sentiment", values="posts",
            hole=0.35, title="Posts by sentiment"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Hashtag Sentiment Breakdown")
    if not hashtags.empty:
        breakdown_cols = ["hashtag", "mentions", "Positive", "Neutral", "Negative"]
        fig3 = px.bar(
            hashtags.head(15), x="hashtag",
            y=["Positive", "Neutral", "Negative"],
            barmode="stack", title="Sentiment for the most-mentioned hashtags"
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(hashtags[breakdown_cols], use_container_width=True, hide_index=True)
    else:
        st.info("Hashtag breakdown is unavailable because no hashtags were detected.")

    st.subheader("Analyzed Posts")
    show_cols = [col for col in ["date", "platform", "text", "sentiment", "sentiment_score"] if col in analyzed.columns]
    st.dataframe(analyzed[show_cols], use_container_width=True, hide_index=True)

    csv_bytes = analyzed.drop(columns=["hashtags"]).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download analyzed posts as CSV",
        data=csv_bytes,
        file_name="analyzed_posts.csv",
        mime="text/csv"
    )

    st.caption("Note: VADER is a rule-based English sentiment tool. Sarcasm, slang, mixed-language posts, and context may be misclassified.")
except FileNotFoundError:
    st.error("sample_posts.csv was not found. Put it in the same folder as app.py, or upload your own CSV.")
except Exception as exc:
    st.error(f"Could not analyze this dataset: {exc}")
    st.write("Check that your CSV is valid and contains a column named `text`.")
