import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_REQUIRED_COLUMNS = {"text"}

def load_posts(file_or_path):
    """Load a CSV and validate that it has a text column."""
    df = pd.read_csv(file_or_path)
    # Normalize column names so Text / text / TEXT are accepted.
    df.columns = [str(col).strip().lower() for col in df.columns]
    if "text" not in df.columns:
        raise ValueError("CSV must contain a column named 'text'.")
    df["text"] = df["text"].fillna("").astype(str)
    if "date" not in df.columns:
        df["date"] = ""
    if "platform" not in df.columns:
        df["platform"] = "Unknown"
    return df

def extract_hashtags(text):
    """Return normalized hashtags without the # symbol."""
    return [tag.lower() for tag in re.findall(r"(?<!\w)#([A-Za-z0-9_]+)", str(text))]

def analyze_posts(df):
    """Add sentiment labels and compound scores to a copy of the dataset."""
    result = df.copy()
    analyzer = SentimentIntensityAnalyzer()
    scores = result["text"].apply(lambda text: analyzer.polarity_scores(str(text)))
    result["sentiment_score"] = scores.apply(lambda item: round(item["compound"], 3))
    result["sentiment"] = result["sentiment_score"].apply(
        lambda score: "Positive" if score >= 0.05
        else ("Negative" if score <= -0.05 else "Neutral")
    )
    result["hashtags"] = result["text"].apply(extract_hashtags)
    return result

def hashtag_frequency(analyzed_df):
    """Build a table of hashtag counts and sentiment summaries."""
    rows = []
    for _, row in analyzed_df.iterrows():
        for hashtag in row["hashtags"]:
            rows.append({
                "hashtag": f"#{hashtag}",
                "sentiment": row["sentiment"],
                "sentiment_score": row["sentiment_score"],
                "date": row.get("date", "")
            })
    if not rows:
        return pd.DataFrame(columns=["hashtag", "mentions", "Positive", "Neutral", "Negative"])
    expanded = pd.DataFrame(rows)
    counts = expanded.groupby("hashtag").size().rename("mentions")
    sentiment_counts = pd.crosstab(expanded["hashtag"], expanded["sentiment"])
    summary = counts.to_frame().join(sentiment_counts, how="left").fillna(0)
    for col in ["Positive", "Neutral", "Negative"]:
        if col not in summary.columns:
            summary[col] = 0
    return summary.reset_index().sort_values("mentions", ascending=False)

def sentiment_summary(analyzed_df):
    order = ["Positive", "Neutral", "Negative"]
    counts = analyzed_df["sentiment"].value_counts()
    return pd.DataFrame({
        "sentiment": order,
        "posts": [int(counts.get(label, 0)) for label in order]
    })
