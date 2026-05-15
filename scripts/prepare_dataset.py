from pathlib import Path

import numpy as np
import pandas as pd


CLASS_ORDER = ["newbie", "casual", "engaged", "veteran"]
INPUT_PATH = Path("data/raw/steam_reviews_raw.csv")
OUTPUT_PATH = Path("data/processed/reviews_dataset.csv")
MIN_REVIEW_WORDS = 20
MIN_REVIEW_CHARS = 100


def experience_class_offline(hours):
    if hours < 10:
        return "newbie"
    if hours < 50:
        return "casual"
    if hours < 100:
        return "engaged"
    return "veteran"


def experience_class_competitive(hours):
    if hours < 100:
        return "newbie"
    if hours < 500:
        return "casual"
    if hours < 2000:
        return "engaged"
    return "veteran"


def experience_class(hours, game_group="offline"):
    if str(game_group).strip().lower() == "competitive":
        return experience_class_competitive(hours)
    return experience_class_offline(hours)


def clean_review_text(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def parse_bool(value):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return pd.NA


def load_and_prepare(input_path, min_review_words=5, min_review_chars=20):
    df = pd.read_csv(input_path)

    required_columns = {"recommendationid", "review", "playtime_forever"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {missing}")

    df = df.drop_duplicates(subset=["recommendationid"]).copy()
    df["review"] = df["review"].map(clean_review_text)
    df = df[df["review"].str.len() > 0].copy()

    for column in [
        "votes_up",
        "votes_funny",
        "weighted_vote_score",
        "comment_count",
        "author_num_games_owned",
        "author_num_reviews",
        "playtime_forever",
        "playtime_at_review",
        "timestamp_created",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    playtime_source = df["playtime_at_review"] if "playtime_at_review" in df.columns else np.nan
    df["target_playtime_minutes"] = playtime_source.fillna(df["playtime_forever"])
    df = df[df["target_playtime_minutes"].notna()].copy()
    df = df[df["target_playtime_minutes"] >= 0].copy()
    df["target_playtime_hours"] = df["target_playtime_minutes"] / 60.0
    if "game_group" not in df.columns:
        df["game_group"] = "offline"
    df["experience_class"] = df.apply(
        lambda row: experience_class(row["target_playtime_hours"], row["game_group"]),
        axis=1,
    )

    df["review_length_chars"] = df["review"].str.len()
    df["review_length_words"] = df["review"].str.split().str.len()
    df = df[
        (df["review_length_words"] >= min_review_words)
        & (df["review_length_chars"] >= min_review_chars)
    ].copy()
    if "timestamp_created" in df.columns:
        df["created_at"] = pd.to_datetime(df["timestamp_created"], unit="s", errors="coerce", utc=True)
        df["created_at"] = df["created_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    bool_columns = [
        "voted_up",
        "steam_purchase",
        "received_for_free",
        "written_during_early_access",
    ]
    for column in bool_columns:
        if column in df.columns:
            df[column] = df[column].map(parse_bool).astype("boolean")

    ordered_columns = [
        "appid",
        "game",
        "game_group",
        "recommendationid",
        "review",
        "experience_class",
        "target_playtime_hours",
        "target_playtime_minutes",
        "voted_up",
        "votes_up",
        "votes_funny",
        "weighted_vote_score",
        "comment_count",
        "steam_purchase",
        "received_for_free",
        "written_during_early_access",
        "author_num_games_owned",
        "author_num_reviews",
        "review_length_chars",
        "review_length_words",
        "timestamp_created",
        "created_at",
    ]
    return df[[column for column in ordered_columns if column in df.columns]].sort_values(
        ["game", "recommendationid"]
    )


def main():
    prepared = load_and_prepare(
        INPUT_PATH,
        min_review_words=MIN_REVIEW_WORDS,
        min_review_chars=MIN_REVIEW_CHARS,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(prepared)} prepared rows to {OUTPUT_PATH}")
    print("Class distribution:")
    print(prepared["experience_class"].value_counts().reindex(CLASS_ORDER, fill_value=0))


if __name__ == "__main__":
    main()
