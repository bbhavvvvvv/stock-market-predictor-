#!/usr/bin/env python3
"""
Train and use a simple stock market direction predictor.

Expected CSV columns:
Date, Open, High, Low, Close, Volume

The model predicts whether the next closing price will be higher than the
current closing price.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "sma_5_ratio",
    "sma_10_ratio",
    "volatility_5d",
    "volume_change_1d",
    "high_low_range",
    "open_close_gap",
    "momentum_3d",
]


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"CSV is missing required columns: {missing_list}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()

    featured["return_1d"] = featured["Close"].pct_change(1)
    featured["return_5d"] = featured["Close"].pct_change(5)

    sma_5 = featured["Close"].rolling(5).mean()
    sma_10 = featured["Close"].rolling(10).mean()
    featured["sma_5_ratio"] = featured["Close"] / sma_5 - 1
    featured["sma_10_ratio"] = featured["Close"] / sma_10 - 1

    featured["volatility_5d"] = featured["Close"].pct_change().rolling(5).std()
    featured["volume_change_1d"] = featured["Volume"].pct_change(1)
    featured["high_low_range"] = (featured["High"] - featured["Low"]) / featured["Close"]
    featured["open_close_gap"] = (featured["Close"] - featured["Open"]) / featured["Open"]
    featured["momentum_3d"] = featured["Close"] / featured["Close"].shift(3) - 1

    featured["target"] = (featured["Close"].shift(-1) > featured["Close"]).astype(int)
    featured = featured.dropna().reset_index(drop=True)
    return featured


def time_series_split(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    split_index = int(len(df) * (1 - test_size))
    if split_index < 20 or len(df) - split_index < 5:
        raise ValueError(
            "Not enough rows after feature engineering. Provide more history "
            "or reduce test_size."
        )

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def train_model(data_path: Path, model_path: Path, metadata_path: Path, test_size: float) -> None:
    raw_df = load_data(data_path)
    featured_df = build_features(raw_df)
    train_df, test_df = time_series_split(featured_df, test_size=test_size)

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target"]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["target"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=4,
        random_state=42,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    latest_row = featured_df.iloc[-1]
    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "rows_used": int(len(featured_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "accuracy": float(accuracy),
        "latest_date": str(pd.Timestamp(latest_row["Date"]).date()),
        "latest_close": float(latest_row["Close"]),
        "latest_prediction_for_next_day": {
            "probability_up": float(model.predict_proba(featured_df[FEATURE_COLUMNS].tail(1))[0][1]),
            "predicted_direction": "UP"
            if int(model.predict(featured_df[FEATURE_COLUMNS].tail(1))[0]) == 1
            else "DOWN",
        },
        "classification_report": report,
        "recent_test_predictions": [
            {
                "date": str(pd.Timestamp(date).date()),
                "actual": int(actual),
                "predicted": int(predicted),
                "probability_up": float(probability_up),
            }
            for date, actual, predicted, probability_up in zip(
                test_df["Date"].tail(10),
                y_test.tail(10),
                predictions[-10:],
                probabilities[-10:],
            )
        ],
    }

    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Test accuracy: {accuracy:.3f}")
    print("Latest signal:")
    print(
        f"  {metadata['latest_date']} -> next day {metadata['latest_prediction_for_next_day']['predicted_direction']} "
        f"(probability up: {metadata['latest_prediction_for_next_day']['probability_up']:.3f})"
    )


def predict_latest(data_path: Path, model_path: Path) -> None:
    raw_df = load_data(data_path)
    featured_df = build_features(raw_df)

    if featured_df.empty:
        raise ValueError("No rows available after feature engineering.")

    model = joblib.load(model_path)
    latest_features = featured_df[FEATURE_COLUMNS].tail(1)
    latest_row = featured_df.iloc[-1]

    predicted_class = int(model.predict(latest_features)[0])
    probability_up = float(model.predict_proba(latest_features)[0][1])

    result = {
        "as_of_date": str(pd.Timestamp(latest_row["Date"]).date()),
        "close": float(latest_row["Close"]),
        "predicted_next_day_direction": "UP" if predicted_class == 1 else "DOWN",
        "probability_up": probability_up,
        "probability_down": 1 - probability_up,
    }
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stock market direction predictor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model from CSV data")
    train_parser.add_argument("--data", required=True, type=Path, help="Path to OHLCV CSV")
    train_parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("artifacts/stock_model.joblib"),
        help="Where to save the trained model",
    )
    train_parser.add_argument(
        "--metadata-out",
        type=Path,
        default=Path("artifacts/training_report.json"),
        help="Where to save training metadata",
    )
    train_parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of latest rows reserved for testing",
    )

    predict_parser = subparsers.add_parser("predict", help="Predict the next day direction")
    predict_parser.add_argument("--data", required=True, type=Path, help="Path to OHLCV CSV")
    predict_parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to a trained model file",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_model(
            data_path=args.data,
            model_path=args.model_out,
            metadata_path=args.metadata_out,
            test_size=args.test_size,
        )
    elif args.command == "predict":
        predict_latest(data_path=args.data, model_path=args.model)
    else:
        parser.error("Unknown command.")


if __name__ == "__main__":
    main()
