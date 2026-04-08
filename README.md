# Stock Market Predictor

This is a simple Python stock market predictor that learns from historical OHLCV data and predicts whether the next day's closing price will go up or down.

## What it does

- Reads historical stock data from a CSV file
- Creates technical features from price and volume history
- Trains a `RandomForestClassifier`
- Evaluates the model on the most recent holdout data
- Saves the trained model and a JSON training report
- Predicts the next day's direction from the latest available row

## CSV format

Your CSV must contain these columns:

```csv
Date,Open,High,Low,Close,Volume
2024-01-02,185.64,188.44,183.89,185.64,82488700
2024-01-03,184.22,185.88,183.43,184.25,58414500
```

The rows should represent daily historical data for one stock symbol.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Generate sample data

If you do not have market data yet, you can generate a synthetic dataset to test the pipeline:

```bash
python3 generate_sample_data.py --rows 300 --output sample_data.csv
```

## Train

```bash
python3 stock_predictor.py train --data data.csv
```

Optional outputs:

```bash
python3 stock_predictor.py train \
  --data data.csv \
  --model-out artifacts/stock_model.joblib \
  --metadata-out artifacts/training_report.json \
  --test-size 0.2
```

## Predict

```bash
python3 stock_predictor.py predict \
  --data data.csv \
  --model artifacts/stock_model.joblib
```

## Notes

- This predicts next-day direction, not exact price.
- It is a baseline ML project, not a production trading system.
- Real-world performance depends heavily on data quality, market regime, and feature design.
