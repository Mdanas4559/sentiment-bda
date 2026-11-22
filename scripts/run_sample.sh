#!/usr/bin/env bash
# End-to-end run on the committed sample (no HDFS needed). Outputs go to data/lake_sample and results_sample/.
set -euo pipefail
cd "$(dirname "$0")/.."
export SENTIMENT_CONFIG=config/config.sample.yaml
mkdir -p data/lake_sample/raw
cp data/sample/reviews_sample.jsonl data/lake_sample/raw/reviews_sample.jsonl
cp data/sample/meta_sample.jsonl data/lake_sample/raw/meta_sample.jsonl
python -m src.ingest
python -m src.clean
python -m src.train
python -m src.predict
python -m src.aspects
python -m src.analytics
echo "Sample run finished: see results_sample/"
