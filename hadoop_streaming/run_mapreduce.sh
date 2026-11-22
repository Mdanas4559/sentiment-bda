#!/usr/bin/env bash
# Run from the repo root inside WSL, with HDFS and YARN started and src.mr_export already run.
# Do not run Spark at the same time: YARN needs the memory.
set -euo pipefail
: "${HADOOP_HOME:?set HADOOP_HOME to your Hadoop install directory}"
JAR=$(ls "$HADOOP_HOME"/share/hadoop/tools/lib/hadoop-streaming-*.jar | head -n 1)
BASE=/sentiment/mr
mkdir -p results/benchmarks
echo "job,seconds" > results/benchmarks/mr_runtime.csv

hdfs dfs -rm -r -f "$BASE/out_wordcount" "$BASE/out_product"

t0=$(date +%s)
hadoop jar "$JAR" \
  -files hadoop_streaming/mapper_wordcount.py,hadoop_streaming/reducer_sum.py \
  -input "$BASE/in_text" -output "$BASE/out_wordcount" \
  -mapper "python3 mapper_wordcount.py" \
  -combiner "python3 reducer_sum.py" \
  -reducer "python3 reducer_sum.py"
echo "wordcount,$(( $(date +%s) - t0 ))" >> results/benchmarks/mr_runtime.csv

t0=$(date +%s)
hadoop jar "$JAR" \
  -files hadoop_streaming/mapper_product.py,hadoop_streaming/reducer_product.py \
  -input "$BASE/in_rating" -output "$BASE/out_product" \
  -mapper "python3 mapper_product.py" \
  -reducer "python3 reducer_product.py"
echo "product_stats,$(( $(date +%s) - t0 ))" >> results/benchmarks/mr_runtime.csv

cat results/benchmarks/mr_runtime.csv
