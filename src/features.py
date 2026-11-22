from pyspark.ml.feature import (RegexTokenizer, StopWordsRemover, NGram,
                                CountVectorizer, IDF, SQLTransformer)

# Sentiment-bearing words that must NOT be removed as stopwords.
KEEP = {
    "no", "nor", "not", "never", "cannot", "very", "too", "but", "against", "only",
    "don't", "doesn't", "didn't", "isn't", "wasn't", "weren't", "aren't", "won't",
    "wouldn't", "couldn't", "shouldn't", "hasn't", "haven't", "hadn't", "can't",
    "mustn't", "needn't", "shan't", "daren't", "mightn't",
    "don", "doesn", "didn", "isn", "wasn", "weren", "aren", "won", "wouldn",
    "couldn", "shouldn", "hasn", "haven", "hadn", "ain", "mustn", "needn", "shan", "mightn",
}


def custom_stopwords(cfg: dict) -> list[str]:
    defaults = StopWordsRemover.loadDefaultStopWords("english")
    if cfg["features"]["keep_negations"]:
        return [w for w in defaults if w not in KEEP]
    return defaults


def build_feature_stages(cfg: dict) -> list:
    f = cfg["features"]
    stages = [
        RegexTokenizer(inputCol="text_clean", outputCol="tokens", pattern="[^a-z']+", minTokenLength=2),
        StopWordsRemover(inputCol="tokens", outputCol="filtered", stopWords=custom_stopwords(cfg)),
    ]
    tok_col = "filtered"
    if f["use_bigrams"]:
        stages += [
            NGram(n=2, inputCol="filtered", outputCol="bigrams"),
            SQLTransformer(statement="SELECT *, concat(filtered, bigrams) AS all_tokens FROM __THIS__"),
        ]
        tok_col = "all_tokens"
    stages += [
        CountVectorizer(inputCol=tok_col, outputCol="tf", vocabSize=f["vocab_size"], minDF=f["min_df"]),
        IDF(inputCol="tf", outputCol="features", minDocFreq=f["min_df"]),
    ]
    return stages
