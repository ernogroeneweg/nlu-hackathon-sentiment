from helpers import corpus_from_csv
from sentiment_analysers import (
    roberta_sentiment,
    luis_sentiment,
)
import json

FILEPATH = "../data/labeled_data.csv"


def get_sentiment_ranking(user_input: str, ground_truth: str) -> dict[str: dict]:
    return {
        "user_input": user_input,
        "ground_truth": ground_truth,
        "roberta": roberta_sentiment(user_input),
        "luis": luis_sentiment(user_input)
    }


# get corpus
corpus = corpus_from_csv(FILEPATH)

# run pipeline and get results
results = []
for sentence in corpus:
    results.append(
        get_sentiment_ranking(
            sentence["sentence"],
            sentence["ground_truth"]
        ))

with open("../data/results.json", "w") as outfile:
    json.dump(results, outfile, indent=2)
