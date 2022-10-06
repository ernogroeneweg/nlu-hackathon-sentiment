"""
Script containing helper functions for sentiment analysis
"""
import csv


def preprocess_roberta(text: str):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def corpus_from_csv(path: str) -> list[dict]:
    """
    read CSV file into list of dicts containing utterance and ground truth label
    :param path: filepath of CSV containing test data
    :return: list of dicts with utterance and ground truth label
    """
    result = []
    with open(path, "r", encoding="utf8") as infile:
        reader = csv.reader(infile, delimiter=";")
        for line in reader:
            line[0] = line[0].strip()
            if line[1].lower() == "boos":
                line[1] = "negative"
            elif line[1].lower() == "negatief":
                line[1] = "negative"
            elif line[1].lower() == "positief":
                line[1] = "positive"
            elif line[1].lower() == "neutraal":
                line[1] = "neutral"
            else:
                continue
            result.append({
                "sentence": line[0],
                "ground_truth": line[1]
            })
    return result
