import json
from math import sqrt
from tabulate import tabulate

with open("../data/results.json", "r") as infile:
    results = json.load(infile)


def accuracies_per_method():
    accuracies = []
    for item in results:
        luis_match = False
        roberta_match = False
        if item["ground_truth"] == item["luis"]["sentiment"]:
            luis_match = True
        ground = 0
        roberta_label = ""
        for key, val in item["roberta"].items():
            if val > ground:
                roberta_label = key
                ground = val
            else:
                continue

        if item["ground_truth"] == roberta_label:
            roberta_match = True

        accuracies.append(
            {
                "luis_match": luis_match,
                "roberta_match": roberta_match
            }
        )

    luis_matches = [item for item in accuracies if item["luis_match"]]
    roberta_matches = [item for item in accuracies if item["roberta_match"]]

    return [f"LUIS has an accuracy of {len(luis_matches)/len(accuracies)}",
            f"RoBERTa has an accuracy of {len(roberta_matches)/len(accuracies)}"]


def roberta_label_accuracies():
    label_accuracies = {
        "positive": {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "total": 0
        },
        "negative": {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "total": 0
        },
        "neutral": {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "total": 0
        }
    }
    for item in results:
        ground = 0
        roberta_label = ""
        for key, val in item["roberta"].items():
            if val > ground:
                roberta_label = key
                ground = val
            else:
                continue
        label_accuracies[item["ground_truth"]][roberta_label] += 1
        label_accuracies[item["ground_truth"]]["total"] += 1

    data = [
        ["positive", label_accuracies["positive"]["positive"], label_accuracies["positive"]["negative"],
         label_accuracies["positive"]["neutral"], _get_mcc("positive", label_accuracies)],
        ["negative", label_accuracies["negative"]["positive"], label_accuracies["negative"]["negative"],
         label_accuracies["negative"]["neutral"], _get_mcc("negative", label_accuracies)],
        ["neutral", label_accuracies["neutral"]["positive"], label_accuracies["neutral"]["negative"],
         label_accuracies["neutral"]["neutral"], _get_mcc("neutral", label_accuracies)],
    ]
    col_names = ["positive", "negative", "neutral", "mcc"]

    return tabulate(
        data,
        headers=col_names,
        tablefmt="fancy_grid"
    )


def luis_label_accuracies():
    label_accuracies = {
        "positive": {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "total": 0
        },
        "negative": {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "total": 0
        },
        "neutral": {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "total": 0
        }
    }
    for item in results:
        label_accuracies[item["ground_truth"]][item["luis"]["sentiment"]] += 1
        label_accuracies[item["ground_truth"]]["total"] += 1

    data = [
        ["positive", label_accuracies["positive"]["positive"], label_accuracies["positive"]["negative"],
         label_accuracies["positive"]["neutral"], _get_mcc("positive", label_accuracies)],
        ["negative", label_accuracies["negative"]["positive"], label_accuracies["negative"]["negative"],
         label_accuracies["negative"]["neutral"], _get_mcc("negative", label_accuracies)],
        ["neutral", label_accuracies["neutral"]["positive"], label_accuracies["neutral"]["negative"],
         label_accuracies["neutral"]["neutral"], _get_mcc("neutral", label_accuracies)],
    ]
    col_names = ["positive", "negative", "neutral", "MCC"]

    return tabulate(
        data,
        headers=col_names,
        tablefmt="fancy_grid"
    )


def _get_mcc(label: str, label_acc: dict):
    tp = label_acc[label][label]
    tn = 0
    for key, val in label_acc[label].items():
        if key != label:
            tn += val
    fp = 0
    fn = 0
    for key, val in label_acc.items():
        if key != label:
            for reskey, resval in label_acc[key].items():
                if reskey == label:
                    fp += resval
        elif key == label:
            for reskey, resval in label_acc[key].items():
                if reskey != label:
                    fn += resval

    return ((tp*tn)-(fp*fn))/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))


print(f"The results are in!",
      f"First, our overall accuracies:\n"
      f"{accuracies_per_method()[0]}\n"
      f"{accuracies_per_method()[1]}\n"
      f"RoBERTa's results per label are (rows are the ground truth):\n"
      f"{roberta_label_accuracies()}\n"
      f"And LUIS's results per label are (rows are the ground truth):\n"
      f"{luis_label_accuracies()}")
