"""
Sentiment analysis pipeline with pretrained RoBERTa model
"""
import os
import shutil
import numpy as np
from dotenv import load_dotenv

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig

from scipy.special import softmax

from azure.cognitiveservices.language.luis.runtime import LUISRuntimeClient
from msrest.authentication import CognitiveServicesCredentials

from helpers import preprocess_roberta


# Prepare ROBERTA model
if os.path.isdir("./cardiffnlp/"):
    shutil.rmtree("./cardiffnlp/")
MODEL_NAME = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
CONFIG = AutoConfig.from_pretrained(MODEL_NAME)
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
if not os.path.isdir("./cardiffnlp/"):
    MODEL.save_pretrained(MODEL_NAME)


def roberta_sentiment(user_input: str) -> dict:
    text = preprocess_roberta(user_input)
    encoded_input = TOKENIZER(text, return_tensors='pt')
    output = MODEL(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    result = {}
    for i in range(scores.shape[0]):
        label = CONFIG.id2label[ranking[i]].lower()
        score = scores[ranking[i]]
        result[label] = np.round(float(score), 4)

    return result


def luis_sentiment(user_input: str) -> dict[str: any]:
    load_dotenv()
    prediction_key = os.getenv('PREDICTION_KEY')
    prediction_endpoint = os.getenv('PREDICTION_ENDPOINT')
    app_id = os.getenv('APP_ID')

    runtime_credentials = CognitiveServicesCredentials(prediction_key)
    runtime_client = LUISRuntimeClient(
        endpoint=prediction_endpoint,
        credentials=runtime_credentials
    )
    prediction_request = {"query": user_input}
    prediction_response = runtime_client.prediction.get_slot_prediction(
        app_id,
        "Staging",
        prediction_request
    )
    score = prediction_response.prediction.sentiment.score
    if score < 0.34:
        sentiment = "negative"
    elif score < 0.67:
        sentiment = "neutral"
    else:
        sentiment = "positive"
    return {
        "sentiment": sentiment,
        "score": score
    }


