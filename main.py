from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# تحميل موديل المشاعر
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# تعريف شكل الطلب
class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(data: TextInput):
    result = classifier(data.text)[0]
    return {
        "label": result["label"],
        "score": round(result["score"], 4)
    }
