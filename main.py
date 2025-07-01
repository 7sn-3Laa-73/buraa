from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

app = FastAPI()

# تحميل موديل التحليل من Hugging Face
sentiment_model = pipeline("sentiment-analysis")

class TextInput(BaseModel):
    text: str

@app.post("/analyze")
def analyze_sentiment(input: TextInput):
    result = sentiment_model(input.text)
    return result[0]  # بيرجع: {label: 'POSITIVE', score: 0.98}
