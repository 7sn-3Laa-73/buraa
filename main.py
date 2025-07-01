# app.py
from fastapi import FastAPI, Request
from transformers import pipeline
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# تحميل الموديل
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# شكل البيانات اللي هتجيلك من Flutter
class InputText(BaseModel):
    text: str

@app.post("/predict")
async def predict(input: InputText):
    result = classifier(input.text)[0]
    return {
        "label": result['label'],
        "score": round(result['score'] * 100, 2)  # كنسبة مئوية
    }

# لتشغيل السيرفر محليًا
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
