from fastapi import FastAPI
from pydantic import BaseModel
from embed_classifier.predictor import Predictor

app = FastAPI()
predictor = Predictor()

class PromptRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: PromptRequest):
    try:
        label = predictor.predict(request.text)
        return {"label": int(label)}  
    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return {"error": str(e)}
