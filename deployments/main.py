import uvicorn
from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

model = load('../pipelines/StudentPerformance.joblib')


class InputData(BaseModel):
    hours_studied: float
    previous_scores: float
    sleep_hours: float
    sample_questions: float

@app.post("/predict")
def predict(input_data: InputData):
    x = pd.DataFrame(
        [[input_data.hours_studied, input_data.previous_scores, input_data.sleep_hours, input_data.sample_questions]],
        columns=["Hours Studied", "Previous Scores", "Sleep Hours",
                 "Sample Question Papers Practiced"])
    y_pred = model.predict(x)
    return {"prediction": int(y_pred[0])}




