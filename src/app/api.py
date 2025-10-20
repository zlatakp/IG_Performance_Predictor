import joblib, pandas as pd
from fastapi import FastAPI
from pathlib import Path
from pydantic import BaseModel, Field, field_serializer, field_validator
from enum import IntEnum, StrEnum
from typing import Literal
from src.models.train import TARGETS

pipe = joblib.load(Path("src/models/artifacts")/"model_pipeline_best.pkl")

app = FastAPI(title="Post Performance Predictor")

class PostType(StrEnum):
    reel = "IG Reel"
    post = "Post"


class WeekDay(IntEnum):
    monday = 0
    tuesday = 1
    wednesday = 2
    thursday = 3
    friday = 4
    saturday = 5
    sunday = 6


weekdays = [day.name for day in WeekDay]


class Payload(BaseModel):
    post_type: PostType = Field(..., serialization_alias = 'Post type')
    duration: float = Field(..., ge=0, description='seconds', serialization_alias='Duration (sec)')
    desc_length: int = Field(..., ge=0, description="number of words", serialization_alias = 'Desc length') #need ig limit
    publish_hour: int = Field(ge=0, le=23, serialization_alias = 'Hour') #convert from CA
    day_of_week: Literal[*weekdays] = Field(..., serialization_alias = 'Day of Week')
        #"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    @field_serializer('day_of_week')
    def serialize_day_of_week(self, dow: str)-> int:
        return WeekDay[dow.lower()].value

    @field_validator('day_of_week', mode='before')
    def normalize_day(dow: str) -> str:
        return dow.lower()

@app.get('/hello')
def hello():
    return {"status": "ok", "message": "Hello World"}

@app.post('/predict')
def predict(p: Payload):
    print(p.model_dump(by_alias=True))
    x = pd.DataFrame([p.model_dump(by_alias=True)])
    y_pred = pipe.predict(x)[0]
    res = {TARGETS[k]: int(y_pred[k]) for k in range(len(TARGETS))}
    print(res)
    return res