from fastapi import APIRouter
from pydantic import BaseModel, Field, field_serializer, field_validator
from enum import IntEnum, StrEnum
from typing import Literal, List
import pandas as pd
from pathlib import Path
import pickle


with open(Path("src/postperf/artifacts")/"model_pipeline_best.pkl", "rb") as f:
    pipe = pickle.load(f)


router = APIRouter()



class PostType(StrEnum):
    reel = "IG reel"
    post = "IG image"
    carousel = "IG carousel"


class WeekDay(IntEnum):
    monday = 0
    tuesday = 1
    wednesday = 2
    thursday = 3
    friday = 4
    saturday = 5
    sunday = 6


weekdays = [day.name for day in WeekDay]
posttype = [ptype.name for ptype in PostType]


class Payload(BaseModel):
    post_type: Literal[*posttype] = Field(..., serialization_alias = 'Post type')
    duration: float = Field(..., ge=0, description='seconds', serialization_alias='Duration (sec)')
    desc_length: int = Field(..., ge=0, description="number of words", serialization_alias = 'Desc length') #need ig limit
    publish_hour: int = Field(ge=0, le=23, serialization_alias = 'Hour') #convert from CA
    day_of_week: Literal[*weekdays] = Field(..., serialization_alias = 'Day of Week')
        #"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    @field_serializer('day_of_week')
    def serialize_day_of_week(self, dow: str)-> int:
        return WeekDay[dow.lower()].value

    @field_serializer('post_type')
    def serialize_post_type(self, pt: str) -> str:
        return PostType[pt.lower()].value

    @field_validator('day_of_week', mode='before')
    def normalize_day(dow: str) -> str:
        return dow.lower()

class PerformanceData():
    def __init__(self, values: List[float]):
        self.views = values[0]
        self.likes = values[1]
        self.shares = values[2]
        self.comments = values[3]
        self.saves = values[4]
        self.reach = values[5]
        self.follows = values[6]


    def to_dict(self):
        return {"Views": self.views, "Likes": self.likes, "Shares": self.shares,
                "Comments": self.comments, "Saves": self.saves, "Reach": self.reach, "Follows": self.follows}

@router.get('/hello')
def hello():
    return {"status": "ok", "message": "Hello World"}

@router.post('/prediction')
def predict(p: Payload):
    x = pd.DataFrame([p.model_dump(by_alias=True)])
    y_pred = pipe.predict(x)[0]
    res = PerformanceData(y_pred).to_dict()
    return res