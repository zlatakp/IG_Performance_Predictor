import pickle
from fastapi import FastAPI
from pathlib import Path
from .router import prediction_router



app = FastAPI(title="Post Performance Predictor")
app.include_router(prediction_router)
