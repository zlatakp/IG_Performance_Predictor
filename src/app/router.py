from fastapi import APIRouter
from .endpoints import router

prediction_router = APIRouter()
prediction_router.include_router(router, prefix='')