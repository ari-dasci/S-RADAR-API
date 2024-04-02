from fastapi import APIRouter

from SADL.time_series.algorithms import get_algorithms

router = APIRouter()

@router.get("/time_series/algorithms", tags=["time_series"])
async def get_algorithms():
    return get_algorithms()['components']
