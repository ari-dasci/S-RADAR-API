from fastapi import APIRouter

from SADL.static_data.algorithms import get_algorithms

router = APIRouter()

@router.get("/static_data/algorithms", tags=["static_data"])
async def get_algorithms():
    return get_algorithms()['components']
