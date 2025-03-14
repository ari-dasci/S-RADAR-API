from fastapi import APIRouter

from SADL.time_series.algorithms import get_algorithms as ga
#from SADL.time_series.algorithms.tods import tods_algorithms
from SADL.time_series.algorithms.tsfedl import tsfedl_algorithms
router = APIRouter()

@router.get("/time_series/algorithms", tags=["time_series"])
async def get_algorithms():
    return ga()['components']

@router.get("/time_series/library_algorithms/{library}", tags=["time_series"])
async def get_library_algorithms(library: str):
    response = []
#    if library == 'tods':
#        response = await get_tods_algorithms()
    if library == 'tsfedl':
        response = await get_tsfedl_algorithms()
    return response


#@router.get("/time_series/library_algorithms_tods", tags=["time_series"])
#async def get_tods_algorithms():
#    return list(tods_algorithms.keys())

@router.get("/time_series/library_algorithms_tsfedl", tags=["time_series"])
async def get_tsfedl_algorithms():
    return list(tsfedl_algorithms.keys())