from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import json

from SADL.static_data.algorithms import get_algorithms as ga
from SADL.static_data.algorithms.pyod import pyod_algorithms
from SADL.static_data.algorithms.sklearn import sklearn_algorithms

router = APIRouter()

@router.get("/static_data/algorithms", tags=["static_data"])
async def get_algorithms():
    return ga()['components']

@router.get("/static_data/library_algorithms/{library}", tags=["static_data"])
async def get_library_algorithms(library: str):
    response = []
    if library == 'pyod':
        response = await get_pyod_algorithms()
    if library == 'sklearn':
        response = await get_sklearn_algorithms()
    return response


@router.get("/static_data/library_algorithms_pyod", tags=["static_data"])
async def get_pyod_algorithms():
    return list(pyod_algorithms.keys())

@router.get("/static_data/library_algorithms_sklearn", tags=["static_data"])
async def get_sklearn_algorithms():
    return list(sklearn_algorithms.keys())