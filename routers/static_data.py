from fastapi import APIRouter
from fastapi import APIRouter, Request, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import json

from SADL.static_data import get_categories as gc
from SADL.static_data.algorithms import get_algorithms as ga
from SADL.static_data.algorithms.pyod import pyod_algorithms
from SADL.static_data.algorithms.sklearn import sklearn_algorithms

from SADL.static_data.algorithms import pyod
from SADL.static_data.algorithms import sklearn

router = APIRouter()

@router.get("/static_data/algorithms", tags=["static_data"])
async def get_algorithms():
    return ga()['components']

@router.get("/static_data/categories", tags=["static_data"])
async def get_categories():
    return gc()['categories']

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


@router.get("/static_data/get_params/{_model}", tags=["static_data"])
async def get_params(_model: str):
    kwargs = {"algorithm_": _model}
    model = pyod.PyodAnomalyDetection(**kwargs)
    return list(model.get_params())

@router.get("/static_data/set_params/", tags=["static_data"])
async def set_params(request: Request):
    try:
        kwargs = dict(request.query_params)  # parse ?key=value&key2=value2 into dict
        model = pyod.PyodAnomalyDetection(**kwargs)
        return model.get_params()
    except Exception as e:
        # Return the error message as JSON
        raise HTTPException(status_code=400, detail=str(e))