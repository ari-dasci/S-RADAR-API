from inspect import signature
from fastapi import APIRouter, HTTPException, Request

from SADL.time_series.algorithms import get_algorithms as ga
#from SADL.time_series.algorithms.tods import tods_algorithms
from SADL.time_series.algorithms.tsfedl import tsfedl_algorithms
from SADL.time_series.algorithms import tsfedl as tsfedl
from SADL.time_series.time_series_datasets_uci import datasets
from SADL.time_series.preprocessing.preprocessing_ts import preprocessing_ts_algorithms
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



@router.get("/time_series/library_algorithms_tsfedl", tags=["time_series"])
async def get_tsfedl_algorithms():
    return list(tsfedl_algorithms.keys())

@router.get("/time_series/datasets", tags=["time_series"])
async def get_datasets():
    return list(datasets.keys())


@router.get("/time_series/preprocessing", tags=["time_series"])
async def get_preprocessing():
    return list(preprocessing_ts_algorithms)

# Obtain the default parameters of a specific algorithm
@router.get("/time_series/get_params/{_model}", tags=["time_series"])
async def get_params(_model: str):
    kwargs = {"algorithm_": _model}
    # Check if algorithm_ is present in any category
    if _model in tsfedl_algorithms:
        kwargs = obtener_parametros(_model, "tsfedl")
    return kwargs



@router.post("/time_series/set_params", tags=["time_series"])
async def set_params_post(request: Request):
    try:
        # Parse the JSON body of the request into a dictionary
        kwargs = await request.json()
        print(kwargs)

        # Check if algorithm_ is present in any category
        if kwargs["algorithm_"] in tsfedl_algorithms:
            model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        return model.get_params()
    except Exception as e:
        # Return the error message as JSON
        raise HTTPException(status_code=400, detail=str(e))
    

def obtener_parametros(_model: str, _type: str):
    if _type == 'tsfedl':
        init_signature = signature(tsfedl_algorithms[_model].__init__)
    
    # Consider the constructor parameters excluding 'self'
    parameters = [p for p in init_signature.parameters.values()
                  if p.name != 'self' and p.kind != p.VAR_KEYWORD]
    kwargs = {"algorithm_": _model}
    for p in parameters:
        if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            kwargs[p.name] = "REQUIRED"
        else:
            kwargs[p.name] = p.default
    print(kwargs)
    return kwargs