from inspect import signature
from fastapi import APIRouter, HTTPException,Request

from SADL.federated_data.algorithms import get_algorithms as ga
from SADL.federated_data.algorithms import flexanomalies as flexanomalies
from SADL.federated_data.algorithms.flexanomalies import flexanomalies_algorithms

router = APIRouter()

@router.get("/federated_data/algorithms", tags=["federated_data"])
async def get_algorithms():
    return ga()['components']

@router.get("/federated_data/library_algorithms/{library}", tags=["federated_data"])
async def get_library_algorithms(library: str):
    response = []
    if library == 'flexanomalies':
        response = await get_flexanomalies_algorithms()
    return response

@router.get("/federated_data/library_algorithms_flexanomalies", tags=["federated_data"])
async def get_flexanomalies_algorithms():
    return list(flexanomalies_algorithms.keys())


# Obtain the default parameters of a specific algorithm
@router.get("/federated_data/get_params/{_model}", tags=["federated_data"])
async def get_params(_model: str):
    kwargs = {"algorithm_": _model}
    # Check if algorithm_ is present in any category
    if _model in flexanomalies_algorithms:
       kwargs = obtener_parametros(_model, "flexanomalies")
    return kwargs

@router.post("/federated_data/set_params", tags=["federated_data"])
async def set_params_post(request: Request):
    try:
        # Parse the JSON body of the request into a dictionary
        kwargs = await request.json()
        kwargs.pop("model", None)
        print(f"modificar parametros: {kwargs}")

        # Check if algorithm_ is present in any category
        if kwargs["algorithm_"] in flexanomalies_algorithms:
            model = flexanomalies.FlexAnomalyDetection(**kwargs)
        return model.get_params()
    except Exception as e:
        # Return the error message as JSON
        raise HTTPException(status_code=400, detail=str(e))
    


def obtener_parametros(_model: str, _type: str):
    if _type == 'flexanomalies':
        init_signature = signature(flexanomalies_algorithms[_model].__init__)
    
    # Consider the constructor parameters excluding 'self'
    parameters = [p for p in init_signature.parameters.values()
                  if p.name != 'self' and p.kind != p.VAR_KEYWORD]
    kwargs = {"algorithm_": _model}
    for p in parameters:
        if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            kwargs[p.name] = "REQUIRED"
        else:
            kwargs[p.name] = p.default
    if _type == 'flexanomalies':
        kwargs = {
        "algorithm_": "isolationForest",
        "contamination":0.1,
        "label_parser": None,
        "n_estimators": 100, 
        "n_rounds": 10,
        "n_clients":5,
        }
        model = flexanomalies.FlexAnomalyDetection(**kwargs)
        kwargs = model.get_params()
        print(f"obtener_parametros: {kwargs}")
    return kwargs