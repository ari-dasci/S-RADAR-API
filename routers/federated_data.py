from inspect import signature

import numpy as np
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




# @router.post("/federated_data/set_params", tags=["federated_data"])
# async def set_params_post(request: Request):
#     try:
#         # Parse the JSON body of the request into a dictionary
#         kwargs = await request.json()
#         kwargs.pop("model", None)
#         print(f"modificar parametros: {kwargs}")
#         if "input_dim" in kwargs:
#             default = np.random.rand(100, 1)
#             kwargs["input_dim"] = default.shape[1] # Special value for FlexAnomalies, only for pipeline to work
#             kwargs["hidden_act"]= ['relu', 'relu', 'relu']
#             kwargs["neurons"] =[16, 8, 16]

#         # Check if algorithm_ is present in any category
#         if kwargs["algorithm_"] in flexanomalies_algorithms:
#             model = flexanomalies.FlexAnomalyDetection(**kwargs)
#         return model.get_params()
#     except Exception as e:
#         # Return the error message as JSON
#         raise HTTPException(status_code=400, detail=str(e))
    
@router.post("/federated_data/set_params", tags=["federated_data"])
async def set_params_post(request: Request):
    try:
        kwargs = await request.json()
        kwargs.pop("model", None)
        # print(f"modificar parametros aaaaaa: {kwargs}")
        
        # if "input_dim" in kwargs:
        #     default = np.random.rand(100, 1)
        #     kwargs["input_dim"] = default.shape[1]
            
        #     # Asegurar que hidden_act y neurons sean listas
        #     # kwargs["hidden_act"] = ['relu', 'relu', 'relu']                    
        #     # kwargs["neurons"] = [16, 8, 16]
        
        # print(f"modificar parametros DESPUES: {kwargs}")

        # Check if algorithm_ is present in any category
        if kwargs["algorithm_"] in flexanomalies_algorithms:
            model = flexanomalies.FlexAnomalyDetection(**kwargs)
            params = model.get_params()
            
            # Convertir a JSON serializable
            serialized_params = convert_to_serializable(params)
            print('Seriaslized params:',serialized_params)
            return serialized_params
        
        return {}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    
# def obtener_parametros(_model: str, _type: str):
#     if _type == 'flexanomalies':
#         init_signature = signature(flexanomalies_algorithms[_model].__init__)
    
#     # Consider the constructor parameters excluding 'self'
#     parameters = [p for p in init_signature.parameters.values()
#                   if p.name != 'self' and p.kind != p.VAR_KEYWORD]
#     kwargs = {"algorithm_": _model, "n_clients": "REQUIRED", "n_rounds":"REQUIRED"}
#     for p in parameters:
#         if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
#             kwargs[p.name] = "REQUIRED"
            
#         else:
#             kwargs[p.name] = p.default
    
#     print(f"obtener_parametros: {kwargs}")
#     return kwargs

def obtener_parametros(_model: str, _type: str):
    if _type == 'flexanomalies':
        init_signature = signature(flexanomalies_algorithms[_model].__init__)
    
   
    fixed_params = {
        "neurons": [16, 8, 16],
        "hidden_act": ["relu", "relu", "relu"],
        "input_dim": 1,  # se obtine luego con X.shape[1]
        "filters_cnn": [8, 6],
        "units_lstm": [8,6],
        "kernel_size": [4,4]
    }
    
    kwargs = {"algorithm_": _model, "n_clients": "REQUIRED", "n_rounds":"REQUIRED"}
    
    for p in init_signature.parameters.values():
        if p.name == "self" or p.kind == p.VAR_KEYWORD:
            continue

        if p.name in fixed_params:
            kwargs[p.name] = fixed_params[p.name]  # valor fijo
        elif p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            kwargs[p.name] = "REQUIRED"
        else:
            kwargs[p.name] = p.default
    
    print(f"obtener_parametros: {kwargs}")
    return kwargs

def convert_to_serializable(obj):
    """Convierte objetos complejos a tipos serializables por JSON"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif hasattr(obj, '__dict__'):
        # For complex objects, extract their dictionary
        return str(obj)
    else:
        return obj