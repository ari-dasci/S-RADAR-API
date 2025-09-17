from inspect import signature
from fastapi import APIRouter, HTTPException, Request

from SADL.time_series.algorithms import get_algorithms as ga
from SADL.time_series.algorithms.transformers import transformers_algorithms
from SADL.time_series.algorithms.tsfedl import tsfedl_algorithms
from SADL.time_series.algorithms import tsfedl as tsfedl
from SADL.time_series.time_series_datasets_uci import datasets
from SADL.time_series.preprocessing.preprocessing_ts import preprocessing_ts_algorithms
from SADL.time_series.algorithms import transformers
import torch

class topModuleTDFEDL(torch.nn.Module):
    def __init__(self, in_features=103, out_features=103, npred=1):
        super(topModuleTDFEDL, self).__init__()
        self.npred = npred
        self.model = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=in_features, out_features=50),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=50, out_features=npred*out_features)
        )

    def forward(self, x):
        out = self.model(x)
        if len(out.shape)>2:
            out = out[:, -1, :]
        if self.npred > 1:
            # Reshape to (batch_size, npred, out_features)
            out = out.reshape(out.shape[0], self.npred, -1)

router = APIRouter()

@router.get("/time_series/algorithms", tags=["time_series"])
async def get_algorithms():
    return ga()['components']

@router.get("/time_series/library_algorithms/{library}", tags=["time_series"])
async def get_library_algorithms(library: str):
    response = []
    if library == 'tsfedl':
        response = await get_tsfedl_algorithms()
    elif library == 'transformers':
        response = await get_transformers_algorithms()
    return response



@router.get("/time_series/library_algorithms_tsfedl", tags=["time_series"])
async def get_tsfedl_algorithms():
    return list(tsfedl_algorithms.keys())


@router.get("/time_series/library_algorithms_transformers", tags=["time_series"])
async def get_transformers_algorithms():
    return list(transformers_algorithms.keys())


@router.get("/time_series/datasets", tags=["time_series"])
async def get_datasets():
    filtered_datasets = []
    for key,value in datasets.items():
        if 'url' not in value[0].__name__ and value[0].__name__!= 'load_gas_sensor_dataset':
            filtered_datasets.append(key)
    return filtered_datasets


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

        #Set or modify some specific parameters for tsfedl
        kwargs["in_features_topmodule"] = "REQUIRED"
        kwargs["out_features_topmodule"] = "REQUIRED"
        if "loss" in kwargs:
            kwargs["loss"] = "torch.nn.MSELoss()"
        if "optimizer" in kwargs:
            kwargs["optimizer"] = None
        if "input_shape" in kwargs:
            kwargs["input_shape"] = "(126,126)"
    elif _model in transformers_algorithms:
        kwargs = obtener_parametros(_model, "transformers")
        print(f"kwargs transformers: {kwargs}")
    return kwargs



@router.post("/time_series/set_params", tags=["time_series"])
async def set_params_post(request: Request):
    try:
        # Parse the JSON body of the request into a dictionary
        kwargs = await request.json()
        if kwargs.get("algorithm_") in tsfedl_algorithms:
            in_features_ = int(kwargs["in_features_topmodule"])
            out_features_ = int(kwargs["out_features_topmodule"])
            if "in_features_topmodule" in kwargs and "out_features_topmodule" in kwargs:
                kwargs["top_module"] = topModuleTDFEDL(in_features=in_features_, out_features=out_features_)
            if "in_features" in kwargs:
                kwargs["in_features"] = int(kwargs["in_features"])
            kwargs.pop("in_features_topmodule", None)
            kwargs.pop("out_features_topmodule", None)
            if "loss" in kwargs:
                kwargs["loss"] = torch.nn.MSELoss()
            if "input_shape" in kwargs:
                kwargs["input_shape"] = (126,126)
        elif kwargs.get("algorithm_") in transformers_algorithms:
            kwargs["label_parser"] = None
        print(f"kwargs before setting params: {kwargs}")

        # Check if algorithm_ is present in any category
        if kwargs["algorithm_"] in tsfedl_algorithms:
            model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        elif kwargs["algorithm_"] in transformers_algorithms:
            model = transformers.TransformersAnomalyDetection(**kwargs)
        #return model.get_params()
    except Exception as e:
        # Return the error message as JSON
        raise HTTPException(status_code=400, detail=str(e))
    

def obtener_parametros(_model: str, _type: str):
    if _type == 'tsfedl':
        init_signature = signature(tsfedl_algorithms[_model].__init__)
    elif _type == 'transformers':\
        init_signature = signature(transformers_algorithms[_model].__init__)
    
    # Consider the constructor parameters excluding 'self'
    parameters = [p for p in init_signature.parameters.values()
                  if p.name != 'self' and p.kind != p.VAR_KEYWORD]
    kwargs = {"algorithm_": _model}
    for p in parameters:
        if p.name == 'device':
            kwargs[p.name] = "cuda" if torch.cuda.is_available() else "cpu"
        elif p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            kwargs[p.name] = "REQUIRED"
        
        else:
            kwargs[p.name] = p.default
    print(kwargs)
    return kwargs