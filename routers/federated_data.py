from fastapi import APIRouter

from SADL.federated_data.algorithms import get_algorithms as ga
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