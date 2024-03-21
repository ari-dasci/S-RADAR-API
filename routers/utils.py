from fastapi import APIRouter

from SADL import get_components

router = APIRouter()

@router.get("/utils/blocks", tags=["utils"])
async def get_all_blocks():
    return get_components()
